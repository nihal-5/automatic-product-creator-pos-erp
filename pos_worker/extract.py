"""Extract product data from a photograph using a vision model.

This is the light path. `ocr_pipeline.py` in this repo is the full stack --
YOLO for region proposal, EasyOCR and PaddleOCR for text, plus a vision model on
top. It is more precise on dense nutrition panels and small print, and it needs
PyTorch, which is roughly two gigabytes of dependency.

This module needs `openai` and `Pillow`. It sends the photograph to a vision
model and asks for the fields the POS actually stores. For a shop owner
photographing a product to create a catalogue entry, that is usually enough --
and it can run anywhere, including a laptop, without a GPU.

Start here. Move to the full pipeline when measured accuracy on real shelf
photographs says you need to, not before.

Everything below returns *what was read*. It never infers a price, a category,
or a unit -- that is the adapter's job on the POS side, and prices are a
commercial decision no camera can make.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ImageOps

MODEL = os.getenv("VISION_MODEL", "gpt-4o")

# Long edge in pixels. Vision models charge by image tiles, and product labels
# stay legible well below original camera resolution.
MAX_EDGE = int(os.getenv("VISION_MAX_EDGE", "1600"))

# Asks only for what the POS schema can store, and is explicit that unread
# fields must come back null. A model that guesses a plausible barcode is worse
# than one that admits it could not read the barcode.
PROMPT = """You are reading a photograph of a retail product to create a catalogue entry.

Return a single JSON object with exactly these keys:

  name                 the full product name as printed, including size or weight
  brand                the manufacturer or brand name as printed
  variant              flavour or variant, if the pack states one
  barcode              the barcode digits, only if clearly legible
  mrp                  the printed maximum retail price, digits only, no symbol
  net_weight           the net quantity as a number
  unit                 the unit for that quantity, exactly as printed (g, ml, kg, L)
  category             a broad retail category, e.g. Dairy & Eggs, Beverages, Snacks
  subcategory          a narrower one, e.g. Butter, Sparkling Water
  hsn_code             the HSN code, if printed
  gst_percent          the GST percentage, if printed
  long_description     one or two sentences describing the product
  ingredients          a list of ingredients, if legible
  nutrition_facts      a list of {label, value, unit} read from the nutrition panel

Rules:

- Report only what you can actually read on the pack. Use null for anything you
  cannot read, and an empty list where a list is expected.
- Never guess a barcode, an HSN code, or a price. A wrong number is worse than a
  missing one, because a missing one gets filled in by a person and a wrong one
  does not.
- Do not convert or calculate. If the pack says 227g, report 227 and "g".
- Read the nutrition panel exactly as printed, including its serving basis if
  stated, rather than normalising to a per-100g figure.

Return the JSON object and nothing else."""


def prepare_image(path: Path, max_edge: int = MAX_EDGE) -> str:
    """Load, orient and downscale an image, returning base64 JPEG.

    EXIF orientation matters: a phone photograph taken in portrait is stored
    rotated with an orientation flag, and a model reading the raw pixels sees a
    sideways label.
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        if max(img.size) > max_edge:
            scale = max_edge / max(img.size)
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)), Image.LANCZOS
            )
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _coerce(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise the model's output into the shape the POS adapter expects.

    The vision model returns what it read. The POS adapter
    (app/services/vision_intake_service.py) resolves references and enforces
    DRAFT status. Nothing here invents a value that was not on the pack.
    """
    def clean(key: str) -> Optional[Any]:
        value = payload.get(key)
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() in {"null", "none", "n/a", "unknown"}:
                return None
        return value

    facts = payload.get("nutrition_facts") or []
    if not isinstance(facts, list):
        facts = []

    return {
        "name": clean("name"),
        "brand": clean("brand"),
        "variant": clean("variant"),
        "barcode": clean("barcode"),
        "mrp": clean("mrp"),
        "net_weight": clean("net_weight"),
        "unit": clean("unit"),
        "category": clean("category"),
        "subcategory": clean("subcategory"),
        "hsn_code": clean("hsn_code"),
        "gst_percent": clean("gst_percent"),
        "long_description": clean("long_description"),
        "ingredients": payload.get("ingredients") or [],
        "nutrition_facts": [f for f in facts if isinstance(f, dict)],
    }


def extract(path: Path, *, model: str = MODEL) -> Dict[str, Any]:
    """Read one product photograph and return the extracted fields.

    Raises:
        RuntimeError: if the model returns something that is not JSON.
    """
    from openai import OpenAI  # imported here so the module loads without a key

    client = OpenAI()
    encoded = prepare_image(path)

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                ],
            }
        ],
    )

    raw = (response.choices[0].message.content or "").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Vision model did not return JSON: {raw[:200]}") from exc

    extracted = _coerce(payload)
    extracted["_source_image"] = path.name
    extracted["_model"] = model
    return extracted
