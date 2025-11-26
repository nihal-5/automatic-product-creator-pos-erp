#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter
from datetime import datetime, timezone
import os, sys, json, re, time, shutil, argparse, html
from typing import Any
from pdfminer.high_level import extract_text

# ==================== CONFIG ====================
# Put your OpenAI key once here (or leave blank and export OPENAI_API_KEY in env).
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
PRODUCT_API_URL = os.getenv("PRODUCT_API_URL", "").strip()
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
LLM_VALIDATION_ENDPOINT = os.getenv("LLM_VALIDATION_ENDPOINT", "").strip()
LLM_ENHANCE_ENDPOINT = os.getenv("LLM_ENHANCE_ENDPOINT", "").strip()
UTC = timezone.utc

def _rel_path(path: str|None) -> str|None:
    if not path:
        return None
    try:
        return os.path.relpath(path, OUT_DIR)
    except Exception:
        return path

def _copy_catalog_asset(source: str|None, key: str, view_label: str, assets_dir: Path) -> str|None:
    if not source or not os.path.exists(source):
        return _rel_path(source)
    ext = Path(source).suffix or ".jpeg"
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", key or "product")
    dest = assets_dir / f"{safe}_{view_label}{ext}"
    try:
        shutil.copyfile(source, dest)
        return os.path.relpath(dest, OUT_DIR)
    except Exception:
        return _rel_path(source)

BASE = Path(__file__).resolve().parent
IN_DIR = BASE / "input"
OUT_DIR = BASE / "output"
CROPS_DIR = OUT_DIR / "crops"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = OUT_DIR / "_tmp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
PRODUCTS_DIR = IN_DIR / "products"
BILLS_DIR = IN_DIR / "bills"
PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)
BILLS_DIR.mkdir(parents=True, exist_ok=True)

PRODUCT_FILTER = {s.strip().lower() for s in os.getenv("PRODUCT_FILTER", "").split(",") if s.strip()}


def _find_catalog_template() -> Path | None:
    env_template = os.getenv("CATALOG_TEMPLATE", "").strip()
    if env_template:
        candidate = Path(env_template)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        for root in (BASE, IN_DIR):
            alt = root / candidate
            if alt.exists():
                return alt
    candidate_names = [
        "catalog 3.29.19\u202fPM.html",
        "catalog 3.29.19 PM.html",
        "catalog_template.html",
        "catalog.html",
    ]
    seen: set[str] = set()
    for name in candidate_names:
        if not name or name in seen:
            continue
        seen.add(name)
        for root in (IN_DIR, BASE):
            candidate = root / name
            if candidate.exists():
                return candidate
    for root in (IN_DIR, BASE):
        matches = sorted(root.glob("catalog*.html"))
        if matches:
            return matches[0]
    return None


def _clean_path(path: str | None) -> str | None:
    if not path:
        return None
    return os.path.relpath(path, OUT_DIR) if os.path.isabs(path) else path


def _extract_barcode_value(data: dict) -> str | None:
    items = (data.get("barcode") or {}).get("items") or []
    for item in items:
        value = (item or {}).get("data")
        if value:
            return str(value)
    return None


def _normalize_views_for_api(views: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if isinstance(views, dict):
        iterator = views.items()
    elif isinstance(views, list):
        return [
            {
                "label": v.get("label"),
                "hero": _clean_path(v.get("hero")),
                "ocr": _clean_path(v.get("ocr")),
                "masked": _clean_path(v.get("masked")),
                "anchor": _clean_path(v.get("anchor")),
            }
            for v in views
            if isinstance(v, dict)
        ]
    else:
        return normalized

    for label, info in iterator:
        if not isinstance(info, dict):
            continue
        normalized.append(
            {
                "label": label,
                "hero": _clean_path(info.get("hero")),
                "ocr": _clean_path(info.get("ocr")),
                "masked": _clean_path(info.get("masked")),
                "anchor": _clean_path(info.get("anchor")),
            }
        )
    return normalized


def _build_product_payload(combined: dict) -> dict:
    summary = combined.get("summary") or {}
    views = combined.get("views") or {}
    views_list = _normalize_views_for_api(views)
    hero_image = None
    for view in views_list:
        if view.get("label") == "front" and view.get("hero"):
            hero_image = view["hero"]
            break
    if not hero_image and views_list:
        hero_image = views_list[0].get("hero")

    attributes = {
        "summary": summary,
        "dates": combined.get("dates"),
        "mrp": combined.get("mrp"),
        "ingredients": combined.get("ingredients"),
        "claims": combined.get("claims"),
        "certifications": combined.get("certifications"),
        "nutrition_facts": combined.get("nutrition_facts"),
        "other_text": combined.get("other_text"),
        "inventory": combined.get("inventory"),
        "link_report": combined.get("link_report"),
    }

    payload = {
        "slug": combined.get("id") or summary.get("slug") or _slugify(summary.get("product_name") or "product"),
        "name": summary.get("product_name") or combined.get("id") or "Unnamed Product",
        "brand": summary.get("brand"),
        "variant": summary.get("variant_or_flavor") or combined.get("variant_or_flavor"),
        "barcode": _extract_barcode_value(combined),
        "hero_image": hero_image,
        "attributes": {k: v for k, v in attributes.items() if v},
        "views": views_list,
        "embedding": combined.get("embedding"),
    }
    return payload


def _submit_product_to_api(combined: dict) -> None:
    if not PRODUCT_API_URL:
        return

    url = PRODUCT_API_URL.rstrip("/")
    if not url.endswith("/products"):
        url = url + "/products"

    payload = _build_product_payload(combined)
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code not in (200, 201):
            snippet = response.text[:200] if response.text else ""
            print(f"product_api_error status={response.status_code} detail={snippet}")
    except Exception as exc:
        print(f"product_api_error={exc}")

MAX_SIDE = int(os.getenv("MAX_SIDE", "2000"))
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8m-world.pt")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
YOLO_IOU  = float(os.getenv("YOLO_IOU", "0.45"))
YOLO_CLASSES = [c.strip() for c in os.getenv(
    "YOLO_CLASSES",
    "product,package,box,bottle,can,jar,bag,label,sticker,seal,logo,brand,brand name,"
    "nutrition facts,nutrition label,nutrition panel,table,ingredients,barcode,qr code"
).split(",") if c.strip()]
OCR_LANGS = [s.strip() for s in os.getenv("OCR_LANGS", "en").split(",") if s.strip()]

HERO_SIZE   = int(os.getenv("HERO_SIZE", "1024"))
CUT_PAD_PCT = float(os.getenv("CUT_PAD_PCT", "0.02"))
CUT_ITERS   = int(os.getenv("CUT_ITERS", "5"))

BRAND_NEGATIVE_RE = re.compile(
    r"(nutrition|facts|serving|distributed|store in|storage|quality|ingredients|daily value|per serving|calories|fat|sodium|carbohydrate|protein|address|website|directions|suggestion)",
    re.I,
)

# ===== Cost controls for the LLM call =====
# We will send: full image (always) + ONE "anchor crop" (if we can find) + optional masked image.
LLM_INCLUDE_MASKED  = os.getenv("LLM_INCLUDE_MASKED", "0") == "1"  # default off
LLM_IMAGE_DETAIL    = os.getenv("LLM_IMAGE_DETAIL", "low")         # low keeps token cost down

# Anchor crop heuristics (use OCR JSON to find MAR/LOT/EXP/MRP lines and expand around them)
ANCHOR_BAND_PX   = int(os.getenv("ANCHOR_BAND_PX", "70"))     # vertical half-band around anchor midline
ANCHOR_PAD       = int(os.getenv("ANCHOR_PAD", "40"))         # extra padding around the final union box
ANCHOR_MIN_WIDTH = int(os.getenv("ANCHOR_MIN_WIDTH", "260"))  # ensure crop is not too skinny
ANCHOR_MIN_WIDTH_RATIO = float(os.getenv("ANCHOR_MIN_WIDTH_RATIO", "0.18"))
ANCHOR_MIN_HEIGHT_RATIO = float(os.getenv("ANCHOR_MIN_HEIGHT_RATIO", "0.18"))
ANCHOR_EXTRA_PAD = int(os.getenv("ANCHOR_EXTRA_PAD", "30"))

AI_ENHANCE_FIELDS = [
    ("summary.product_name", "Product Name"),
    ("summary.brand", "Brand"),
    ("summary.variant_or_flavor", "Variant or Flavor"),
    ("summary.net_weight", "Net Quantity"),
    ("summary.serving_size", "Serving Size (Summary)"),
    ("summary.expiry", "Expiry (Summary)"),
    ("net_quantity.value", "Net Quantity Value"),
    ("net_quantity.unit", "Net Quantity Unit"),
    ("serving.serving_size", "Serving Size"),
    ("serving.servings_per_container", "Servings Per Container"),
    ("dates.best_by", "Best By"),
    ("dates.expiration", "Expiration"),
    ("dates.manufactured_on", "Manufactured On"),
    ("dates.lot", "Lot Code"),
    ("mrp.amount", "MRP Amount"),
    ("mrp.currency", "MRP Currency"),
    ("mrp.raw", "MRP Raw Text"),
    ("barcode.items[0].data", "Primary Barcode"),
    ("barcode.items[0].type", "Barcode Type"),
    ("inventory.stock", "Inventory Stock"),
    ("ingredients", "Ingredients"),
    ("claims", "Claims"),
    ("certifications", "Certifications"),
    ("other_text", "Other Text"),
]

AI_ENHANCE_CRITICAL_FIELDS = {
    "dates.best_by",
    "dates.expiration",
    "dates.manufactured_on",
    "dates.lot",
    "barcode.items[0].data",
    "barcode.items[0].type",
    "mrp.amount",
    "mrp.currency",
    "mrp.raw",
}

# ==================== HELPERS ====================
def _slugify(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", name.strip().lower()).strip("_")
    if not base:
        base = "product"
    return base

def _list_images_in_dir(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".heic", ".heif"}
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)

def collect_product_groups(max_images: int = 6) -> list[dict]:
    if not PRODUCTS_DIR.exists():
        sys.exit("Missing input/products/ folder")

    used_slugs: set[str] = set()
    groups: list[dict] = []

    subdirs = sorted(p for p in PRODUCTS_DIR.iterdir() if p.is_dir())
    for directory in subdirs:
        images = _list_images_in_dir(directory)
        if not images:
            continue
        if len(images) > max_images:
            raise SystemExit(f"Folder '{directory.name}' has {len(images)} images. Limit per product is {max_images}.")
        slug = _slugify(directory.name)
        base_slug = slug
        idx = 1
        while slug in used_slugs:
            slug = f"{base_slug}_{idx}"
            idx += 1
        used_slugs.add(slug)
        if PRODUCT_FILTER and slug.lower() not in PRODUCT_FILTER and directory.name.lower() not in PRODUCT_FILTER:
            continue
        groups.append({
            "name": directory.name,
            "slug": slug,
            "images": images,
            "source_dir": directory
        })

    root_images = _list_images_in_dir(PRODUCTS_DIR)
    # remove images that are inside directories (already handled)
    root_images = [p for p in root_images if p.parent == PRODUCTS_DIR]
    if root_images:
        if len(root_images) > max_images and not groups:
            raise SystemExit(f"Found {len(root_images)} images. Limit is {max_images}. Remove extras before running.")
        slug = _slugify(root_images[0].stem)
        base_slug = slug
        idx = 1
        while slug in used_slugs:
            slug = f"{base_slug}_{idx}"
            idx += 1
        used_slugs.add(slug)
        if not PRODUCT_FILTER or slug.lower() in PRODUCT_FILTER or root_images[0].stem.lower() in PRODUCT_FILTER:
            groups.append({
                "name": root_images[0].stem,
                "slug": slug,
                "images": sorted(root_images),
                "source_dir": PRODUCTS_DIR
            })

    if not groups:
        sys.exit("No product images found. Add images to input/products/ or its subfolders.")

    return groups

def resize_limit(pil, max_side):
    w, h = pil.size; ms = max(w, h)
    if ms <= max_side:
        return pil, 1.0
    s = max_side / ms
    return pil.resize((int(w*s), int(h*s)), Image.LANCZOS), s

def upscale_back(items, scale):
    if scale == 1.0 or not items:
        return
    s = 1.0 / scale
    for it in items:
        b = it["bbox"]
        b["x"] = int(b["x"] * s)
        b["y"] = int(b["y"] * s)
        b["w"] = int(b["w"] * s)
        b["h"] = int(b["h"] * s)

def clamp_box(x, y, w, h, W, H):
    x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
    return x, y, w, h

def _union_bbox(records, W: int, H: int, pad: int = 0):
    if not records:
        return None
    xs = []
    ys = []
    xe = []
    ye = []
    for rec in records:
        if not rec:
            continue
        if isinstance(rec, dict) and "bbox" in rec:
            bbox = rec["bbox"]
        else:
            bbox = rec
        if not isinstance(bbox, dict):
            continue
        x = int(bbox.get("x", 0))
        y = int(bbox.get("y", 0))
        w = int(bbox.get("w", 0))
        h = int(bbox.get("h", 0))
        if w <= 0 or h <= 0:
            continue
        xs.append(x)
        ys.append(y)
        xe.append(x + w)
        ye.append(y + h)
    if not xs:
        return None
    x0 = max(0, min(xs) - pad)
    y0 = max(0, min(ys) - pad)
    x1 = min(W, max(xe) + pad)
    y1 = min(H, max(ye) + pad)
    return clamp_box(x0, y0, x1 - x0, y1 - y0, W, H)

def _resolve_field_path(data: Any, path: str) -> Any:
    if data is None:
        return None
    cur = data
    tokens = path.split(".")
    for token in tokens:
        if not token:
            continue
        # Handle list indexes, e.g. items[0]
        while True:
            if "[" in token and token.endswith("]"):
                base, _, idx_part = token.partition("[")
                idx_part = idx_part[:-1]
                if base:
                    if isinstance(cur, dict):
                        cur = cur.get(base)
                    else:
                        return None
                if cur is None:
                    return None
                try:
                    idx = int(idx_part)
                except ValueError:
                    return None
                if not isinstance(cur, (list, tuple)) or idx >= len(cur) or idx < 0:
                    return None
                cur = cur[idx]
                token = ""
            else:
                break
        if not token:
            continue
        if isinstance(cur, dict):
            cur = cur.get(token)
        else:
            return None
    return cur

def _normalize_ai_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            norm = _normalize_ai_value(item)
            if norm is None:
                continue
            flattened.append(str(norm))
        return ", ".join(flattened) if flattened else None
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)
    text = str(value).strip()
    return text or None

def _collect_ai_snapshot(data: dict[str, Any] | None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for field_id, _label in AI_ENHANCE_FIELDS:
        snapshot[field_id] = _normalize_ai_value(_resolve_field_path(data or {}, field_id))
    return snapshot

def _safe_suggestion(field_id: str, expected: Any) -> Any:
    if expected is None:
        return None
    if field_id in AI_ENHANCE_CRITICAL_FIELDS:
        return None
    return expected

def _diff_ai_snapshots(display_snapshot: dict[str, Any], source_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for field_id, label in AI_ENHANCE_FIELDS:
        disp = display_snapshot.get(field_id)
        src = source_snapshot.get(field_id)
        if disp is None and src is None:
            continue
        if disp == src:
            continue
        diffs.append({
            "field": field_id,
            "label": label,
            "displayed": disp,
            "expected": src,
            "suggested": _safe_suggestion(field_id, src)
        })
    return diffs

def _build_diff_issues(diffs: list[dict[str, Any]]) -> dict[str, Any]:
    if not diffs:
        return {"summary": "No discrepancies detected between display and source JSON.", "issues": []}
    issues = []
    for item in diffs:
        issues.append({
            "field": item.get("field"),
            "label": item.get("label"),
            "displayed": item.get("displayed"),
            "expected": item.get("expected"),
            "suggested": item.get("suggested"),
            "reason": "Displayed value differs from source JSON."
        })
    summary = f"{len(issues)} field{'s' if len(issues) != 1 else ''} differ between display and source JSON."
    return {"summary": summary, "issues": issues}

def _enrich_ai_issues(issues: list[dict[str, Any]] | None, source_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    if not issues:
        return enriched
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        field_id = str(issue.get("field") or issue.get("id") or issue.get("path") or "").strip()
        if not field_id:
            continue
        suggested = issue.get("suggested")
        expected = issue.get("expected")
        src_value = source_snapshot.get(field_id)
        if expected is None and src_value is not None:
            issue["expected"] = src_value
        if suggested is None:
            issue["suggested"] = _safe_suggestion(field_id, src_value if src_value is not None else issue.get("expected"))
        else:
            issue["suggested"] = _safe_suggestion(field_id, suggested)
        issue.setdefault("label", dict(AI_ENHANCE_FIELDS).get(field_id, field_id))
        issue.setdefault("reason", "Displayed value differs from expected source data.")
        enriched.append(issue)
    return enriched


# ==================== DRAWING ====================
def draw_lines(pil, lines, out_path, color=(255,0,0), width=2):
    img = pil.copy().convert("RGB"); d = ImageDraw.Draw(img)
    try: font = ImageFont.load_default()
    except: font = None
    for it in lines:
        b=it["bbox"]; x,y,w,h = b["x"],b["y"],b["w"],b["h"]
        d.rectangle([x,y,x+w,y+h], outline=color, width=width)
        label = f"{it['id']}:{int(it.get('conf',0)*100)}"
        if font: tw,th = d.textbbox((0,0),label,font=font)[2:]
        else: tw,th = (len(label)*6,10)
        tx,ty = x+1, max(0, y - th - 6)
        d.rectangle([tx,ty,tx+tw+6,ty+th+4], fill=color)
        d.text((tx+3,ty+2), label, fill=(255,255,255), font=font)
    img.save(out_path, quality=95)
    return out_path

# ==================== MASKING (for output/preview; optional for LLM) ====================
def build_text_mask(im, boxes, pad=4, dilate=7):
    W,H = im.size
    mask = Image.new("L",(W,H),0); d = ImageDraw.Draw(mask)
    for b in boxes:
        x,y,w,h = b["x"],b["y"],b["w"],b["h"]
        x,y,w,h = clamp_box(x-pad,y-pad,w+2*pad,h+2*pad,W,H)
        d.rectangle([x,y,x+w,y+h], fill=255)
    if dilate and dilate>1:
        mask = mask.filter(ImageFilter.MaxFilter(size=(dilate//2)*2+1))
    return mask

def apply_mask_keep_text(im, text_mask, bg_color=(255,255,255)):
    bg = Image.new("RGB", im.size, bg_color)
    if text_mask.mode!="L": text_mask = text_mask.convert("L")
    return Image.composite(im, bg, text_mask)

# ==================== MODELS ====================
def run_yolo_world(pil):
    try:
        from ultralytics import YOLOWorld
    except Exception as e:
        return [], f"ultralytics not available: {e}"
    m=YOLOWorld(YOLO_MODEL)
    m.set_classes(YOLO_CLASSES)
    res=m.predict(pil, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
    dets=[]
    if res:
        r=res[0]; names=r.names if hasattr(r,"names") and isinstance(r.names,dict) else {}
        if getattr(r,"boxes",None) is not None:
            for b in r.boxes:
                x0,y0,x1,y1=[int(v) for v in b.xyxy[0].tolist()]
                cls_id=int(b.cls[0].item()) if b.cls is not None else -1
                name=names.get(cls_id,str(cls_id))
                conf=float(b.conf[0].item()) if b.conf is not None else 0.0
                dets.append({"cls":name,"conf":conf,"bbox":{"x":x0,"y":y0,"w":x1-x0,"h":y1-y0}})
    return dets, None

def run_easyocr(pil):
    try:
        import easyocr, numpy as np
    except Exception as e:
        return [], f"easyocr not available: {e}"
    reader=easyocr.Reader(OCR_LANGS, gpu=False, verbose=False)
    arr=ImageOps.exif_transpose(pil).convert("RGB")
    import numpy as _np; arr=_np.array(arr)
    res=reader.readtext(arr, detail=1, paragraph=False)
    lines=[]
    for i,(box,text,conf) in enumerate(res,1):
        xs=[int(p[0]) for p in box]; ys=[int(p[1]) for p in box]
        x0,y0,x1,y1=min(xs),min(ys),max(xs),max(ys)
        lines.append({"id":i,"text":str(text),"conf":float(conf),
                      "bbox":{"x":x0,"y":y0,"w":x1-x0,"h":y1-y0}})
    return lines, None

# ==================== BARCODE (ZXing-CPP) ====================
def decode_barcodes(pil):
    img = ImageOps.exif_transpose(pil).convert("RGB")

    def _pos_to_bbox(pos):
        pts=[]
        for attr in ("top_left","top_right","bottom_right","bottom_left"):
            if hasattr(pos, attr):
                p=getattr(pos, attr)
                if hasattr(p,"x") and hasattr(p,"y"):
                    pts.append((int(p.x), int(p.y)))
        if not pts and hasattr(pos,"points"):
            try:
                for p in pos.points:
                    if hasattr(p,"x") and hasattr(p,"y"):
                        pts.append((int(p.x), int(p.y)))
            except Exception:
                pass
        if not pts:
            return {"x":0,"y":0,"w":0,"h":0}
        xs=[x for x,_ in pts]; ys=[y for _,y in pts]
        x0,y0,x1,y1=min(xs),min(ys),max(xs),max(ys)
        return {"x":x0,"y":y0,"w":x1-x0,"h":y1-y0}

    try:
        import numpy as np, zxingcpp
        arr=np.array(img)
        results=zxingcpp.read_barcodes(arr)
        items=[]
        for r in results:
            fmt=getattr(r,"format",None)
            fmt_name=getattr(fmt,"name",None) or (str(fmt) if fmt is not None else None)
            pos=getattr(r,"position",None)
            bbox=_pos_to_bbox(pos) if pos is not None else {"x":0,"y":0,"w":0,"h":0}
            data_text = getattr(r,"text","") or ""
            if data_text:
                items.append({"type":fmt_name or "UNKNOWN","data":data_text,"bbox":bbox})
        if items:
            return {"note":None,"items":items}
        zxing_note = "no barcodes found"
    except Exception as e:
        zxing_note = f"zxing-cpp not available: {e}"

    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        import numpy as np
        arr = np.array(img)
        decoded = pyzbar_decode(arr)
        items=[]
        for d in decoded:
            data_bytes = getattr(d, "data", b"")
            try:
                data_text = data_bytes.decode("utf-8")
            except Exception:
                data_text = data_bytes.decode("latin-1", errors="ignore")
            rect = getattr(d, "rect", None)
            if rect:
                bbox = {"x": int(rect.left), "y": int(rect.top), "w": int(rect.width), "h": int(rect.height)}
            else:
                bbox = {"x":0,"y":0,"w":0,"h":0}
            items.append({"type": getattr(d, "type", None) or "UNKNOWN", "data": data_text, "bbox": bbox})
        note = None if items else zxing_note or "no barcodes found"
        return {"note":note, "items":items}
    except Exception as e:
        fallback_note = f"{zxing_note}; pyzbar decode error: {e}" if zxing_note else f"pyzbar decode error: {e}"
        return {"note":fallback_note, "items":[]}

# ==================== QUICK SUMMARY (light heuristics) ====================
def find_best(lines, W, H):
    text_all=" ".join(l["text"] for l in lines)
    expiry=None
    m=re.search(r"(best\s*before|use\s*by|expiry|exp\.?|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[^\n]{0,25}(\d{1,2}[\-/\s]\d{1,2}[\-/\s]\d{2,4}|\d{4}|\d{1,2}\s*[A-Za-z]{3,}\s*\d{2,4})", text_all, re.I)
    if m: expiry=m.group(0)
    brand=None
    for l in lines:
        t=l["text"].strip()
        if re.search(r"(distributed\s+by|brand|company|inc\.|llc|ltd)", t, re.I):
            brand=t; break
    if brand is None:
        top=[l for l in lines if l["bbox"]["y"]<H*0.25 and len(l["text"])<=30]
        if top: brand=sorted(top, key=lambda k: -len(k["text"]))[0]["text"]
    product=None
    mids=[l for l in lines if l["bbox"]["y"]<H*0.35 and not re.search(r"nutrition|facts|ingredients|calories", l["text"], re.I)]
    if mids: product=sorted(mids, key=lambda k:(k["bbox"]["y"],-len(k["text"])))[0]["text"]
    weight=None
    mw=re.search(r"(\d+(\.\d+)?\s*(g|kg|oz|lb|ml|l))", text_all, re.I)
    if mw: weight=mw.group(0)
    serving=None
    ms=re.search(r"(serving\s*size|per\s*serving)[^\n]{0,20}", text_all, re.I)
    if ms: serving=ms.group(0)
    return {"brand":brand,"product_name":product,"expiry":expiry,"net_weight":weight,"serving_size":serving}

# ==================== ANCHOR CROP FROM OCR ====================
MONTH_RE = re.compile(r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b", re.I)
KEY_RE   = re.compile(
    r"\b("
    r"BEST\s*BY|USE\s*BY|EXP|EXPI?RY|MFD|MFG|PKD|LOT|MRP|"
    r"BEST\s*IF\s*USED\s*BY|BEST\s*IF\s*USED|USED\s*BY"
    r")\b",
    re.I
)
NUM_RE   = re.compile(r"\b\d{1,4}\b")
DATE_VALUE_RE = re.compile(
    r"(?:(?:\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})|"
    r"(?:\b\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})|"
    r"(?:\b\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Za-z]*\s*,?\s*\d{2,4})|"
    r"(?:\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Za-z]*\s+\d{1,2},?\s*\d{2,4}))",
    re.I
)
PRICE_RE = re.compile(r"(₹|\$|£|€|Rs\.?|INR|USD|CAD|SGD|AUD)\s*[:\-]?\s*([\d,]+(?:\.\d+)?)", re.I)
LOT_CODE_RE = re.compile(r"\bLOT[:\-\s]*([A-Z0-9\-]+)", re.I)
GSTIN_RE = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b", re.I)
BARCODE_RE = re.compile(r"\b\d{12,14}\b")
QUANTITY_RE = re.compile(r"(?:qty|quantity|q\.?ty)\s*[:=]?\s*(\d+(?:\.\d+)?)", re.I)
PDF_EXTENSIONS = {".pdf"}

def _normalize_date_str(token: str) -> str|None:
    token = token.strip()
    if not token:
        return None
    token = re.sub(r"[^\w\s/.\-]", " ", token)
    token = re.sub(r"\s{2,}", " ", token).strip()
    patterns = [
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%y", "%d/%m/%y", "%d.%m.%y",
        "%m-%d-%Y", "%m/%d/%Y", "%m.%d.%Y", "%m-%d-%y", "%m/%d/%y", "%m.%d.%y",
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        "%d %b %Y", "%d %b %y", "%d %B %Y", "%d %B %y",
        "%b %d %Y", "%b %d %y", "%B %d %Y", "%B %d %y"
    ]
    for fmt in patterns:
        try:
            dt = datetime.strptime(token, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return None

def _extract_anchor_info(ocr_lines: list[dict]) -> dict:
    info = {
        "dates": {
            "best_by": None,
            "expiration": None,
            "manufactured_on": None,
            "lot": None,
            "raw_candidates": []
        },
        "mrp": {
            "amount": None,
            "currency": None,
            "raw": None
        }
    }
    seen_candidates = set()

    for line in ocr_lines or []:
        text = str(line.get("text","")).strip()
        if not text:
            continue
        lower = text.lower()

        if KEY_RE.search(text) or DATE_VALUE_RE.search(text):
            if text not in seen_candidates:
                info["dates"]["raw_candidates"].append(text)
                seen_candidates.add(text)

        date_match = DATE_VALUE_RE.search(text)
        normalized = _normalize_date_str(date_match.group(0)) if date_match else None

        if any(k in lower for k in ("best by","best before","use by","best if used")):
            if normalized and info["dates"]["best_by"] is None:
                info["dates"]["best_by"] = normalized
            elif info["dates"]["best_by"] is None:
                info["dates"]["best_by"] = date_match.group(0) if date_match else text

        if "exp" in lower or "expiry" in lower or "expiration" in lower:
            if normalized and info["dates"]["expiration"] is None:
                info["dates"]["expiration"] = normalized
            elif info["dates"]["expiration"] is None:
                info["dates"]["expiration"] = date_match.group(0) if date_match else text

        if any(k in lower for k in ("mfg","mfd","pkd","packed","manufactured")):
            if normalized and info["dates"]["manufactured_on"] is None:
                info["dates"]["manufactured_on"] = normalized
            elif info["dates"]["manufactured_on"] is None and date_match:
                info["dates"]["manufactured_on"] = date_match.group(0)

        lot_match = LOT_CODE_RE.search(text)
        if lot_match and info["dates"]["lot"] is None:
            info["dates"]["lot"] = lot_match.group(1).strip()

        price_match = PRICE_RE.search(text)
        if price_match and info["mrp"]["raw"] is None:
            currency = price_match.group(1).upper().replace("RS.", "Rs").replace("INR", "INR")
            amount = price_match.group(2).replace(",", "")
            info["mrp"] = {
                "amount": amount,
                "currency": currency,
                "raw": text
            }

    return info

def collect_supplier_bills() -> list[Path]:
    if not BILLS_DIR.exists():
        return []
    bills = [p for p in BILLS_DIR.iterdir()
             if p.is_file() and p.suffix.lower() in PDF_EXTENSIONS]
    return sorted(bills)

def _parse_supplier_bill(path: Path) -> dict:
    info = {
        "path": str(path),
        "gstins": [],
        "items": [],
        "errors": []
    }
    try:
        text = extract_text(str(path))
    except Exception as e:
        info["errors"].append(f"text extraction failed: {e}")
        return info

    if not text:
        info["errors"].append("no text extracted")
        return info

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    gstins = []
    for line in lines:
        for gst in GSTIN_RE.findall(line.upper()):
            gst_norm = gst.upper()
            if gst_norm not in gstins:
                gstins.append(gst_norm)

    item_map: dict[str, dict] = {}
    barcode_order: list[str] = []
    quantity_pool: list[float] = []
    capture_qty_block = False

    for idx, line in enumerate(lines):
        normalized = line.replace(",", "").strip()

        if re.match(r"^(qty|quantity|q\.?ty)\b", line, re.I):
            capture_qty_block = True
            continue
        if capture_qty_block:
            if re.fullmatch(r"\d+(?:\.\d+)?", normalized):
                try:
                    quantity_pool.append(float(normalized))
                    continue
                except ValueError:
                    pass
            capture_qty_block = False

        barcodes = BARCODE_RE.findall(line)
        if not barcodes:
            continue

        qty_value = None
        qty_match = QUANTITY_RE.search(line)
        if qty_match:
            try:
                qty_value = float(qty_match.group(1))
            except ValueError:
                qty_value = None
        if qty_value is None:
            for look_ahead in range(1, 3):
                if idx + look_ahead < len(lines):
                    m2 = QUANTITY_RE.search(lines[idx + look_ahead])
                    if m2:
                        try:
                            qty_value = float(m2.group(1))
                        except ValueError:
                            qty_value = None
                        break

        for bc in barcodes:
            entry = item_map.setdefault(
                bc,
                {"barcode": bc, "quantity": 0.0, "occurrences": 0, "unknown": False}
            )
            entry["occurrences"] += 1
            if bc not in barcode_order:
                barcode_order.append(bc)
            if qty_value is None:
                entry["unknown"] = True
            else:
                entry["quantity"] += qty_value

    # Assign quantities from pooled values for entries still unknown
    pool_idx = 0
    for bc in barcode_order:
        entry = item_map.get(bc)
        if not entry:
            continue
        if entry.get("quantity", 0) == 0 and not entry.get("unknown"):
            entry["unknown"] = True
        if entry.get("unknown") and pool_idx < len(quantity_pool):
            entry["quantity"] = quantity_pool[pool_idx]
            entry["unknown"] = False
            pool_idx += 1

    items = []
    for bc, entry in item_map.items():
        raw_qty = entry["quantity"]
        quantity = None if entry.get("unknown") else raw_qty
        if isinstance(quantity, (int, float)):
            if abs(quantity - round(quantity)) < 1e-6:
                quantity = int(round(quantity))
        items.append({
            "barcode": bc,
            "quantity": quantity,
            "occurrences": entry["occurrences"]
        })

    info["gstins"] = gstins
    info["items"] = items
    if not gstins:
        info["errors"].append("gstin not found")
    if not items:
        info["errors"].append("no barcodes found")
    return info

def parse_supplier_bills() -> list[dict]:
    bills = collect_supplier_bills()
    parsed = []
    for bill_path in bills:
        parsed.append(_parse_supplier_bill(bill_path))
    return parsed

def _anchor_crop_from_ocr(orig_img: Image.Image, ocr_lines: list) -> tuple[Image.Image|None, dict|None]:
    if not ocr_lines:
        return None, {"note":"no ocr lines"}
    W, H = orig_img.size

    # Normalize OCR rows
    rows=[]
    for l in ocr_lines:
        b=l["bbox"]; x,y,w,h = int(b["x"]),int(b["y"]),int(b["w"]),int(b["h"])
        rows.append({"text":str(l.get("text","")), "x":x, "y":y, "w":w, "h":h, "xm":x+w/2, "ym":y+h/2})

    # Find anchors by keyword or month
    anchors=[]
    for r in rows:
        txt = r["text"]
        if MONTH_RE.search(txt) or KEY_RE.search(txt):
            anchors.append(r)
            continue
        digits = re.findall(r"\d{2,4}", txt or "")
        if digits:
            if any(2020 <= int(d) <= 2035 for d in digits if len(d) == 4):
                anchors.append(r)
                continue
            if any(20 <= int(d) <= 35 for d in digits if len(d) == 2):
                anchors.append(r)
                continue
            if any(int(d) >= 1900 for d in digits if len(d) == 4):
                anchors.append(r)
                continue
    if not anchors:
        return None, {"note":"no anchors found"}

    # For each anchor, collect neighbors in a horizontal band; keep the widest union
    def band_union(a):
        y0 = max(0, int(a["ym"] - ANCHOR_BAND_PX))
        y1 = min(H, int(a["ym"] + ANCHOR_BAND_PX))

        def collect_rows(y_start, y_end, digits_only=False):
            band = [r for r in rows if y_start <= r["ym"] <= y_end]
            if not band:
                return []
            if digits_only:
                digits = [r for r in band if NUM_RE.search(r["text"])]
                if digits:
                    return digits
            return band

        band = collect_rows(y0, y1)
        if not band:
            return None

        numeric_rows = collect_rows(y0, y1, digits_only=True)
        max_expansion = int(H * 0.4)
        step = max(ANCHOR_BAND_PX // 2, 40)
        prev_digit_count = len(numeric_rows)

        while (y1 - y0) < max_expansion:
            if numeric_rows:
                extended = collect_rows(y0, min(H, y1 + step), digits_only=True)
                if extended and len(extended) > prev_digit_count:
                    y1 = min(H, y1 + step)
                    numeric_rows = extended
                    prev_digit_count = len(numeric_rows)
                    continue
                break
            y1 = min(H, y1 + step)
            numeric_rows = collect_rows(y0, y1, digits_only=True)
            prev_digit_count = len(numeric_rows)
            if numeric_rows:
                continue
        use = numeric_rows if numeric_rows else band
        if a not in use:
            use = use + [a]
        x0 = min(r["x"] for r in use)
        x1 = max(r["x"] + r["w"] for r in use)
        min_width = max(ANCHOR_MIN_WIDTH, int(W * ANCHOR_MIN_WIDTH_RATIO))
        w = max(min_width, x1 - x0)
        cx = (x0 + x1) // 2
        x0 = max(0, cx - w // 2)
        x1 = min(W, cx + w // 2)
        x0 = max(0, x0 - ANCHOR_PAD)
        x1 = min(W, x1 + ANCHOR_PAD)

        y0 = max(0, y0 - ANCHOR_PAD)
        y1 = min(H, y1 + ANCHOR_PAD)
        current_height = y1 - y0
        min_height = max(int(H * ANCHOR_MIN_HEIGHT_RATIO), current_height)
        if current_height < min_height:
            cy = (y0 + y1) // 2
            half = min_height // 2
            y0 = max(0, cy - half)
            y1 = min(H, cy + half)
        y0 = max(0, y0 - ANCHOR_EXTRA_PAD)
        y1 = min(H, y1 + ANCHOR_EXTRA_PAD)
        return [x0, y0, x1, y1]

    boxes=[]
    for a in anchors:
        b = band_union(a)
        if b: boxes.append(b)

    if not boxes:
        return None, {"note":"anchors had no band boxes"}

    # Merge all candidate boxes into a single union crop
    x0=min(b[0] for b in boxes); y0=min(b[1] for b in boxes)
    x1=max(b[2] for b in boxes); y1=max(b[3] for b in boxes)
    x0,y0,w,h = clamp_box(x0, y0, x1-x0, y1-y0, W, H)
    crop = orig_img.crop((x0,y0,x0+w,y0+h))
    meta={"note":None, "bbox":{"x":x0,"y":y0,"w":w,"h":h}, "anchors":len(anchors)}
    return crop, meta

# ==================== LLM SCHEMA & PROMPT ====================
SCHEMA = {
    "type":"object",
    "properties":{
        "brand":{"type":["string","null"]},
        "product_name":{"type":["string","null"]},
        "variant_or_flavor":{"type":["string","null"]},
        "net_quantity":{"type":"object","properties":{
            "value":{"type":["string","number","null"]},
            "unit":{"type":["string","null"]}
        }},
        "serving":{"type":"object","properties":{
            "serving_size":{"type":["string","null"]},
            "servings_per_container":{"type":["string","null"]}
        }},
        "barcode":{"type":"object","properties":{
            "value":{"type":["string","null"]},
            "type":{"type":["string","null"]}
        }},
        "dates":{"type":"object","properties":{
            "best_by":{"type":["string","null"]},
            "expiration":{"type":["string","null"]},
            "manufactured_on":{"type":["string","null"]},
            "lot":{"type":["string","null"]},
            "raw_candidates":{"type":"array","items":{"type":"string"}}
        }},
        "mrp":{"type":"object","properties":{
            "amount":{"type":["string","number","null"]},
            "currency":{"type":["string","null"]},
            "raw":{"type":["string","null"]}
        }},
        "ingredients":{"type":"array","items":{"type":"string"}},
        "allergen_info":{"type":["string","null"]},
        "facility_warning":{"type":["string","null"]},
        "claims":{"type":"array","items":{"type":"string"}},
        "certifications":{"type":"array","items":{"type":"string"}},
        "distributor":{"type":["string","null"]},
        "address":{"type":["string","null"]},
        "website":{"type":["string","null"]},
        "storage_instructions":{"type":["string","null"]},
        "item_number":{"type":["string","null"]},
        "nutrition_facts":{"type":"object","properties":{
            "calories":{"type":["string","number","null"]},
            "nutrients":{"type":"array","items":{
                "type":"object","properties":{
                    "name":{"type":"string"},
                    "amount":{"type":["string","null"]},
                    "dv_percent":{"type":["number","string","null"]}
                },"required":["name"]
            }}
        },"additionalProperties":True},
        "other_text":{"type":"array","items":{"type":"string"}},
        "evidence":{"type":"object","additionalProperties":{
            "type":"object","properties":{
                "source":{"type":"string","enum":["image","anchor_crop","masked","both","ocr","inferred"]},
                "notes":{"type":["string","null"]}
            }
        }}
    },
    "required":["product_name","brand","nutrition_facts"]
}

EXTRACTION_INSTRUCTIONS = """Task: Using the OCR-derived JSON snippets provided below, output ONE JSON object exactly matching the JSON Schema that follows.

Rules:
- Use the per-image JSON metadata (summary, raw OCR lines, barcode reads, anchor extraction) as authoritative evidence.
- Do not invent values. If uncertain, leave the field null and add any strings to dates.raw_candidates.
- Normalize dates to ISO (YYYY-MM-DD) when possible. Also capture 'lot'.
- Extract MRP/price if shown: mrp.amount (number), mrp.currency (symbol/code), mrp.raw (exact print).
- Nutrition facts: list every row with amount and %DV if shown.
- Ingredients: split into an array; keep parentheses.
- Claims/certifications (e.g., USDA Organic, Gluten-Free) should be included.
- Return STRICT JSON only, no extra text.

JSON Schema:
"""

def _build_llm_prompt(raw_json) -> str:
    if isinstance(raw_json, list):
        images=[]
        for idx,item in enumerate(raw_json,1):
            ocr=item.get("ocr",{})
            images.append({
                "index": idx,
                "file": item.get("file"),
                "summary": item.get("summary", {}),
                "barcode": item.get("barcode", {}),
                "ocr_line_count": ocr.get("line_count", 0)
            })
        compact={"image_count": len(raw_json), "images": images}
    else:
        ocr = raw_json.get("ocr", {})
        compact={"image_count": 1, "images":[{
            "index": 1,
            "file": raw_json.get("file"),
            "summary": raw_json.get("summary", {}),
            "barcode": raw_json.get("barcode", {}),
            "ocr_line_count": ocr.get("line_count", 0)
        }]}
    return EXTRACTION_INSTRUCTIONS + json.dumps(SCHEMA, ensure_ascii=False) + "\n---\nOCR_META:\n" + json.dumps(compact, ensure_ascii=False)

def structurize_with_gpt4o(items: list[dict], model: str = OPENAI_VISION_MODEL, output_name: str|None = None):
    if not items:
        return None, "no items provided"
    try:
        from openai import OpenAI
    except Exception as e:
        print(f"llm_note=openai sdk not installed: {e}")
        return None, "openai sdk not installed"

    api_key = OPENAI_API_KEY.strip()
    if not api_key or api_key == "PASTE_YOUR_OPENAI_KEY_HERE":
        return None, "api key missing"

    raws=[]
    payload=[]
    try:
        for idx,item in enumerate(items,1):
            raw_path=item["raw_json_path"]
            with raw_path.open("r", encoding="utf-8") as f:
                raw=json.load(f)
            raws.append(raw)
            payload.append({
                "index": idx,
                "file": raw.get("file"),
                "view_label": item.get("view_label"),
                "json": raw
            })
    except Exception as e:
        return None, f"io error: {e}"

    prompt = _build_llm_prompt(raws if len(raws)>1 else raws[0])
    prompt += "\n---\nRAW_INPUT_JSONS:\n" + json.dumps(payload, ensure_ascii=False)
    client = OpenAI(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":"You transcribe and structure packaging text with extreme care. Return STRICT JSON only."},
                {"role":"user","content":[{"type":"text","text":prompt}]},
            ],
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
    except Exception as e:
        print(f"llm_note=error calling GPT-4o: {e}")
        return None, f"error: {e}"

    if output_name:
        out = OUT_DIR / output_name
    elif len(items) == 1:
        out = OUT_DIR / f"{items[0]['img_path'].stem}_structured.json"
    else:
        out = OUT_DIR / f"{items[0]['img_path'].stem}_combined_structured.json"

    try:
        with out.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        label = "structured_json" if len(items) == 1 else "combined_structured_json"
        print(f"{label}={out.resolve()}")
    except Exception as e:
        print(f"llm_note=write error: {e}")
    return data, None

def _generate_ai_enhance_report(displayed: dict, structured: dict | None) -> dict | None:
    display_snapshot = _collect_ai_snapshot(displayed)
    source_snapshot = _collect_ai_snapshot(structured or {})
    diffs = _diff_ai_snapshots(display_snapshot, source_snapshot)
    fallback = _build_diff_issues(diffs)
    fallback["issues"] = _enrich_ai_issues(fallback.get("issues"), source_snapshot)
    if not structured:
        fallback.setdefault("source", "baseline_missing")
        return fallback
    if not diffs:
        fallback.setdefault("source", "diff")
        fallback.setdefault("summary", "No discrepancies detected between display and source JSON.")
        return fallback

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        print(f"ai_enhance_note=openai sdk not installed: {exc}")
        fallback.setdefault("source", "diff")
        return fallback

    api_key = OPENAI_API_KEY.strip()
    if not api_key or api_key == "PASTE_YOUR_OPENAI_KEY_HERE":
        fallback.setdefault("source", "diff")
        return fallback

    client = OpenAI(api_key=api_key)
    analysis_payload = {
        "displayed_fields": display_snapshot,
        "source_fields": source_snapshot,
        "diff_candidates": diffs[:20]
    }

    instruction = (
        "You are auditing a product catalog entry.\n"
        "You will receive JSON with displayed_fields (current UI values), source_fields (values from OCR/LLM extraction),\n"
        "and diff_candidates (fields that differ).\n"
        "Decide which displayed values are incorrect or risky. For each problematic field, provide:\n"
        "- field: the field id exactly as given in diff_candidates\n"
        "- label: a readable label for the field\n"
        "- displayed: the incorrect value (string, number, or null)\n"
        "- expected: the best available correction from source_fields (or null if unknown)\n"
        "- reason: short explanation why the displayed value is wrong\n"
        "Return strict JSON with keys 'summary' and 'issues'.\n"
        "'issues' must be an array (possibly empty).\n"
        "If everything looks correct, return an empty issues array and a reassuring summary."
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a meticulous product data QA assistant and must respond with strict JSON."},
                {"role": "user", "content": instruction},
                {"role": "user", "content": json.dumps(analysis_payload, ensure_ascii=False)},
            ],
        )
        content = resp.choices[0].message.content if resp.choices else None
        if not content:
            fallback.setdefault("source", "diff")
            return fallback
        report = json.loads(content)
        if not isinstance(report, dict):
            fallback.setdefault("source", "diff")
            return fallback
        summary = report.get("summary") or fallback.get("summary")
        issues_raw = report.get("issues")
        issues = _enrich_ai_issues(issues_raw if isinstance(issues_raw, list) else [], source_snapshot)
        if not issues:
            issues = fallback.get("issues", [])
        return {
            "summary": summary,
            "issues": issues,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "source": "openai"
        }
    except Exception as exc:
        print(f"ai_enhance_note=openai_error: {exc}")
        fallback.setdefault("source", "diff")
        return fallback

# ==================== HERO HELPERS ====================
_PRODUCT_LIKE = {"product","package","box","bottle","can","jar","bag","label"}

def _pick_best_product_box(dets):
    if not dets: return None
    scored=[]
    for d in dets:
        name=(d.get("cls") or "").lower()
        is_prod = 1 if any(k in name for k in _PRODUCT_LIKE) else 0
        b=d["bbox"]; area=b["w"]*b["h"]
        scored.append((is_prod, d.get("conf",0.0), area, d))
    scored.sort(key=lambda t:(t[0],t[1],t[2]), reverse=True)
    return scored[0][3] if scored else None

def _square_from_rect(x,y,w,h,W,H,scale=1.07):
    cx,cy = x+w/2, y+h/2
    side=int(max(w,h)*scale)
    sx=int(cx-side/2); sy=int(cy-side/2)
    return clamp_box(sx,sy,side,side,W,H)

def _make_hero_from_box(pil, det, size=HERO_SIZE, white=(255,255,255)):
    W,H=pil.size
    b=det["bbox"]; x,y,w,h=b["x"],b["y"],b["w"],b["h"]
    sx,sy,sw,sh=_square_from_rect(x,y,w,h,W,H,scale=1.07)
    crop=pil.crop((sx,sy,sx+sw,sy+sh))
    canvas=Image.new("RGB",(size,size),white)
    canvas.paste(crop.resize((size,size), Image.LANCZOS),(0,0))
    return canvas

def _cutout_on_white(pil, det, size=HERO_SIZE, white=(255,255,255)):
    W,H = pil.size
    b = det["bbox"]; x,y,w,h = b["x"], b["y"], b["w"], b["h"]
    pad = int(CUT_PAD_PCT * max(w, h))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)

    # Build rectangular mask and composite with a white background
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x0, y0, x1, y1], fill=255)
    white_bg = Image.new("RGB", (W, H), white)
    boxed = Image.composite(pil, white_bg, mask)

    crop = boxed.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    if cw == 0 or ch == 0:
        return None, "cutout skipped (empty crop)"

    scale = min(size / cw, size / ch)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    resized = crop.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (size, size), white)
    off_x = (size - new_w) // 2
    off_y = (size - new_h) // 2
    canvas.paste(resized, (off_x, off_y))
    return canvas, None

# ==================== ORDERING HELPERS ====================
def _brand_signal_score(data: dict) -> float:
    ocr = data.get("ocr") or {}
    lines = ocr.get("lines") or []
    if not lines:
        return 0.0
    img_sz = data.get("image_size") or {}
    W = max(1, int(img_sz.get("width") or 1))
    H = max(1, int(img_sz.get("height") or 1))
    base_area = float(W * H)
    score = 0.0
    for line in lines:
        text = str(line.get("text", "")).strip()
        if not text:
            continue
        if BRAND_NEGATIVE_RE.search(text):
            continue
        lower_text = text.lower()
        bbox = line.get("bbox") or {}
        w = max(1, int(bbox.get("w", 1)))
        h = max(1, int(bbox.get("h", 1)))
        area_ratio = (w * h) / base_area
        if area_ratio <= 0:
            continue
        y_mid = bbox.get("y", 0) + h / 2.0
        vertical_weight = 1.5 if y_mid <= H * 0.55 else 1.0
        length_weight = min(len(text), 40) / 10.0
        uppercase_bonus = 2.0 if re.search(r"[A-Z]{3,}", text) else 0.0
        strong_brand_bonus = 0.0
        for keyword in ("brooke", "bond", "red", "label"):
            if keyword in lower_text:
                strong_brand_bonus += 1.2
        score += area_ratio * 520.0 * vertical_weight + length_weight + uppercase_bonus + strong_brand_bonus

        # consider barcode presence as penalty
        if "barcode" in lower_text:
            score -= 12.0
    return score

def _front_score(data: dict) -> float:
    summary = data.get("summary") or {}
    ocr = data.get("ocr") or {}
    barcode = data.get("barcode") or {}
    area_ratio = data.get("product_area_ratio") or 0.0
    text_metrics = data.get("text_metrics") or {}
    score = 0.0
    brand = (summary.get("brand") or "").strip()
    product = (summary.get("product_name") or "").strip()
    if area_ratio:
        score += min(area_ratio * 110.0, 36.0)
    max_text_area_ratio = text_metrics.get("max_text_area_ratio", 0.0)
    total_text_area_ratio = text_metrics.get("total_text_area_ratio", 0.0)
    if max_text_area_ratio:
        score += min(max_text_area_ratio * 260.0, 68.0)
    if total_text_area_ratio:
        score += min(total_text_area_ratio * 220.0, 72.0)

    brand_clean = brand.lower()
    product_clean = product.lower()
    if brand and not BRAND_NEGATIVE_RE.search(brand):
        score += min(len(brand), 32) * 0.8
        if re.search(r"\b(red|brooke|bond|label)\b", brand_clean):
            score += 12.0
    if product and not BRAND_NEGATIVE_RE.search(product):
        score += min(len(product), 32) * 0.9

    score += _brand_signal_score(data)

    line_count = text_metrics.get("line_count", ocr.get("line_count", 0) or 0)
    if line_count > 30:
        score -= (line_count - 30) * 0.9
    if line_count > 46:
        score -= 24.0

    nutrition_penalty = _nutrition_signal_score(data) * 4.2
    score -= nutrition_penalty

    forbidden = ("exp", "expiry", "expiration", "best if used", "best by", "used by")
    if any(k in (product + " " + brand).lower() for k in forbidden):
        score -= 40.0

    texts_blob = " ".join(str(line.get("text", "")).lower() for line in (ocr.get("lines") or [])[:12])
    if "nutrition facts" in texts_blob or "serving size" in texts_blob:
        score -= 90.0

    if barcode.get("items"):
        score -= 8.0

    return score

def _has_hero_output(res: dict) -> bool:
    data = res.get("data") or {}
    outputs = data.get("outputs") or {}
    return bool(outputs.get("hero_cutout_image") or outputs.get("hero_image"))

def _info_score(data: dict) -> float:
    summary = data.get("summary") or {}
    ocr = data.get("ocr") or {}
    barcode = data.get("barcode") or {}
    anchor = data.get("anchor_extraction") or {}
    dates = anchor.get("dates") or {}
    mrp = anchor.get("mrp") or {}
    text_len = len((summary.get("product_name") or "")) + len((summary.get("brand") or "")) + len((summary.get("serving_size") or ""))
    line_count = ocr.get("line_count", 0) or 0
    barcode_count = len(barcode.get("items") or [])
    date_signals = len(dates.get("raw_candidates") or [])
    price_signal = 1 if mrp.get("amount") else 0
    return text_len/10.0 + line_count + barcode_count*3 + date_signals*2 + price_signal*4

_EXP_KEYWORDS = ("exp", "expiry", "expiration", "use by", "used by", "best if used", "best by", "use before")

def _nutrition_signal_score(data: dict) -> int:
    if not data:
        return 0
    score = 0
    def _score_text(text: str) -> int:
        if not text:
            return 0
        text = text.lower()
        strong_terms = ("nutrition facts", "nutrition information", "nutritional facts")
        common_terms = ("calories", "total fat", "saturated fat", "trans fat", "cholesterol",
                        "sodium", "total carbohydrate", "dietary fiber", "total sugars",
                        "added sugars", "protein", "vitamin", "minerals", "iron", "calcium",
                        "serving size", "per serving", "daily value", "% dv", "%dv")
        score = 0
        for term in strong_terms:
            if term in text:
                score += 3
        for term in common_terms:
            if term in text:
                score += 1
        return score

    summary = data.get("summary") or {}
    for key in ("product_name", "brand", "serving_size"):
        val = summary.get(key)
        if isinstance(val, str):
            score += _score_text(val)

    ocr = data.get("ocr") or {}
    for line in ocr.get("lines") or []:
        txt = line.get("text")
        if isinstance(txt, str):
            score += _score_text(txt)
    return score

def _is_exp_panel(data: dict) -> bool:
    if not data:
        return False
    summary = data.get("summary") or {}
    text_bits = []
    for key in ("brand", "product_name", "serving_size", "expiry", "net_weight"):
        val = summary.get(key)
        if isinstance(val, str):
            text_bits.append(val.lower())
    anchor = data.get("anchor_extraction") or {}
    dates = anchor.get("dates") or {}
    raw_candidates = dates.get("raw_candidates") or []
    for cand in raw_candidates:
        text_bits.append(str(cand).lower())
    ocr = data.get("ocr") or {}
    for line in ocr.get("lines") or []:
        txt = str(line.get("text","")).lower()
        text_bits.append(txt)
    blob = " ".join(text_bits)
    return any(k in blob for k in _EXP_KEYWORDS)

def _is_nutrition_panel(data: dict) -> bool:
    score = _nutrition_signal_score(data)
    if score < 8:
        return False
    tm = (data or {}).get("text_metrics") or {}
    line_count = tm.get("line_count") or (data.get("ocr") or {}).get("line_count", 0)
    max_ratio = tm.get("max_text_area_ratio", 0.0)
    area_ratio = data.get("product_area_ratio") or 0.0
    if line_count < 12:
        return False
    if max_ratio >= 0.035:
        return False
    if area_ratio >= 0.05:
        return False
    return True

def _front_candidate_rank(res: dict):
    data = res.get("data") or {}
    tm = data.get("text_metrics") or {}
    return (
        tm.get("max_text_area_ratio", 0.0),
        tm.get("total_text_area_ratio", 0.0),
        tm.get("top_char_count", 0),
        tm.get("headline_len", 0),
        res.get("area_ratio", 0.0)
    )

def _ensure_inventory_block(data: dict) -> dict:
    inv = data.get("inventory")
    if not isinstance(inv, dict):
        inv = {}
        data["inventory"] = inv
    inv.setdefault("stock", 0)
    inv.setdefault("supplier_links", [])
    return inv

def link_supplier_data(processed: list[dict], bills: list[dict]) -> dict:
    barcode_index: dict[str, list[dict]] = defaultdict(list)
    for res in processed:
        product_data = res.get("data") or {}
        for item in (product_data.get("barcode") or {}).get("items", []):
            code = (item.get("data") or "").strip()
            if code:
                barcode_index[code].append(res)

    report = {"linked": 0, "unmatched": [], "errors": []}
    for bill in bills:
        gstins = bill.get("gstins") or []
        bill_path = bill.get("path")
        if bill.get("errors"):
            report["errors"].append({"path": bill_path, "errors": bill["errors"]})
        for item in bill.get("items") or []:
            barcode = item.get("barcode")
            if not barcode:
                continue
            matches = barcode_index.get(barcode)
            if not matches:
                report["unmatched"].append({"barcode": barcode, "bill_path": bill_path})
                continue
            quantity = item.get("quantity")
            occurrences = item.get("occurrences")
            for res in matches:
                data = res.get("data") or {}
                inv = _ensure_inventory_block(data)
                link_entry = {
                    "supplier_gstin": gstins[0] if gstins else None,
                    "all_gstins": gstins,
                    "bill_path": bill_path,
                    "quantity": quantity,
                    "occurrences": occurrences
                }
                inv["supplier_links"].append(link_entry)
                if isinstance(quantity, (int, float)):
                    inv["stock"] = float(inv.get("stock", 0)) + float(quantity)
                report["linked"] += 1
    return report

def _merge_summaries(target: dict, source: dict):
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        if not value:
            continue
        if not target.get(key):
            target[key] = value

def _collect_unique_barcodes(processed: list[dict]) -> dict:
    seen = {}
    for res in processed:
        data = res.get("data") or {}
        for item in (data.get("barcode") or {}).get("items", []):
            code = (item.get("data") or "").strip()
            if not code:
                continue
            seen.setdefault(code, item)
    if not seen and processed:
        direct = (processed[0].get("data") or {}).get("barcode") or {}
        if isinstance(direct, dict):
            value = direct.get("value")
            if isinstance(value, str):
                seen[value] = {"type": direct.get("type"), "data": value}
    return {"items": list(seen.values())}

def build_combined_product(processed: list[dict], report: dict|None, output_path: Path, structured_data: dict|None = None, product_id: str|None = None):
    if not processed:
        return None
    combined = {
        "id": product_id or processed[0]["img_path"].stem,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00","Z"),
        "summary": {},
        "inventory": {"stock": 0.0, "supplier_links": []},
        "barcode": {},
        "views": {},
        "dates": {"raw_candidates": []},
        "mrp": {},
        "link_report": report or {},
        "ingredients": [],
        "claims": [],
        "certifications": [],
        "allergen_info": None,
        "facility_warning": None,
        "nutrition_facts": {},
        "other_text": [],
        "net_quantity": {"value": None, "unit": None},
        "serving": {"serving_size": None, "servings_per_container": None},
        "variant_or_flavor": None,
        "distributor": None,
        "address": None,
        "website": None,
        "storage_instructions": None,
        "item_number": None,
        "evidence": {}
    }

    supplier_seen = set()
    barcode_candidates: set[str] = set()
    for res in processed:
        data = res.get("data") or {}
        view = res.get("view_label") or "other"
        outputs = data.get("outputs") or {}
        combined["views"][view] = {
            "json": str(res.get("raw_json_path")),
            "hero": outputs.get("hero_cutout_image") or outputs.get("hero_image"),
            "ocr": outputs.get("ocr_image"),
            "masked": outputs.get("masked_image"),
            "anchor": outputs.get("anchor_image")
        }
        _merge_summaries(combined["summary"], data.get("summary") or {})
        anchor = data.get("anchor_extraction") or {}
        dates = anchor.get("dates") or {}
        mrp = anchor.get("mrp") or {}
        for field in ("best_by", "expiration", "manufactured_on", "lot"):
            if dates.get(field) and not combined["dates"].get(field):
                combined["dates"][field] = dates[field]
        if dates.get("raw_candidates"):
            existing_rc = combined["dates"].setdefault("raw_candidates", [])
            for rc in dates["raw_candidates"]:
                if rc not in existing_rc:
                    existing_rc.append(rc)
        if mrp and mrp.get("amount") and not combined["mrp"]:
            combined["mrp"] = mrp

        inv = data.get("inventory") or {}
        stock = inv.get("stock")
        if isinstance(stock, (int, float)):
            combined["inventory"]["stock"] += float(stock)
        for link in inv.get("supplier_links") or []:
            key_sup = (link.get("supplier_gstin"), link.get("bill_path"), link.get("quantity"))
            if key_sup in supplier_seen:
                continue
            supplier_seen.add(key_sup)
            combined["inventory"]["supplier_links"].append(link)

        for item in data.get("ingredients") or []:
            if item and item not in combined["ingredients"]:
                combined["ingredients"].append(item)
        for item in data.get("claims") or []:
            if item and item not in combined["claims"]:
                combined["claims"].append(item)
        for item in data.get("certifications") or []:
            if item and item not in combined["certifications"]:
                combined["certifications"].append(item)
        allergen = data.get("allergen_info")
        if allergen and not combined["allergen_info"]:
            combined["allergen_info"] = allergen
        facility = data.get("facility_warning")
        if facility and not combined["facility_warning"]:
            combined["facility_warning"] = facility
        nutrition = data.get("nutrition_facts") or {}
        if nutrition and not combined["nutrition_facts"]:
            combined["nutrition_facts"] = nutrition
        for item in data.get("other_text") or []:
            if item and item not in combined["other_text"]:
                combined["other_text"].append(item)

        lines = (data.get("ocr") or {}).get("lines") or []
        for line in lines:
            txt = str(line.get("text", "")).strip()
            if not txt:
                continue
            for match in BARCODE_RE.findall(txt):
                cleaned = re.sub(r"\D", "", match)
                if cleaned and 12 <= len(cleaned) <= 20:
                    barcode_candidates.add(cleaned)

    summary_snapshot = combined.get("summary") or {}
    if summary_snapshot.get("serving_size") and not combined["serving"].get("serving_size"):
        combined["serving"]["serving_size"] = summary_snapshot["serving_size"]
    if summary_snapshot.get("variant_or_flavor") and not combined.get("variant_or_flavor"):
        combined["variant_or_flavor"] = summary_snapshot["variant_or_flavor"]

    combined["inventory"]["stock"] = round(combined["inventory"]["stock"], 3)
    combined["barcode"] = _collect_unique_barcodes(processed)
    items_list = combined["barcode"].setdefault("items", [])
    if structured_data:
        barcode_struct = structured_data.get("barcode") or {}
        value = barcode_struct.get("value")
        if value:
            value_str = str(value).strip()
            if value_str and not any((itm.get("data") or "") == value_str for itm in items_list):
                items_list.append({"type": barcode_struct.get("type"), "data": value_str})
    if not items_list and barcode_candidates:
        for code in sorted(barcode_candidates):
            items_list.append({"type": None, "data": code})
    if structured_data:
        _apply_structured_data(combined, structured_data)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    return combined

def _apply_structured_data(combined: dict, structured: dict):
    if not structured:
        return
    summary = combined.setdefault("summary", {})
    def _set_summary(key, value):
        if value:
            summary[key] = value

    brand_struct = structured.get("brand")
    _set_summary("brand", brand_struct)

    variant = (structured.get("variant_or_flavor") or "").strip()
    if variant:
        combined["variant_or_flavor"] = variant
        _set_summary("variant_or_flavor", variant)

    product_struct = (structured.get("product_name") or "").strip()
    product_choice = product_struct
    if variant:
        variant_lower = variant.lower()
        brand_lower = (brand_struct or "").strip().lower()
        if not product_choice:
            product_choice = variant
        else:
            simple = product_choice.lower()
            if simple == brand_lower or simple == variant_lower:
                product_choice = variant
            elif brand_lower and (simple.startswith(brand_lower) or simple in brand_lower or brand_lower in simple):
                product_choice = f"{brand_struct} {variant}".strip()
            elif len(simple) <= 6 and variant_lower not in simple:
                product_choice = f"{brand_struct} {variant}".strip()
    if product_choice:
        _set_summary("product_name", product_choice)

    net_quantity = structured.get("net_quantity") or {}
    value = net_quantity.get("value")
    unit = net_quantity.get("unit")
    if value:
        value_str = str(value)
        qty_str = f"{value_str} {unit}".strip() if unit else value_str
        _set_summary("net_weight", qty_str)
        combined["net_quantity"] = {"value": value_str, "unit": unit}

    serving = structured.get("serving") or {}
    serving_size = serving.get("serving_size")
    servings_per_container = serving.get("servings_per_container")
    if serving_size:
        combined["serving"]["serving_size"] = serving_size
        _set_summary("serving_size", serving_size)
    if servings_per_container:
        combined["serving"]["servings_per_container"] = servings_per_container

    barcode_block = combined.setdefault("barcode", {"items": []})
    barcode_items = barcode_block.setdefault("items", [])
    barcode_struct = structured.get("barcode") or {}
    barcode_value = barcode_struct.get("value")
    barcode_type = barcode_struct.get("type")
    if barcode_value:
        exists = any((item.get("data") or "") == barcode_value for item in barcode_items)
        if not exists:
            barcode_items.append({"type": barcode_type, "data": barcode_value})

    dates_struct = structured.get("dates") or {}
    combined_dates = combined.setdefault("dates", {"raw_candidates": []})
    for field in ("best_by", "expiration", "manufactured_on", "lot"):
        value = dates_struct.get(field)
        if value:
            combined_dates[field] = value
    summary_exp = dates_struct.get("expiration") or dates_struct.get("best_by")
    if summary_exp:
        _set_summary("expiry", summary_exp)
    raw_candidates = dates_struct.get("raw_candidates") or []
    if raw_candidates:
        existing_rc = combined_dates.setdefault("raw_candidates", [])
        for rc in raw_candidates:
            if rc not in existing_rc:
                existing_rc.append(rc)

    mrp_struct = structured.get("mrp") or {}
    if mrp_struct.get("amount") or mrp_struct.get("currency") or mrp_struct.get("raw"):
        combined["mrp"] = mrp_struct

    for key in ("ingredients", "claims", "certifications", "other_text"):
        values = structured.get(key) or []
        if not values:
            continue
        existing = combined.setdefault(key, [])
        for val in values:
            if val and val not in existing:
                existing.append(val)

    for key in ("allergen_info", "facility_warning"):
        val = structured.get(key)
        if val:
            combined[key] = val

    nutrition = structured.get("nutrition_facts") or {}
    if nutrition:
        combined["nutrition_facts"] = nutrition

    for attr in ("distributor", "address", "website", "storage_instructions", "item_number"):
        val = structured.get(attr)
        if val:
            combined[attr] = val

    evidence = structured.get("evidence")
    if isinstance(evidence, dict):
        combined["evidence"] = evidence

def update_global_catalog(combined: dict|None, aggregate_path: Path, views: dict):
    if combined is None:
        return
    aggregate = {"catalog": []}
    if aggregate_path.exists():
        try:
            with aggregate_path.open("r", encoding="utf-8") as f:
                aggregate = json.load(f)
        except Exception:
            aggregate = {"catalog": []}
    catalog_list = aggregate.get("catalog") or []
    catalog_map = {entry.get("key"): entry for entry in catalog_list}

    items = combined.get("barcode", {}).get("items") or []
    key = None
    if items:
        key = items[0].get("data")
    if not key:
        key = combined.get("id")

    summary = combined.get("summary") or {}
    brand_norm = (summary.get("brand") or "").strip().lower()
    name_norm = (summary.get("product_name") or "").strip().lower()
    existing = catalog_map.get(key)
    if not existing and name_norm:
        for entry in catalog_list:
            es = entry.get("summary") or {}
            entry_name = (es.get("product_name") or "").strip().lower()
            entry_brand = (es.get("brand") or "").strip().lower()
            if entry_name == name_norm and (not brand_norm or not entry_brand or entry_brand == brand_norm):
                existing = entry
                key = entry.get("key") or key
                break
    else:
        existing = catalog_map.get(key)

    assets_dir = OUT_DIR / "catalog_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    view_order = ["front", "back", "expiry", "expiry2", "expiry3", "nutrition", "side", "side2", "side3", "side4", "side5"]
    view_entries = []
    hero_rel = None
    used_labels = set()
    for label in view_order:
        info = views.get(label)
        if not info:
            continue
        hero_candidate = info.get("hero") or info.get("masked") or info.get("ocr")
        hero_path = _copy_catalog_asset(hero_candidate, key or combined.get("id", "product"), label, assets_dir) if hero_candidate else None
        if hero_rel is None and hero_path:
            hero_rel = hero_path
        view_entries.append({
            "label": label,
            "hero": hero_path,
            "ocr": _rel_path(info.get("ocr")),
            "masked": _rel_path(info.get("masked")),
            "anchor": _rel_path(info.get("anchor")),
            "json": _rel_path(info.get("json"))
        })
        used_labels.add(label)

    # Include any remaining views not covered by predefined ordering
    for label, info in views.items():
        if label in used_labels:
            continue
        hero_candidate = info.get("hero") or info.get("masked") or info.get("ocr")
        hero_path = _copy_catalog_asset(hero_candidate, key or combined.get("id", "product"), label, assets_dir) if hero_candidate else None
        if hero_rel is None and hero_path:
            hero_rel = hero_path
        view_entries.append({
            "label": label,
            "hero": hero_path,
            "ocr": _rel_path(info.get("ocr")),
            "masked": _rel_path(info.get("masked")),
            "anchor": _rel_path(info.get("anchor")),
            "json": _rel_path(info.get("json"))
        })

    if existing:
        existing["summary"] = combined.get("summary") or existing.get("summary") or {}
        existing["inventory"] = combined.get("inventory") or existing.get("inventory") or {"stock": 0.0, "supplier_links": []}
        existing["barcode"] = combined.get("barcode") or existing.get("barcode") or {}
        existing["dates"] = combined.get("dates") or existing.get("dates") or {}
        existing["mrp"] = combined.get("mrp") or existing.get("mrp") or {}
        existing["ingredients"] = combined.get("ingredients") or existing.get("ingredients") or []
        existing["claims"] = combined.get("claims") or existing.get("claims") or []
        existing["certifications"] = combined.get("certifications") or existing.get("certifications") or []
        existing["allergen_info"] = combined.get("allergen_info") or existing.get("allergen_info")
        existing["facility_warning"] = combined.get("facility_warning") or existing.get("facility_warning")
        existing["nutrition_facts"] = combined.get("nutrition_facts") or existing.get("nutrition_facts") or {}
        existing["other_text"] = combined.get("other_text") or existing.get("other_text") or []
        existing["serving"] = combined.get("serving") or existing.get("serving") or {"serving_size": None, "servings_per_container": None}
        existing["variant_or_flavor"] = combined.get("variant_or_flavor") or existing.get("variant_or_flavor")
        existing["distributor"] = combined.get("distributor") or existing.get("distributor")
        existing["address"] = combined.get("address") or existing.get("address")
        existing["website"] = combined.get("website") or existing.get("website")
        existing["storage_instructions"] = combined.get("storage_instructions") or existing.get("storage_instructions")
        existing["item_number"] = combined.get("item_number") or existing.get("item_number")
        existing["net_quantity"] = combined.get("net_quantity") or existing.get("net_quantity")
        existing["evidence"] = combined.get("evidence") or existing.get("evidence") or {}
        existing["views"] = view_entries or existing.get("views") or []
        existing["ai_enhance_report"] = combined.get("ai_enhance_report") or existing.get("ai_enhance_report")
        existing["last_updated"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00","Z")
        if hero_rel:
            existing["hero_image"] = hero_rel
    else:
        entry = {
            "key": key,
            "id": combined.get("id"),
            "summary": combined.get("summary") or {},
            "inventory": combined.get("inventory") or {},
            "barcode": combined.get("barcode") or {},
            "dates": combined.get("dates") or {},
            "mrp": combined.get("mrp") or {},
            "ingredients": combined.get("ingredients") or [],
            "claims": combined.get("claims") or [],
            "certifications": combined.get("certifications") or [],
            "allergen_info": combined.get("allergen_info"),
            "facility_warning": combined.get("facility_warning"),
            "nutrition_facts": combined.get("nutrition_facts") or {},
            "other_text": combined.get("other_text") or [],
            "serving": combined.get("serving") or {"serving_size": None, "servings_per_container": None},
            "variant_or_flavor": combined.get("variant_or_flavor"),
            "distributor": combined.get("distributor"),
            "address": combined.get("address"),
            "website": combined.get("website"),
            "storage_instructions": combined.get("storage_instructions"),
            "item_number": combined.get("item_number"),
            "net_quantity": combined.get("net_quantity") or {"value": None, "unit": None},
            "evidence": combined.get("evidence") or {},
            "views": view_entries,
            "ai_enhance_report": combined.get("ai_enhance_report"),
            "hero_image": hero_rel,
            "created_at": combined.get("created_at"),
            "last_updated": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00","Z")
        }
        catalog_list.append(entry)

    # Remove stale duplicates referencing the same source id but different keys
    source_id = combined.get("id")
    if source_id:
        catalog_list[:] = [
            entry for entry in catalog_list
            if not (entry.get("id") == source_id and entry.get("key") != key)
        ]

    aggregate["catalog"] = catalog_list
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    render_catalog_html(aggregate, OUT_DIR / "catalog.html")


def apply_catalog_overrides(path: Path):
    catalog_path = OUT_DIR / "catalog.json"
    if not catalog_path.exists():
        raise SystemExit("catalog.json not found. Run the pipeline once before applying overrides.")

    try:
        with path.open("r", encoding="utf-8") as f:
            overrides = json.load(f)
    except Exception as e:
        raise SystemExit(f"failed to read overrides: {e}")

    if isinstance(overrides, dict):
        if "products" in overrides and isinstance(overrides["products"], list):
            overrides_list = overrides["products"]
        elif "catalog" in overrides and isinstance(overrides["catalog"], list):
            overrides_list = overrides["catalog"]
        else:
            overrides_list = []
    elif isinstance(overrides, list):
        overrides_list = overrides
    else:
        overrides_list = []

    if not overrides_list:
        print("override file contained no products; nothing to apply")
        return

    with catalog_path.open("r", encoding="utf-8") as f:
        aggregate = json.load(f)

    catalog_list = aggregate.get("catalog") or []
    catalog_map = {entry.get("key") or entry.get("id"): entry for entry in catalog_list if isinstance(entry, dict)}

    updated = False
    for item in overrides_list:
        if not isinstance(item, dict):
            continue
        key = item.get("key") or item.get("id")
        if not key:
            continue
        item.setdefault("id", item.get("id") or key)
        catalog_map[key] = item
        updated = True

    if not updated:
        print("no matching products were updated")
        return

    aggregate["catalog"] = list(catalog_map.values())
    with catalog_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    render_catalog_html(aggregate, OUT_DIR / "catalog.html")
    print(f"overrides from {path} applied to catalog.json and catalog.html")


def _hero_for_catalog_entry(entry: dict[str, Any]) -> str | None:
    hero = entry.get("hero_image")
    if hero:
        return hero
    for view in entry.get("views") or []:
        if not isinstance(view, dict):
            continue
        for key in ("hero", "ocr", "masked"):
            candidate = view.get(key)
            if candidate:
                return candidate
    return None


def _primary_barcode_value(entry: dict[str, Any]) -> str | None:
    barcode = entry.get("barcode") or {}
    items = barcode.get("items") or []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get("data")
        if value:
            return str(value)
    return None


def _format_stock_value(stock: Any) -> str:
    if isinstance(stock, (int, float)):
        value = f"{stock:.2f}".rstrip("0").rstrip(".")
        return value or "0"
    return ""


def _render_catalog_table_rows(catalog_sorted: list[dict[str, Any]]) -> str:
    if not catalog_sorted:
        return "                <tr><td colspan=\"6\" class=\"empty\">No products yet.</td></tr>"

    rows: list[str] = []
    for idx, entry in enumerate(catalog_sorted, 1):
        summary = entry.get("summary") or {}
        hero = _hero_for_catalog_entry(entry)
        hero_html = (
            f'<img src="{html.escape(hero)}" alt="Product image" loading="lazy">'
            if hero else '<span class="no-image">--</span>'
        )
        barcode_value = _primary_barcode_value(entry) or ""
        barcode_html = html.escape(barcode_value) if barcode_value else "--"
        brand_html = html.escape(summary.get("brand") or "") or "--"
        name_html = html.escape(summary.get("product_name") or "") or "--"
        stock_raw = (entry.get("inventory") or {}).get("stock")
        stock_value = _format_stock_value(stock_raw)
        stock_html = html.escape(stock_value) if stock_value else "--"
        rows.append(
            "                <tr data-index=\"{idx}\">\n"
            "                    <td>{order}</td>\n"
            "                    <td class=\"thumb\">{hero}</td>\n"
            "                    <td>{barcode}</td>\n"
            "                    <td>{brand}</td>\n"
            "                    <td>{name}</td>\n"
            "                    <td>{stock}</td>\n"
            "                </tr>".format(
                idx=idx - 1,
                order=idx,
                hero=hero_html,
                barcode=barcode_html,
                brand=brand_html,
                name=name_html,
                stock=stock_html,
            )
        )
    return "\n".join(rows)


def render_catalog_html(aggregate: dict, html_path: Path):
    catalog = aggregate.get('catalog') or []
    catalog_sorted = sorted(
        catalog,
        key=lambda e: e.get('summary', {}).get('product_name') or e.get('key') or '',
        reverse=False,
    )
    slash_escape = "<" + "\\/"
    catalog_json = json.dumps(catalog_sorted, ensure_ascii=False).replace("</", slash_escape)
    validation_endpoint_json = json.dumps(LLM_VALIDATION_ENDPOINT, ensure_ascii=False)
    enhance_endpoint_json = json.dumps(LLM_ENHANCE_ENDPOINT, ensure_ascii=False)
    critical_fields_json = json.dumps(sorted(AI_ENHANCE_CRITICAL_FIELDS), ensure_ascii=False)
    enhance_reports: dict[str, dict] = {}
    for entry in catalog_sorted:
        key = entry.get('key') or entry.get('id')
        if not key:
            continue
        report = entry.get('ai_enhance_report')
        if isinstance(report, dict):
            enhance_reports[key] = report
    ai_enhance_json = json.dumps(enhance_reports, ensure_ascii=False).replace("</", slash_escape)
    table_rows = _render_catalog_table_rows(catalog_sorted)

    fallback_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Product Catalog</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fb; margin: 0; padding: 24px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
        .header h1 { color: #1f2937; margin: 0; }
    </style>
</head>
<body>
    <div class="header"><h1>Product Catalog</h1></div>
    <div id="detail">Select a product to view details.</div>
    <script id="catalog-data" type="application/json">__CATALOG_JSON__</script>
    <script id="ai-enhance-data" type="application/json">__AI_ENHANCE_JSON__</script>
    <script></script>
</body>
</html>
"""

    template_path = _find_catalog_template()
    if template_path:
        try:
            html_template = template_path.read_text(encoding='utf-8')
        except Exception as exc:
            print(f"catalog_note=template_read_failed path={template_path} err={exc}")
            html_template = fallback_template
    else:
        print("catalog_note=template_not_found_using_fallback")
        html_template = fallback_template

    html_template = re.sub(
        r'<script id="catalog-data" type="application/json">.*?</script>',
        '<script id="catalog-data" type="application/json">__CATALOG_JSON__</script>',
        html_template,
        count=1,
        flags=re.S
    )
    if 'id="ai-enhance-data"' not in html_template:
        html_template = html_template.replace(
            '<script id="catalog-data" type="application/json">__CATALOG_JSON__</script>',
            '<script id="catalog-data" type="application/json">__CATALOG_JSON__</script>\n    <script id="ai-enhance-data" type="application/json">__AI_ENHANCE_JSON__</script>',
            1
        )
    else:
        html_template = re.sub(
            r'<script id="ai-enhance-data" type="application/json">.*?</script>',
            '<script id="ai-enhance-data" type="application/json">__AI_ENHANCE_JSON__</script>',
            html_template,
            count=1,
            flags=re.S
        )

    style_extra = '\n/* AI Enhance additions */\n.validation-status { margin-top: 8px; font-size: 0.88rem; color: #2563eb; min-height: 18px; }\n.validation-status.error { color: #dc2626; }\n.validation-status.success { color: #047857; }\n.analysis-body { padding: 0 4px 12px; display: flex; flex-direction: column; gap: 12px; }\n.analysis-summary { margin: 0; color: #1f2937; font-weight: 600; }\n.analysis-results { display: flex; flex-direction: column; gap: 10px; }\n.analysis-issue { background: #f3f4f6; border-radius: 10px; padding: 12px; border: 1px solid #e5e7eb; transition: border-color 0.15s ease, box-shadow 0.15s ease; }\n.analysis-issue.disabled { opacity: 0.65; }\n.analysis-issue.selected { border-color: #2563eb; box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15); }\n.analysis-issue-header { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }\n.analysis-issue-header input[type="checkbox"] { margin: 0; width: 16px; height: 16px; }\n.analysis-issue-title { font-size: 0.95rem; font-weight: 600; color: #111827; }\n.analysis-issue .badge.small { font-size: 0.7rem; background: #fee2e2; color: #b91c1c; padding: 2px 8px; border-radius: 999px; }\n.analysis-issue .note { font-size: 0.78rem; color: #6b7280; margin-top: 4px; }\n.analysis-actions { display: flex; justify-content: flex-end; gap: 12px; padding: 0 4px 4px; flex-wrap: wrap; }\n.analysis-actions button { background: #2563eb; color: #fff; border: none; padding: 10px 18px; border-radius: 8px; font-size: 0.9rem; cursor: pointer; }\n.analysis-actions button.primary { background: #2563eb; }\n.analysis-actions button.secondary { background: #111827; }\n.analysis-actions button.ghost { background: #f9fafb; color: #1f2937; border: 1px solid #e5e7eb; }\n.analysis-actions button.ghost:hover { background: #e5e7eb; }\n.analysis-actions button.outline { background: #f9fafb; color: #1f2937; border: 1px solid #cbd5f5; }\n.analysis-actions button:disabled { opacity: 0.6; cursor: not-allowed; }\n#enhance-modal .modal-content { width: min(580px, 90%); max-height: 80vh; overflow-y: auto; }\n'
    if '.analysis-issue' not in html_template:
        html_template = html_template.replace('</style>', style_extra + '\n    </style>', 1)
    else:
        html_template = html_template.replace('</style>', style_extra + '\n    </style>', 1)
    html_template = re.sub(
        r'<tbody>\s*</tbody>',
        '                <tbody>\n' + table_rows + '\n                </tbody>',
        html_template,
        count=1,
        flags=re.S
    )

    js_block = 'const data = JSON.parse(document.getElementById(\\\'catalog-data\\\').textContent || \\\'[]\\\');\\n    const validationEndpoint = __VALIDATION_ENDPOINT__;\\n    const enhanceEndpoint = __ENHANCE_ENDPOINT__;\\n    const aiEnhanceReports = JSON.parse(document.getElementById(\\\'ai-enhance-data\\\').textContent || \\\'{}\\\');\\n    const criticalFields = new Set(__CRITICAL_FIELDS__);\\n    const originalData = (data || []).map(function(item) { return JSON.parse(JSON.stringify(item)); });\\n    const tableBody = document.querySelector(\\\'#product-table tbody\\\');\\n    const detail = document.getElementById(\\\'detail\\\');\\n    const downloadCatalogBtn = document.getElementById(\\\'download-catalog-btn\\\');\\n    const editModal = document.getElementById(\\\'edit-modal\\\');\\n    const editForm = document.getElementById(\\\'edit-form\\\');\\n    const editCancelBtn = document.getElementById(\\\'edit-cancel-btn\\\');\\n    const editCloseBtn = document.getElementById(\\\'edit-close-btn\\\');\\n    const editViewList = document.getElementById(\\\'edit-view-list\\\');\\n    const enhanceModal = document.getElementById(\\\'enhance-modal\\\');\\n    const enhanceCloseBtn = document.getElementById(\\\'enhance-close-btn\\\');\\n    const enhanceDismissBtn = document.getElementById(\\\'enhance-dismiss-btn\\\');\\n    const enhanceResults = document.getElementById(\\\'enhance-results\\\');\\n    const enhanceSummary = document.getElementById(\\\'enhance-summary\\\');\\n    let currentProductIndex = 0;\\n    let modalViewOrder = [];\\n    let dragIndex = null;\\n    let validationStatusTimer = null;\\n\\n    function deepClone(value) {\\n        return JSON.parse(JSON.stringify(value));\\n    }\\n\\n    function normalizeValue(value) {\\n        if (value === undefined || value === null) { return null; }\\n        if (typeof value === \\\'string\\\') {\\n            const trimmed = value.trim();\\n            return trimmed === \\\'\\\' ? null : trimmed;\\n        }\\n        return value;\\n    }\\n\\n    const FIELD_MAP = [\\n        {\\n            id: \\\'summary.product_name\\\',\\n            label: \\\'Product Name\\\',\\n            get: function(prod) { return prod.summary && prod.summary.product_name; },\\n            set: function(prod, value) {\\n                prod.summary = prod.summary || {};\\n                prod.summary.product_name = value || null;\\n            }\\n        },\\n        {\\n            id: \\\'summary.brand\\\',\\n            label: \\\'Brand\\\',\\n            get: function(prod) { return prod.summary && prod.summary.brand; },\\n            set: function(prod, value) {\\n                prod.summary = prod.summary || {};\\n                prod.summary.brand = value || null;\\n            }\\n        },\\n        {\\n            id: \\\'summary.variant_or_flavor\\\',\\n            label: \\\'Variant or Flavor\\\',\\n            get: function(prod) {\\n                if (prod.summary && prod.summary.variant_or_flavor) { return prod.summary.variant_or_flavor; }\\n                return prod.variant_or_flavor || null;\\n            },\\n            set: function(prod, value) {\\n                prod.summary = prod.summary || {};\\n                prod.summary.variant_or_flavor = value || null;\\n                prod.variant_or_flavor = value || null;\\n            }\\n        },\\n        {\\n            id: \\\'summary.net_weight\\\',\\n            label: \\\'Net Quantity Label\\\',\\n            get: function(prod) { return prod.summary && prod.summary.net_weight; },\\n            set: function(prod, value) {\\n                prod.summary = prod.summary || {};\\n                prod.summary.net_weight = value || null;\\n            }\\n        },\\n        {\\n            id: \\\'summary.serving_size\\\',\\n            label: \\\'Serving Size (label)\\\',\\n            get: function(prod) { return prod.summary && prod.summary.serving_size; },\\n            set: function(prod, value) {\\n                prod.summary = prod.summary || {};\\n                prod.summary.serving_size = value || null;\\n                prod.serving = prod.serving || {"serving_size": null, "servings_per_container": null};\\n                prod.serving.serving_size = value || null;\\n            }\\n        },\\n        {\\n            id: \\\'summary.expiry\\\',\\n            label: \\\'Expiry Summary\\\',\\n            get: function(prod) { return prod.summary && prod.summary.expiry; },\\n            set: function(prod, value) {\\n                prod.summary = prod.summary || {};\\n                prod.summary.expiry = value || null;\\n            }\\n        },\\n        {\\n            id: \\\'net_quantity.value\\\',\\n            label: \\\'Net Quantity Value\\\',\\n            get: function(prod) { return prod.net_quantity && prod.net_quantity.value; },\\n            set: function(prod, value) {\\n                prod.net_quantity = prod.net_quantity || {"value": null, "unit": null};\\n                prod.net_quantity.value = value || null;\\n                prod.summary = prod.summary || {};\\n                const valueClean = value === null || value === undefined ? \\\'\\\' : String(value).trim();\\n                const unitClean = prod.net_quantity.unit ? \\\' \\\' + String(prod.net_quantity.unit).trim() : \\\'\\\';\\n                if (valueClean) {\\n                    prod.summary.net_weight = valueClean + unitClean;\\n                } else if (!prod.net_quantity.unit) {\\n                    prod.summary.net_weight = null;\\n                } else {\\n                    prod.summary.net_weight = String(prod.net_quantity.unit).trim() || null;\\n                }\\n            }\\n        },\\n        {\\n            id: \\\'net_quantity.unit\\\',\\n            label: \\\'Net Quantity Unit\\\',\\n            get: function(prod) { return prod.net_quantity && prod.net_quantity.unit; },\\n            set: function(prod, value) {\\n                prod.net_quantity = prod.net_quantity || {"value": prod.net_quantity && prod.net_quantity.value || null, "unit": null};\\n                prod.net_quantity.unit = value || null;\\n                prod.summary = prod.summary || {};\\n                const qtyValue = prod.net_quantity.value === null || prod.net_quantity.value === undefined ? \\\'\\\' : String(prod.net_quantity.value).trim();\\n                const unitClean = value === null || value === undefined ? \\\'\\\' : String(value).trim();\\n                if (qtyValue) {\\n                    prod.summary.net_weight = qtyValue + (unitClean ? \\\' \\\' + unitClean : \\\'\\\');\\n                } else if (unitClean) {\\n                    prod.summary.net_weight = unitClean;\\n                } else {\\n                    prod.summary.net_weight = null;\\n                }\\n            }\\n        },\\n        {\\n            id: \\\'serving.serving_size\\\',\\n            label: \\\'Serving Size\\\',\\n            get: function(prod) { return prod.serving && prod.serving.serving_size; },\\n            set: function(prod, value) {\\n                prod.serving = prod.serving || {"serving_size": null, "servings_per_container": null};\\n                prod.serving.serving_size = value || null;\\n                prod.summary = prod.summary || {};\\n                prod.summary.serving_size = value || null;\\n            }\\n        },\\n        {\\n            id: \\\'serving.servings_per_container\\\',\\n            label: \\\'Servings Per Container\\\',\\n            get: function(prod) { return prod.serving && prod.serving.servings_per_container; },\\n            set: function(prod, value) {\\n                prod.serving = prod.serving || {"serving_size": null, "servings_per_container": null};\\n                prod.serving.servings_per_container = value || null;\\n            }\\n        },\\n        {\\n            id: \\\'dates.expiration\\\',\\n            label: \\\'Expiration\\\',\\n            get: function(prod) { return prod.dates && prod.dates.expiration; },\\n            set: function(prod, value) {\\n                prod.dates = prod.dates || {"raw_candidates": []};\\n                prod.dates.expiration = value || null;\\n                prod.summary = prod.summary || {};\\n                if (value) {\\n                    prod.summary.expiry = value;\\n                }\\n            }\\n        },\\n        {\\n            id: \\\'dates.best_by\\\',\\n            label: \\\'Best By\\\',\\n            get: function(prod) { return prod.dates && prod.dates.best_by; },\\n            set: function(prod, value) {\\n                prod.dates = prod.dates || {"raw_candidates": []};\\n                prod.dates.best_by = value || null;\\n            }\\n        },\\n        {\\n            id: \\\'dates.lot\\\',\\n            label: \\\'Lot Code\\\',\\n            get: function(prod) { return prod.dates && prod.dates.lot; },\\n            set: function(prod, value) {\\n                prod.dates = prod.dates || {"raw_candidates": []};\\n                prod.dates.lot = value || null;\\n            }\\n        },\\n        {\\n            id: \\\'mrp.amount\\\',\\n            label: \\\'MRP Amount\\\',\\n            get: function(prod) { return prod.mrp && prod.mrp.amount; },\\n            set: function(prod, value) {\\n                prod.mrp = prod.mrp || {"amount": null, "currency": null, "raw": null};\\n                prod.mrp.amount = value || null;\\n            }\\n        },\\n        {\\n            id: \\\'mrp.currency\\\',\\n            label: \\\'MRP Currency\\\',\\n            get: function(prod) { return prod.mrp && prod.mrp.currency; },\\n            set: function(prod, value) {\\n                prod.mrp = prod.mrp || {"amount": null, "currency": null, "raw": null};\\n                prod.mrp.currency = value || null;\\n            }\\n        },\\n        {\\n            id: \\\'mrp.raw\\\',\\n            label: \\\'MRP Raw\\\',\\n            get: function(prod) { return prod.mrp && prod.mrp.raw; },\\n            set: function(prod, value) {\\n                prod.mrp = prod.mrp || {"amount": null, "currency": null, "raw": null};\\n                prod.mrp.raw = value || null;\\n            }\\n        },\\n        {\\n            id: \\\'barcode.items[0].data\\\',\\n            label: \\\'Barcode Value\\\',\\n            get: function(prod) {\\n                if (!prod.barcode || !Array.isArray(prod.barcode.items) || !prod.barcode.items.length) { return null; }\\n                return prod.barcode.items[0].data;\\n            },\\n            set: function(prod, value) {\\n                prod.barcode = prod.barcode || {"items": []};\\n                if (!Array.isArray(prod.barcode.items)) { prod.barcode.items = []; }\\n                if (!prod.barcode.items.length) { prod.barcode.items.push({"type": null, "data": null}); }\\n                prod.barcode.items[0].data = value || null;\\n            }\\n        },\\n        {\\n            id: \\\'barcode.items[0].type\\\',\\n            label: \\\'Barcode Type\\\',\\n            get: function(prod) {\\n                if (!prod.barcode || !Array.isArray(prod.barcode.items) || !prod.barcode.items.length) { return null; }\\n                return prod.barcode.items[0].type;\\n            },\\n            set: function(prod, value) {\\n                prod.barcode = prod.barcode || {"items": []};\\n                if (!Array.isArray(prod.barcode.items)) { prod.barcode.items = []; }\\n                if (!prod.barcode.items.length) { prod.barcode.items.push({"type": null, "data": null}); }\\n                prod.barcode.items[0].type = value || null;\\n            }\\n        },\\n        {\\n            id: \\\'inventory.stock\\\',\\n            label: \\\'Inventory Stock\\\',\\n            get: function(prod) {\\n                if (!prod.inventory) { return null; }\\n                return prod.inventory.stock;\\n            },\\n            set: function(prod, value) {\\n                prod.inventory = prod.inventory || {"stock": 0, "supplier_links": []};\\n                var numeric = value;\\n                if (numeric === null || numeric === undefined || numeric === \\\'\\\') {\\n                    prod.inventory.stock = null;\\n                    return;\\n                }\\n                var numVal = Number(numeric);\\n                prod.inventory.stock = isNaN(numVal) ? numeric : numVal;\\n            },\\n            normalize: function(value) {\\n                if (value === null || value === undefined || value === \\\'\\\') { return null; }\\n                var num = Number(value);\\n                return isNaN(num) ? value : Number(num.toFixed(4));\\n            }\\n        }\\n    ];\\n\\n    const FIELD_BY_ID = FIELD_MAP.reduce(function(map, field) {\\n        map[field.id] = field;\\n        return map;\\n    }, {});\\n\\n    function collectDifferences(product, baseline) {\\n        const diffs = [];\\n        FIELD_MAP.forEach(function(field) {\\n            const currentRaw = field.get ? field.get(product || {}) : null;\\n            const baselineRaw = field.get ? field.get(baseline || {}) : null;\\n            const normalizeFn = typeof field.normalize === \\\'function\\\' ? field.normalize : normalizeValue;\\n            const currentVal = normalizeFn(currentRaw);\\n            const baselineVal = normalizeFn(baselineRaw);\\n            if (JSON.stringify(currentVal) !== JSON.stringify(baselineVal)) {\\n                diffs.push({\\n                    id: field.id,\\n                    label: field.label,\\n                    current: currentRaw,\\n                    baseline: baselineRaw\\n                });\\n            }\\n        });\\n        return diffs;\\n    }\\n\\n    function snapshotFields(product) {\\n        const snapshot = {};\\n        FIELD_MAP.forEach(function(field) {\\n            snapshot[field.id] = field.get ? field.get(product || {}) : null;\\n        });\\n        return snapshot;\\n    }\\n\\n    function computeDiffIssues(product, baseline, diffs) {\\n        return (diffs || []).map(function(diff) {\\n            return {\\n                field: diff.id,\\n                label: diff.label,\\n                displayed: diff.current,\\n                expected: diff.baseline,\\n                reason: \\\'Mismatch between displayed value and source JSON.\\\'\\n            };\\n        });\\n    }\\n\\n    function applyCorrections(product, baseline, updates, diffs) {\\n        let applied = 0;\\n        const handled = {};\\n        if (updates && typeof updates === \\\'object\\\') {\\n            Object.keys(updates).forEach(function(fieldId) {\\n                const field = FIELD_BY_ID[fieldId];\\n                if (!field) { return; }\\n                field.set(product, updates[fieldId]);\\n                handled[fieldId] = true;\\n                applied += 1;\\n            });\\n        }\\n        (diffs || []).forEach(function(diff) {\\n            if (handled[diff.id]) { return; }\\n            const field = FIELD_BY_ID[diff.id];\\n            if (!field) { return; }\\n            const value = field.get ? field.get(baseline || {}) : null;\\n            field.set(product, value);\\n            applied += 1;\\n        });\\n        return applied;\\n    }\\n\\n    function buildValidationPayload(index, product, baseline, diffs) {\\n        const payload = {\\n            product_id: product.key || product.id || product.slug || (\\\'product-\\\' + (index + 1)),\\n            diff_fields: (diffs || []).map(function(diff) {\\n                return { id: diff.id, label: diff.label };\\n            }),\\n            displayed: {},\\n            source: {}\\n        };\\n        (diffs || []).forEach(function(diff) {\\n            const field = FIELD_BY_ID[diff.id];\\n            if (!field) { return; }\\n            payload.displayed[diff.id] = field.get ? field.get(product || {}) : null;\\n            payload.source[diff.id] = field.get ? field.get(baseline || {}) : null;\\n        });\\n        return payload;\\n    }\\n\\n    function buildEnhancePayload(index, product, baseline, diffs) {\\n        return {\\n            product_id: product.key || product.id || product.slug || (\\\'product-\\\' + (index + 1)),\\n            diffs: (diffs || []).map(function(diff) {\\n                return {\\n                    id: diff.id,\\n                    label: diff.label,\\n                    displayed: diff.current,\\n                    source: diff.baseline\\n                };\\n            }),\\n            displayed_fields: snapshotFields(product),\\n            source_fields: snapshotFields(baseline),\\n            displayed_summary: {\\n                summary: product.summary || {},\\n                other_text: product.other_text || [],\\n                claims: product.claims || [],\\n                certifications: product.certifications || [],\\n                nutrition_facts: product.nutrition_facts || {}\\n            },\\n            source_summary: {\\n                summary: baseline.summary || {},\\n                other_text: baseline.other_text || [],\\n                claims: baseline.claims || [],\\n                certifications: baseline.certifications || [],\\n                nutrition_facts: baseline.nutrition_facts || {}\\n            },\\n            displayed_raw: deepClone(product),\\n            source_raw: deepClone(baseline)\\n        };\\n    }\\n\\n    function sendValidationRequest(payload) {\\n        return fetch(validationEndpoint, {\\n            method: \\\'POST\\\',\\n            headers: { \\\'Content-Type\\\': \\\'application/json\\\' },\\n            body: JSON.stringify(payload)\\n        }).then(function(response) {\\n            if (!response.ok) {\\n                throw new Error(\\\'HTTP \\\' + response.status);\\n            }\\n            return response.json();\\n        });\\n    }\\n\\n    function sendEnhanceRequest(payload) {\\n        return fetch(enhanceEndpoint, {\\n            method: \\\'POST\\\',\\n            headers: { \\\'Content-Type\\\': \\\'application/json\\\' },\\n            body: JSON.stringify(payload)\\n        }).then(function(response) {\\n            if (!response.ok) {\\n                throw new Error(\\\'HTTP \\\' + response.status);\\n            }\\n            return response.json();\\n        });\\n    }\\n\\n    function resetValidationStatus() {\\n        const statusEl = document.getElementById(\\\'validation-status\\\');\\n        if (!statusEl) { return; }\\n        statusEl.textContent = \\\'\\\';\\n        statusEl.className = \\\'validation-status\\\';\\n        if (validationStatusTimer) {\\n            clearTimeout(validationStatusTimer);\\n            validationStatusTimer = null;\\n        }\\n    }\\n\\n    function showValidationStatus(message, type, persist) {\\n        const statusEl = document.getElementById(\\\'validation-status\\\');\\n        if (!statusEl) { return; }\\n        if (validationStatusTimer) {\\n            clearTimeout(validationStatusTimer);\\n            validationStatusTimer = null;\\n        }\\n        statusEl.textContent = message || \\\'\\\';\\n        let className = \\\'validation-status\\\';\\n        if (type) {\\n            className += \\\' \\\' + type;\\n        }\\n        statusEl.className = className;\\n        if (!persist && message) {\\n            validationStatusTimer = setTimeout(function() {\\n                statusEl.textContent = \\\'\\\';\\n                statusEl.className = \\\'validation-status\\\';\\n            }, 4000);\\n        }\\n    }\\n\\n    function extractEnhanceIssues(response) {\\n        if (!response) { return []; }\\n        if (Array.isArray(response.issues)) { return response.issues; }\\n        if (response.analysis && Array.isArray(response.analysis.issues)) { return response.analysis.issues; }\\n        if (response.result && Array.isArray(response.result.issues)) { return response.result.issues; }\\n        return [];\\n    }\\n\\n    function formatIssueValue(value) {\\n        if (value === undefined || value === null) { return \\\'—\\\'; }\\n        if (typeof value === \\\'string\\\') {\\n            const trimmed = value.trim();\\n            return trimmed === \\\'\\\' ? \\\'—\\\' : escapeHtml(trimmed);\\n        }\\n        if (typeof value === \\\'number\\\') {\\n            return escapeHtml(String(value));\\n        }\\n        try {\\n            return escapeHtml(JSON.stringify(value));\\n        } catch (error) {\\n            return escapeHtml(String(value));\\n        }\\n    }\\n\\n    function renderEnhanceIssues(issues) {\\n        if (!enhanceResults) { return; }\\n        if (!issues || !issues.length) {\\n            enhanceResults.innerHTML = \\\'<p class="empty">No issues detected.</p>\\\';\\n            return;\\n        }\\n        const html = issues.map(function(issue, index) {\\n            const label = escapeHtml(String(issue.label || issue.field || (\\\'Field \\\' + (index + 1))));\\n            const displayedValue = issue.displayed !== undefined ? issue.displayed :\\n                (issue.actual !== undefined ? issue.actual : (issue.value !== undefined ? issue.value : null));\\n            const expectedValue = issue.expected !== undefined ? issue.expected :\\n                (issue.correct !== undefined ? issue.correct :\\n                    (issue.suggested !== undefined ? issue.suggested :\\n                        (issue.source !== undefined ? issue.source : null)));\\n            const reason = issue.reason || issue.explanation || issue.notes || \\\'\\\';\\n            const displayedText = formatIssueValue(displayedValue);\\n            const expectedText = formatIssueValue(expectedValue);\\n            const reasonText = reason ? escapeHtml(String(reason)) : \\\'\\\';\\n            return \\\'\\\' +\\n                \\\'<div class="analysis-issue">\\\' +\\n                    \\\'<h4>\\\' + label + \\\'</h4>\\\' +\\n                    \\\'<p><strong>Displayed:</strong> \\\' + displayedText + \\\'</p>\\\' +\\n                    \\\'<p><strong>Suggested:</strong> \\\' + expectedText + \\\'</p>\\\' +\\n                    (reasonText ? \\\'<p><strong>Reason:</strong> \\\' + reasonText + \\\'</p>\\\' : \\\'\\\') +\\n                \\\'</div>\\\';\\n        }).join(\\\'\\\');\\n        enhanceResults.innerHTML = html;\\n    }\\n\\n    function openEnhanceModal(issues, summaryText) {\\n        if (!enhanceModal) {\\n            // Fallback to alert if modal is unavailable\\n            const summary = summaryText || \\\'AI analysis completed.\\\';\\n            const details = (issues || []).map(function(issue) {\\n                return (issue.label || issue.field || \\\'Field\\\') + \\\': \\\' + (issue.reason || \\\'See suggested value.\\\');\\n            }).join(\\\'\\\\n\\\');\\n            window.alert(summary + (details ? \\\'\\\\n\\\\n\\\' + details : \\\'\\\'));\\n            return;\\n        }\\n        if (enhanceSummary) {\\n            enhanceSummary.textContent = summaryText || \\\'AI analysis completed.\\\';\\n        }\\n        renderEnhanceIssues(issues);\\n        enhanceModal.classList.remove(\\\'hidden\\\');\\n    }\\n\\n    function closeEnhanceModal() {\\n        if (!enhanceModal) { return; }\\n        enhanceModal.classList.add(\\\'hidden\\\');\\n    }\\n\\n    function validateProduct(index) {\\n        const product = data[index];\\n        const baseline = originalData[index];\\n        if (!product || !baseline) {\\n            showValidationStatus(\\\'Baseline JSON unavailable for this product.\\\', \\\'error\\\');\\n            return;\\n        }\\n        const diffs = collectDifferences(product, baseline);\\n        if (!diffs.length) {\\n            showValidationStatus(\\\'All fields already match the generated JSON.\\\', \\\'info\\\');\\n            return;\\n        }\\n        const validateBtn = document.getElementById(\\\'validate-product-btn\\\');\\n        if (validateBtn) {\\n            validateBtn.disabled = true;\\n        }\\n        showValidationStatus(\\\'Validating against source data...\\\', \\\'info\\\', true);\\n        const payload = buildValidationPayload(index, product, baseline, diffs);\\n\\n        function finalize(applied, message, type) {\\n            originalData[index] = deepClone(data[index]);\\n            populateTable();\\n            showValidationStatus(message, type);\\n            if (validateBtn) {\\n                validateBtn.disabled = false;\\n            }\\n        }\\n\\n        if (validationEndpoint) {\\n            sendValidationRequest(payload).then(function(response) {\\n                const fields = response && response.fields;\\n                const applied = applyCorrections(product, baseline, fields, diffs);\\n                const message = applied\\n                    ? \\\'Updated \\\' + applied + \\\' field\\\' + (applied === 1 ? \\\'\\\' : \\\'s\\\') + \\\' using validator response.\\\'\\n                    : \\\'Validator returned no corrections; using existing display.\\\';\\n                finalize(applied, message, applied ? \\\'success\\\' : \\\'info\\\');\\n            }).catch(function(err) {\\n                const applied = applyCorrections(product, baseline, null, diffs);\\n                const reason = err && err.message ? err.message : \\\'validation failed\\\';\\n                finalize(applied, \\\'Fallback to baseline JSON (\\\' + reason + \\\').\\\', \\\'error\\\');\\n            });\\n        } else {\\n            const applied = applyCorrections(product, baseline, null, diffs);\\n            finalize(applied, \\\'Restored \\\' + applied + \\\' field\\\' + (applied === 1 ? \\\'\\\' : \\\'s\\\') + \\\' from source JSON.\\\', \\\'success\\\');\\n        }\\n    }\\n\\n    function enhanceProduct(index) {\\n        const product = data[index];\\n        const baseline = originalData[index];\\n        if (!product || !baseline) {\\n            showValidationStatus(\\\'Baseline JSON unavailable for this product.\\\', \\\'error\\\');\\n            return;\\n        }\\n        const productKey = (product && (product.key || product.id || product.slug)) || null;\\n        const primaryBarcode = (product && product.barcode && Array.isArray(product.barcode.items) && product.barcode.items.length)\\n            ? (product.barcode.items[0].data || null)\\n            : null;\\n        let precomputed = productKey ? aiEnhanceReports[productKey] : null;\\n        if (!precomputed && primaryBarcode && aiEnhanceReports[primaryBarcode]) {\\n            precomputed = aiEnhanceReports[primaryBarcode];\\n        }\\n        const diffs = collectDifferences(product, baseline);\\n        const enhanceBtn = document.getElementById(\\\'enhance-product-btn\\\');\\n        if (enhanceBtn) {\\n            enhanceBtn.disabled = true;\\n        }\\n        const hasEndpoint = Boolean(enhanceEndpoint && String(enhanceEndpoint).trim());\\n        const precomputedIssues = precomputed && Array.isArray(precomputed.issues) ? precomputed.issues : null;\\n        const precomputedSummary = precomputed && (precomputed.summary || precomputed.message || precomputed.status);\\n        if (!hasEndpoint && precomputedIssues) {\\n            openEnhanceModal(precomputedIssues, precomputedSummary || \\\'AI analysis prepared during pipeline run.\\\');\\n            showValidationStatus(precomputedSummary || \\\'AI analysis prepared during pipeline run.\\\', precomputedIssues.length ? \\\'success\\\' : \\\'info\\\');\\n            if (enhanceBtn) {\\n                enhanceBtn.disabled = false;\\n            }\\n            return;\\n        }\\n        if (!hasEndpoint && !precomputedIssues && !diffs.length) {\\n            showValidationStatus(\\\'No differences detected between display and source JSON.\\\', \\\'info\\\');\\n            openEnhanceModal([], \\\'No differences detected between display and source JSON.\\\');\\n            if (enhanceBtn) {\\n                enhanceBtn.disabled = false;\\n            }\\n            return;\\n        }\\n\\n        showValidationStatus(\\\'Sending product display to AI reviewer...\\\', \\\'info\\\', true);\\n        const payload = buildEnhancePayload(index, product, baseline, diffs);\\n        const fallbackIssues = computeDiffIssues(product, baseline, diffs);\\n        const fallbackSummary = fallbackIssues.length\\n            ? \\\'Local diff identified \\\' + fallbackIssues.length + \\\' field\\\' + (fallbackIssues.length === 1 ? \\\'\\\' : \\\'s\\\') + \\\'.\\\'\\n            : \\\'No differences between display and source JSON.\\\';\\n\\n        function finalize(issues, summary, status) {\\n            openEnhanceModal(issues, summary);\\n            showValidationStatus(summary || \\\'AI analysis completed.\\\', status || (issues && issues.length ? \\\'success\\\' : \\\'info\\\'));\\n            if (enhanceBtn) {\\n                enhanceBtn.disabled = false;\\n            }\\n        }\\n\\n        function fallback(summary, status) {\\n            if (precomputedIssues) {\\n                finalize(precomputedIssues, summary || precomputedSummary || fallbackSummary, status || (precomputedIssues.length ? \\\'success\\\' : \\\'info\\\'));\\n                return;\\n            }\\n            const effectiveSummary = summary || fallbackSummary;\\n            const fallbackRecord = { summary: effectiveSummary, issues: fallbackIssues };\\n            if (productKey) {\\n                aiEnhanceReports[productKey] = fallbackRecord;\\n            }\\n            if (primaryBarcode) {\\n                aiEnhanceReports[primaryBarcode] = fallbackRecord;\\n            }\\n            finalize(fallbackIssues, effectiveSummary, status || (fallbackIssues.length ? \\\'info\\\' : \\\'success\\\'));\\n        }\\n\\n        if (hasEndpoint) {\\n            sendEnhanceRequest(payload).then(function(response) {\\n                const issues = extractEnhanceIssues(response);\\n                const summary = response && (response.summary || response.message || response.notes || response.status || \\\'AI analysis completed.\\\');\\n                const normalizedIssues = Array.isArray(issues) ? issues : [];\\n                const record = { summary: summary, issues: normalizedIssues };\\n                if (productKey) {\\n                    aiEnhanceReports[productKey] = record;\\n                }\\n                if (primaryBarcode) {\\n                    aiEnhanceReports[primaryBarcode] = record;\\n                }\\n                if (normalizedIssues.length) {\\n                    finalize(normalizedIssues, summary, \\\'success\\\');\\n                } else if (precomputedIssues) {\\n                    finalize(precomputedIssues, summary || precomputedSummary || fallbackSummary, \\\'info\\\');\\n                } else if (fallbackIssues.length) {\\n                    finalize(fallbackIssues, summary || fallbackSummary, \\\'info\\\');\\n                } else {\\n                    finalize([], summary || \\\'AI did not flag any issues.\\\', \\\'success\\\');\\n                }\\n            }).catch(function(err) {\\n                const reason = err && err.message ? err.message : \\\'analysis failed\\\';\\n                fallback(\\\'AI enhance request failed (\\\' + reason + \\\'). Showing local diff instead.\\\', \\\'error\\\');\\n            });\\n        } else {\\n            fallback(\\\'LLM enhance endpoint not configured. Showing local diff.\\\', \\\'info\\\');\\n        }\\n    }\\n\\n    function heroFor(product) {\\n        if (product.hero_image) { return product.hero_image; }\\n        const views = product.views || [];\\n        for (const view of views) {\\n            if (view.hero) { return view.hero; }\\n        }\\n        return \\\'\\\';\\n    }\\n\\n    function escapeHtml(value) {\\n        if (value === undefined || value === null) { return \\\'\\\'; }\\n        return String(value).replace(/[&<>"\\\']/g, function(ch) {\\n            return ({\\\'&\\\': \\\'&amp;\\\', \\\'<\\\': \\\'&lt;\\\', \\\'>\\\': \\\'&gt;\\\', \\\'"\\\': \\\'&quot;\\\', "\\\'": \\\'&#39;\\\'})[ch];\\n        });\\n    }\\n\\n    function textOrDash(value) {\\n        const escaped = escapeHtml(value);\\n        return escaped === \\\'\\\' ? \\\'—\\\' : escaped;\\n    }\\n\\n    function renderList(items) {\\n        if (!items || !items.length) { return \\\'<p class="empty">None</p>\\\'; }\\n        return \\\'<ul class="inline">\\\' + items.map(function(item) {\\n            return \\\'<li>\\\' + textOrDash(item) + \\\'</li>\\\';\\n        }).join(\\\'\\\') + \\\'</ul>\\\';\\n    }\\n\\n    function renderViews(product) {\\n        const views = (product.views || []).filter(function(v) { return v.hero; });\\n        if (!views.length) { return \\\'<p class="empty">No views captured.</p>\\\'; }\\n        return \\\'<div class="view-gallery">\\\' + views.map(function(v) {\\n            return \\\'<div><img src="\\\' + textOrDash(v.hero) + \\\'" alt="\\\' + textOrDash(v.label) + \\\'"><div>\\\' + textOrDash(v.label.charAt(0).toUpperCase() + v.label.slice(1)) + \\\'</div></div>\\\';\\n        }).join(\\\'\\\') + \\\'</div>\\\';\\n    }\\n\\n    function buildEditViewList() {\\n        if (!editViewList) { return; }\\n        editViewList.innerHTML = \\\'\\\';\\n        modalViewOrder.forEach(function(view, index) {\\n            const item = document.createElement(\\\'div\\\');\\n            item.className = \\\'edit-view-item\\\' + (index === 0 ? \\\' front\\\' : \\\'\\\');\\n            item.draggable = true;\\n            item.dataset.index = index;\\n\\n            const imgSrc = view.hero || view.ocr || view.masked || \\\'\\\';\\n            if (imgSrc) {\\n                const img = document.createElement(\\\'img\\\');\\n                img.src = imgSrc;\\n                img.alt = view.label || (\\\'View \\\' + (index + 1));\\n                item.appendChild(img);\\n            } else {\\n                const placeholder = document.createElement(\\\'div\\\');\\n                placeholder.style.height = \\\'90px\\\';\\n                placeholder.style.display = \\\'flex\\\';\\n                placeholder.style.alignItems = \\\'center\\\';\\n                placeholder.style.justifyContent = \\\'center\\\';\\n                placeholder.style.color = \\\'#9ca3af\\\';\\n                placeholder.textContent = \\\'No image\\\';\\n                item.appendChild(placeholder);\\n            }\\n\\n            const label = document.createElement(\\\'span\\\');\\n            const labelText = view.label ? view.label.charAt(0).toUpperCase() + view.label.slice(1) : (\\\'View \\\' + (index + 1));\\n            label.textContent = labelText;\\n            item.appendChild(label);\\n\\n            item.addEventListener(\\\'dragstart\\\', handleViewDragStart);\\n            item.addEventListener(\\\'dragover\\\', handleViewDragOver);\\n            item.addEventListener(\\\'drop\\\', handleViewDrop);\\n            item.addEventListener(\\\'dragend\\\', handleViewDragEnd);\\n\\n            editViewList.appendChild(item);\\n        });\\n    }\\n\\n    function handleViewDragStart(event) {\\n        dragIndex = parseInt(event.currentTarget.dataset.index || \\\'0\\\', 10);\\n        event.currentTarget.classList.add(\\\'dragging\\\');\\n    }\\n\\n    function handleViewDragOver(event) {\\n        event.preventDefault();\\n    }\\n\\n    function handleViewDrop(event) {\\n        event.preventDefault();\\n        const targetIndex = parseInt(event.currentTarget.dataset.index || \\\'0\\\', 10);\\n        if (isNaN(dragIndex) || isNaN(targetIndex) || dragIndex === targetIndex) {\\n            return;\\n        }\\n        const [moved] = modalViewOrder.splice(dragIndex, 1);\\n        modalViewOrder.splice(targetIndex, 0, moved);\\n        dragIndex = null;\\n        buildEditViewList();\\n    }\\n\\n    function handleViewDragEnd(event) {\\n        event.currentTarget.classList.remove(\\\'dragging\\\');\\n    }\\n\\n    function renderSummaryTable(product) {\\n        const summary = product.summary || {};\\n        const dates = product.dates || {};\\n        const mrp = product.mrp || {};\\n        const inventory = product.inventory || {};\\n        const netQuantity = product.net_quantity || {};\\n        const serving = product.serving || {};\\n        const variant = summary.variant_or_flavor || product.variant_or_flavor;\\n        const netQtyLabel = netQuantity.value ? (netQuantity.unit ? netQuantity.value + \\\' \\\' + netQuantity.unit : netQuantity.value) : summary.net_weight;\\n        const rows = [\\n            [\\\'Product Name\\\', summary.product_name],\\n            [\\\'Brand\\\', summary.brand],\\n            [\\\'Variant\\\', variant],\\n            [\\\'Net Quantity\\\', netQtyLabel],\\n            [\\\'Serving Size\\\', serving.serving_size || summary.serving_size],\\n            [\\\'Servings / Container\\\', serving.servings_per_container],\\n            [\\\'Expiry Hint\\\', summary.expiry],\\n            [\\\'Best By\\\', dates.best_by],\\n            [\\\'Expiration\\\', dates.expiration],\\n            [\\\'Lot\\\', dates.lot],\\n            [\\\'MRP\\\', mrp.amount ? (mrp.currency ? mrp.currency + \\\' \\\' + mrp.amount : mrp.amount) : mrp.raw],\\n            [\\\'Distributor\\\', product.distributor],\\n            [\\\'Address\\\', product.address],\\n            [\\\'Website\\\', product.website],\\n            [\\\'Storage\\\', product.storage_instructions],\\n            [\\\'Item Number\\\', product.item_number],\\n            [\\\'Stock\\\', inventory.stock]\\n        ].filter(function(row) {\\n            return row[1] !== undefined && row[1] !== null && row[1] !== \\\'\\\';\\n        });\\n        if (!rows.length) { return \\\'<p class="empty">No structured details.</p>\\\'; }\\n        return \\\'<table class="info-table">\\\' + rows.map(function(row) {\\n            return \\\'<tr><th>\\\' + escapeHtml(row[0]) + \\\'</th><td>\\\' + textOrDash(row[1]) + \\\'</td></tr>\\\';\\n        }).join(\\\'\\\') + \\\'</table>\\\';\\n    }\\n\\n    function renderNutrition(product) {\\n        const nutrition = product.nutrition_facts || {};\\n        const rows = [];\\n        if (nutrition.calories) { rows.push(\\\'<tr><th>Calories</th><td>\\\' + textOrDash(nutrition.calories) + \\\'</td></tr>\\\'); }\\n        (nutrition.nutrients || []).forEach(function(n) {\\n            const dv = (n.dv_percent !== null && n.dv_percent !== undefined && n.dv_percent !== \\\'\\\') ? \\\' (DV \\\' + textOrDash(n.dv_percent) + \\\'%)\\\' : \\\'\\\';\\n            rows.push(\\\'<tr><td>\\\' + textOrDash(n.name) + \\\'</td><td>\\\' + textOrDash(n.amount) + dv + \\\'</td></tr>\\\');\\n        });\\n        if (!rows.length) { return \\\'<p class="empty">No nutrition info.</p>\\\'; }\\n        return \\\'<table class="info-table">\\\' + rows.join(\\\'\\\') + \\\'</table>\\\';\\n    }\\n\\n    function renderDetail(product) {\\n        const summary = product.summary || {};\\n        const serving = product.serving || {};\\n        const hero = heroFor(product);\\n        const variant = summary.variant_or_flavor || product.variant_or_flavor;\\n        const netQuantity = product.net_quantity || {};\\n        const netQtyLabel = netQuantity.value ? (netQuantity.unit ? netQuantity.value + \\\' \\\' + netQuantity.unit : netQuantity.value) : summary.net_weight;\\n        const thumbViews = (product.views || []).filter(function(v) {\\n            return v.hero && v.hero !== hero;\\n        });\\n        let html = \\\'\\\';\\n        html += \\\'<div class="detail-header">\\\';\\n        html += \\\'<div class="detail-main">\\\';\\n        if (hero) {\\n            html += \\\'<img src="\\\' + textOrDash(hero) + \\\'" class="detail-hero" alt="Product">\\\';\\n        }\\n        html += \\\'<div class="detail-title detail-meta">\\\';\\n        html += \\\'<h2>\\\' + textOrDash(summary.product_name || \\\'Unnamed Product\\\') + \\\'</h2>\\\';\\n        if (product.key) { html += \\\'<p><span class="badge">\\\' + textOrDash(product.key) + \\\'</span></p>\\\'; }\\n        const barcodeItems = (product.barcode && product.barcode.items) || [];\\n        if (barcodeItems.length) {\\n            const first = barcodeItems[0];\\n            const type = first.type ? \\\' (\\\' + textOrDash(first.type) + \\\')\\\' : \\\'\\\';\\n            html += \\\'<p>Barcode: \\\' + textOrDash(first.data) + type + \\\'</p>\\\';\\n        }\\n        html += \\\'<p>\\\' + textOrDash(summary.brand || \\\'\\\') + \\\'</p>\\\';\\n        if (variant) {\\n            html += \\\'<p>\\\' + textOrDash(variant) + \\\'</p>\\\';\\n        }\\n        if (netQtyLabel) {\\n            html += \\\'<p>\\\' + textOrDash(netQtyLabel) + \\\'</p>\\\';\\n        }\\n        if (serving.serving_size) {\\n            html += \\\'<p>Serving: \\\' + textOrDash(serving.serving_size) + \\\'</p>\\\';\\n        }\\n        html += \\\'</div>\\\';\\n        if (thumbViews.length) {\\n            html += \\\'<div class="detail-views-inline">\\\' + thumbViews.map(function(v) {\\n                return \\\'<div class="view-thumb"><img src="\\\' + textOrDash(v.hero) + \\\'" alt="\\\' + textOrDash(v.label) + \\\'"><div class="view-label">\\\' + textOrDash(v.label.charAt(0).toUpperCase() + v.label.slice(1)) + \\\'</div></div>\\\';\\n            }).join(\\\'\\\') + \\\'</div>\\\';\\n        }\\n        html += \\\'</div>\\\';\\n        html += \\\'<div class="detail-actions">\\\';\\n        html += \\\'<button id="edit-product-btn" type="button">Edit</button>\\\';\\n        html += \\\'<button id="validate-product-btn" type="button" class="secondary">Validate View</button>\\\';\\n        html += \\\'<button id="enhance-product-btn" type="button" class="ghost">AI Enhance</button>\\\';\\n        html += \\\'</div>\\\';\\n        html += \\\'<div id="validation-status" class="validation-status"></div>\\\';\\n        html += \\\'</div>\\\';\\n        html += \\\'<div class="detail-grid">\\\';\\n        html += \\\'<div class="section"><h3>Details</h3>\\\' + renderSummaryTable(product) + \\\'</div>\\\';\\n        html += \\\'<div class="section"><h3>Highlights</h3>\\\' + renderList(product.other_text) + \\\'</div>\\\';\\n        html += \\\'<div class="section"><h3>Nutrition Facts</h3>\\\' + renderNutrition(product) + \\\'</div>\\\';\\n        html += \\\'</div>\\\';\\n        detail.innerHTML = html;\\n        const editBtn = document.getElementById(\\\'edit-product-btn\\\');\\n        if (editBtn) {\\n            editBtn.addEventListener(\\\'click\\\', function() { openEditModal(currentProductIndex); });\\n        }\\n        const validateBtn = document.getElementById(\\\'validate-product-btn\\\');\\n        if (validateBtn) {\\n            validateBtn.addEventListener(\\\'click\\\', function() { validateProduct(currentProductIndex); });\\n        }\\n        const enhanceBtn = document.getElementById(\\\'enhance-product-btn\\\');\\n        if (enhanceBtn) {\\n            enhanceBtn.addEventListener(\\\'click\\\', function() { enhanceProduct(currentProductIndex); });\\n        }\\n        resetValidationStatus();\\n    }\\n\\n    function selectRow(index) {\\n        const rows = document.querySelectorAll(\\\'#product-table tbody tr\\\');\\n        if (!rows.length) { return; }\\n        currentProductIndex = Math.max(0, Math.min(index, rows.length - 1));\\n        rows.forEach(function(r) { r.classList.remove(\\\'selected\\\'); });\\n        const row = rows[currentProductIndex];\\n        row.classList.add(\\\'selected\\\');\\n        renderDetail(data[currentProductIndex]);\\n    }\\n\\n    function populateTable() {\\n        tableBody.innerHTML = \\\'\\\';\\n        if (!data.length) {\\n            tableBody.innerHTML = \\\'<tr><td colspan="6" class="empty">No products yet.</td></tr>\\\';\\n            detail.innerHTML = \\\'<p class="empty">Select a product to view details.</p>\\\';\\n            return;\\n        }\\n        data.forEach(function(product, index) {\\n            const row = document.createElement(\\\'tr\\\');\\n            row.dataset.index = index;\\n            const hero = heroFor(product);\\n            const barcodeItems = (product.barcode && product.barcode.items) || [];\\n            row.innerHTML = \\\'<td>\\\' + (index + 1) + \\\'</td>\\\' +\\n                \\\'<td class="thumb">\\\' + (hero ? \\\'<img src="\\\' + textOrDash(hero) + \\\'" alt="thumb">\\\' : \\\'\\\') + \\\'</td>\\\' +\\n                \\\'<td>\\\' + (barcodeItems.length ? textOrDash(barcodeItems[0].data) : \\\'\\\') + \\\'</td>\\\' +\\n                \\\'<td>\\\' + textOrDash((product.summary || {}).brand) + \\\'</td>\\\' +\\n                \\\'<td>\\\' + textOrDash((product.summary || {}).product_name) + \\\'</td>\\\' +\\n                \\\'<td>\\\' + textOrDash((product.inventory || {}).stock) + \\\'</td>\\\';\\n            row.addEventListener(\\\'click\\\', function() {\\n                selectRow(index);\\n            });\\n            tableBody.appendChild(row);\\n        });\\n        selectRow(currentProductIndex);\\n    }\\n\\n    function closeEditModal() {\\n        editModal.classList.add(\\\'hidden\\\');\\n        editForm.reset();\\n        modalViewOrder = [];\\n        dragIndex = null;\\n        if (editViewList) { editViewList.innerHTML = \\\'\\\'; }\\n    }\\n\\n    function openEditModal(index) {\\n        if (!data.length) { return; }\\n        const product = data[index];\\n        const summary = product.summary || {};\\n        const serving = product.serving || {};\\n        const netQuantity = product.net_quantity || {};\\n        const dates = product.dates || {};\\n        const barcodeItems = (product.barcode && product.barcode.items) || [];\\n        const firstBarcode = barcodeItems.length ? barcodeItems[0] : {};\\n\\n        editForm.dataset.index = index;\\n        document.getElementById(\\\'edit-product-name\\\').value = summary.product_name || \\\'\\\';\\n        document.getElementById(\\\'edit-brand\\\').value = summary.brand || \\\'\\\';\\n        document.getElementById(\\\'edit-variant\\\').value = summary.variant_or_flavor || product.variant_or_flavor || \\\'\\\';\\n        document.getElementById(\\\'edit-net-value\\\').value = netQuantity.value || \\\'\\\';\\n        document.getElementById(\\\'edit-net-unit\\\').value = netQuantity.unit || \\\'\\\';\\n        document.getElementById(\\\'edit-serving\\\').value = serving.serving_size || summary.serving_size || \\\'\\\';\\n        document.getElementById(\\\'edit-servings-per\\\').value = serving.servings_per_container || \\\'\\\';\\n        document.getElementById(\\\'edit-expiry\\\').value = dates.expiration || dates.best_by || summary.expiry || \\\'\\\';\\n        document.getElementById(\\\'edit-barcode\\\').value = firstBarcode.data || \\\'\\\';\\n        document.getElementById(\\\'edit-barcode-type\\\').value = firstBarcode.type || \\\'\\\';\\n\\n        const baseViews = product.views || [];\\n        modalViewOrder = baseViews.map(function(v) { return Object.assign({}, v); });\\n        if (!modalViewOrder.length) {\\n            const heroCandidate = product.hero_image || heroFor(product);\\n            if (heroCandidate) {\\n                modalViewOrder.push({ label: \\\'front\\\', hero: heroCandidate });\\n            }\\n        }\\n        buildEditViewList();\\n\\n        editModal.classList.remove(\\\'hidden\\\');\\n    }\\n\\n    function handleEditSubmit(event) {\\n        event.preventDefault();\\n        const idx = parseInt(editForm.dataset.index || \\\'0\\\', 10);\\n        if (isNaN(idx) || !data[idx]) {\\n            closeEditModal();\\n            return;\\n        }\\n        const product = data[idx];\\n        product.summary = product.summary || {};\\n        product.net_quantity = product.net_quantity || {"value": null, "unit": null};\\n        product.serving = product.serving || {"serving_size": null, "servings_per_container": null};\\n        product.dates = product.dates || {"raw_candidates": []};\\n        product.barcode = product.barcode || {"items": []};\\n\\n        function val(id) { return document.getElementById(id).value.trim(); }\\n\\n        const name = val(\\\'edit-product-name\\\');\\n        const brand = val(\\\'edit-brand\\\');\\n        const variant = val(\\\'edit-variant\\\');\\n        const netValue = val(\\\'edit-net-value\\\');\\n        const netUnit = val(\\\'edit-net-unit\\\');\\n        const servingSize = val(\\\'edit-serving\\\');\\n        const servingsPer = val(\\\'edit-servings-per\\\');\\n        const expiryVal = val(\\\'edit-expiry\\\');\\n        const barcodeVal = val(\\\'edit-barcode\\\');\\n        const barcodeType = val(\\\'edit-barcode-type\\\');\\n\\n        product.summary.product_name = name || null;\\n        product.summary.brand = brand || null;\\n        if (variant) {\\n            product.summary.variant_or_flavor = variant;\\n            product.variant_or_flavor = variant;\\n        } else {\\n            product.summary.variant_or_flavor = null;\\n            product.variant_or_flavor = null;\\n        }\\n\\n        product.net_quantity.value = netValue || null;\\n        product.net_quantity.unit = netUnit || null;\\n        product.summary.net_weight = netValue ? (netUnit ? netValue + \\\' \\\' + netUnit : netValue) : null;\\n\\n        product.serving.serving_size = servingSize || null;\\n        product.serving.servings_per_container = servingsPer || null;\\n        product.summary.serving_size = servingSize || null;\\n\\n        if (expiryVal) {\\n            product.dates.expiration = expiryVal;\\n            product.summary.expiry = expiryVal;\\n        } else {\\n            product.summary.expiry = null;\\n        }\\n\\n        if (barcodeVal) {\\n            product.barcode.items = [{ type: barcodeType || null, data: barcodeVal }];\\n        } else {\\n            product.barcode.items = [];\\n        }\\n\\n        if (modalViewOrder && modalViewOrder.length) {\\n            product.views = modalViewOrder.map(function(v) { return Object.assign({}, v); });\\n            const primaryView = product.views[0];\\n            if (primaryView) {\\n                product.hero_image = primaryView.hero || primaryView.ocr || primaryView.masked || product.hero_image || null;\\n            }\\n        }\\n\\n        closeEditModal();\\n        populateTable();\\n    }\\n\\n    if (enhanceCloseBtn) {\\n        enhanceCloseBtn.addEventListener(\\\'click\\\', closeEnhanceModal);\\n    }\\n    if (enhanceDismissBtn) {\\n        enhanceDismissBtn.addEventListener(\\\'click\\\', closeEnhanceModal);\\n    }\\n    if (enhanceModal) {\\n        enhanceModal.addEventListener(\\\'click\\\', function(event) {\\n            if (event.target === enhanceModal) {\\n                closeEnhanceModal();\\n            }\\n        });\\n    }\\n\\n    populateTable();\\n    editForm.addEventListener(\\\'submit\\\', handleEditSubmit);\\n    editCancelBtn.addEventListener(\\\'click\\\', closeEditModal);\\n    editCloseBtn.addEventListener(\\\'click\\\', closeEditModal);\\n    if (downloadCatalogBtn) {\\n        downloadCatalogBtn.addEventListener(\\\'click\\\', function() {\\n            const payload = { products: data };\\n            const blob = new Blob([JSON.stringify(payload, null, 2)], { type: \\\'application/json\\\' });\\n            const url = URL.createObjectURL(blob);\\n            const a = document.createElement(\\\'a\\\');\\n            a.href = url;\\n            a.download = \\\'catalog_overrides.json\\\';\\n            document.body.appendChild(a);\\n            a.click();\\n            document.body.removeChild(a);\\n            URL.revokeObjectURL(url);\\n        });\\n    }\\n\' + \'\\n    '
    js_block = js_block.rstrip()
    js_block = re.sub(r"\\n'\s*\+\s*'\\n\s*$", "\\n", js_block)
    js_block = js_block.rstrip() + '\n'
    if False:
        html_template = re.sub(
            r'<script>\n.*?</script>',
            '<script>\n' + js_block + '\n    </script>',
            html_template,
            count=1,
            flags=re.S
        )

    html = (
        html_template
        .replace('__CATALOG_JSON__', catalog_json)
        .replace('__VALIDATION_ENDPOINT__', validation_endpoint_json)
        .replace('__ENHANCE_ENDPOINT__', enhance_endpoint_json)
        .replace('__AI_ENHANCE_JSON__', ai_enhance_json)
        .replace('__CRITICAL_FIELDS__', critical_fields_json)
    )
    with html_path.open('w', encoding='utf-8') as f:
        f.write(html)
def _prioritize_views(processed: list[dict]) -> list[dict]:
    if len(processed) <= 1:
        if processed:
            processed[0]["view_label"] = "front"
        return processed

    for res in processed:
        res["view_label"] = "other"
        data = res.get("data") or {}
        res["area_ratio"] = data.get("product_area_ratio") or 0.0

    barcode_items: list[dict] = []
    hero_candidates: list[dict] = []
    best_front = None
    best_score = -1.0

    for res in processed:
        data = res.get("data") or {}
        if (data.get("barcode") or {}).get("items"):
            barcode_items.append(res)
        if _has_hero_output(res):
            hero_candidates.append(res)
        score = _front_score(data)
        total_text = (data.get("text_metrics") or {}).get("total_text_area_ratio") or 0.0
        total_text_score = total_text * 200.0
        net_weight = (data.get("summary") or {}).get("net_weight")
        if net_weight:
            total_text_score += 25.0
        score += total_text_score
        if score > best_score:
            best_front = res
            best_score = score

    view_metrics = {}
    for res in processed:
        data = res.get("data") or {}
        brand_score = _brand_signal_score(data)
        nutrition_score = _nutrition_signal_score(data)
        view_metrics[id(res)] = {
            "brand_score": brand_score,
            "nutrition_score": nutrition_score,
            "brand_present": brand_score >= 25.0,
            "nutrition_heavy": nutrition_score >= 12
        }

    hero_non_barcode = [
        res for res in hero_candidates
        if not ((res.get("data") or {}).get("barcode") or {}).get("items")
    ]

    def _front_sort_key(res):
        metrics = view_metrics.get(id(res), {})
        data = res.get("data") or {}
        return (
            1 if metrics.get("brand_present") else 0,
            metrics.get("brand_score", 0.0),
            -metrics.get("nutrition_score", 0.0),
            (data.get("product_area_ratio") or 0.0),
            _front_score(data),
            -(res.get("order", 0))
        )

    if hero_non_barcode:
        hero_non_barcode.sort(key=_front_sort_key, reverse=True)
        for candidate in hero_non_barcode:
            metrics = view_metrics.get(id(candidate), {})
            if metrics.get("nutrition_heavy") and not metrics.get("brand_present"):
                continue
            best_front = candidate
            break
        if best_front is None and hero_non_barcode:
            best_front = hero_non_barcode[0]
    elif hero_candidates:
        hero_candidates.sort(key=_front_sort_key, reverse=True)
        for candidate in hero_candidates:
            metrics = view_metrics.get(id(candidate), {})
            if metrics.get("nutrition_heavy") and not metrics.get("brand_present"):
                continue
            best_front = candidate
            break
        if best_front is None and hero_candidates:
            best_front = hero_candidates[0]
    else:
        area_candidates = [
            res for res in processed
            if not _is_exp_panel(res.get("data"))
        ]
        if area_candidates:
            best_area = max(area_candidates, key=_front_candidate_rank)
            rank = _front_candidate_rank(best_area)
            if any(val > 0 for val in rank):
                best_front = best_area

    if best_front is None:
        best_front = processed[0]
    else:
        bf_metrics = view_metrics.get(id(best_front), {})
        bf_barcode = ((best_front.get("data") or {}).get("barcode") or {}).get("items")
        if bf_barcode:
            non_barcode_options = [
                res for res in processed
                if not (((res.get("data") or {}).get("barcode") or {}).get("items"))
            ]
            if non_barcode_options:
                non_barcode_options.sort(key=_front_sort_key, reverse=True)
                for candidate in non_barcode_options:
                    metrics = view_metrics.get(id(candidate), {})
                    if metrics.get("nutrition_heavy") and not metrics.get("brand_present"):
                        continue
                    best_front = candidate
                    bf_metrics = metrics
                    break

        if bf_metrics and not bf_metrics.get("brand_present"):
            alt_with_brand = [res for res in processed if view_metrics.get(id(res), {}).get("brand_present")]
            if alt_with_brand:
                alt_with_brand.sort(key=_front_sort_key, reverse=True)
                best_front = alt_with_brand[0]

    best_front["view_label"] = "front"

    back_candidate = None
    for res in barcode_items:
        if res is not best_front:
            back_candidate = res
            break
    if back_candidate is None:
        fallback_back = [
            res for res in processed
            if res is not best_front
        ]
        if fallback_back:
            back_candidate = max(
                fallback_back,
                key=lambda r: (
                    _info_score(r.get("data") or {}),
                    (r.get("data") or {}).get("product_area_ratio") or 0.0,
                    -r.get("order", 0)
                )
            )

    if back_candidate:
        back_candidate["view_label"] = "back"

    ordered = []
    used_ids = set()
    if best_front:
        ordered.append(best_front)
        used_ids.add(id(best_front))
    if back_candidate and id(back_candidate) not in used_ids:
        ordered.append(back_candidate)
        used_ids.add(id(back_candidate))

    def _view_priority(res: dict) -> int:
        data = res.get("data") or {}
        if _is_exp_panel(data):
            return 0
        if _is_nutrition_panel(data):
            return 1
        return 2

    remaining = [res for res in processed if id(res) not in used_ids]
    remaining.sort(key=lambda r: (_view_priority(r), -_info_score(r.get("data") or {}), r.get("order", 0)))

    expiry_count = 0
    nutrition_count = 0
    side_count = 0
    for res in remaining:
        data = res.get("data") or {}
        if _is_exp_panel(data):
            expiry_count += 1
            label = "expiry" if expiry_count == 1 else f"expiry{expiry_count}"
        elif _is_nutrition_panel(data):
            nutrition_count += 1
            label = "nutrition" if nutrition_count == 1 else f"nutrition{nutrition_count}"
        else:
            side_count += 1
            label = "side" if side_count == 1 else f"side{side_count}"
        res["view_label"] = label
        ordered.append(res)
        used_ids.add(id(res))

    return ordered

_OUTPUT_SUFFIXES = {
    "ocr_image": "ocr.jpeg",
    "masked_image": "masked.jpeg",
    "anchor_image": "anchor.jpeg",
    "hero_image": "hero.jpeg",
    "hero_cutout_image": "hero_cutout.jpeg"
}

def _rename_artifacts_for_view(res: dict, view_label: str, product_slug: str):
    outputs = res["data"].get("outputs", {})
    new_outputs = {}

    for key, suffix in _OUTPUT_SUFFIXES.items():
        path_str = outputs.get(key)
        if not path_str:
            new_outputs[key] = None
            continue
        old_path = Path(path_str)
        new_path = OUT_DIR / f"{product_slug}_{view_label}_{suffix}"
        try:
            if old_path.resolve() != new_path.resolve():
                if new_path.exists():
                    new_path.unlink()
                old_path.rename(new_path)
        except FileNotFoundError:
            new_outputs[key] = None
            continue
        new_outputs[key] = str(new_path.resolve())

    outputs.update(new_outputs)
    res["data"]["outputs"] = outputs

    res["masked_img_path"] = Path(new_outputs["masked_image"]) if new_outputs.get("masked_image") else None
    res["anchor_img_path"] = Path(new_outputs["anchor_image"]) if new_outputs.get("anchor_image") else None

    old_json = res["raw_json_path"]
    new_json = OUT_DIR / f"{product_slug}_{view_label}.json"
    if old_json.resolve() != new_json.resolve():
        if new_json.exists():
            new_json.unlink()
        old_json.rename(new_json)
    res["raw_json_path"] = new_json


# ==================== MAIN ====================
def _process_single_image(img_path: Path, temp_dir: Path) -> dict:
    print(f"image={img_path}")
    orig=Image.open(img_path); orig=ImageOps.exif_transpose(orig).convert("RGB")
    proc,scale=resize_limit(orig, MAX_SIDE)
    if scale!=1.0: print(f"resized_to={proc.size[0]}x{proc.size[1]} max_side={MAX_SIDE}")

    # 1) YOLO
    t0=time.time(); dets, yolo_note = run_yolo_world(proc); t1=time.time()

    # 2) OCR
    lines, ocr_note = run_easyocr(proc); t2=time.time()

    # 3) Barcode
    bc=decode_barcodes(proc); t3=time.time()

    # Restore coords to original image space if resized
    if scale!=1.0:
        upscale_back(dets,scale)
        upscale_back(lines,scale)

    # Overlays (skip detection render per user preference)
    det_img = None
    ocr_img = temp_dir / f"{img_path.stem}_ocr.jpeg"
    draw_lines(orig, lines, ocr_img)

    # Masked preview image
    text_boxes = [l["bbox"] for l in lines] if lines else []
    if text_boxes:
        mask = build_text_mask(orig, text_boxes, pad=4, dilate=7)
        masked = apply_mask_keep_text(orig, mask, bg_color=(255,255,255))
        masked_img_path = temp_dir / f"{img_path.stem}_masked.jpeg"
        masked.save(masked_img_path, quality=95)
    else:
        masked_img_path = None

    hero_path = None
    hero_cutout_path = None
    cut_note = None
    product_area_ratio = 0.0

    best_box = _pick_best_product_box(dets)
    hero_det = None
    hero_source = "yolo"
    if best_box is not None:
        hero_det = best_box
        bw = max(1, best_box["bbox"]["w"])
        bh = max(1, best_box["bbox"]["h"])
        product_area_ratio = (bw * bh) / float(orig.width * orig.height)
    else:
        hero_source = "fallback"
        text_records = [l.get("bbox") for l in (lines or []) if isinstance(l, dict)]
        bc_boxes = [item.get("bbox") for item in (bc.get("items") or []) if isinstance(item, dict)]
        combined_records = []
        if text_records:
            combined_records.extend(text_records)
        if bc_boxes:
            combined_records.extend(bc_boxes)
        fallback_bbox = _union_bbox(combined_records, orig.width, orig.height, pad=60)
        if fallback_bbox is None and text_records:
            fallback_bbox = _union_bbox(text_records, orig.width, orig.height, pad=55)
        if fallback_bbox is None and bc_boxes:
            fallback_bbox = _union_bbox(bc_boxes, orig.width, orig.height, pad=80)
        if fallback_bbox is not None:
            x0, y0, fw, fh = fallback_bbox
            min_w = max(int(orig.width * 0.4), 260)
            min_h = max(int(orig.height * 0.4), 260)
            if fw < min_w:
                cx = x0 + fw // 2
                fw = min(orig.width, max(min_w, fw))
                x0 = max(0, cx - fw // 2)
                if x0 + fw > orig.width:
                    x0 = orig.width - fw
            if fh < min_h:
                cy = y0 + fh // 2
                fh = min(orig.height, max(min_h, fh))
                y0 = max(0, cy - fh // 2)
                if y0 + fh > orig.height:
                    y0 = orig.height - fh
            x0, y0, fw, fh = clamp_box(x0, y0, fw, fh, orig.width, orig.height)
            hero_det = {"bbox": {"x": x0, "y": y0, "w": fw, "h": fh}}
            product_area_ratio = (fw * fh) / float(orig.width * orig.height)
        else:
            hero_source = "full"

    if hero_det is not None:
        hero_img = _make_hero_from_box(orig, hero_det, size=HERO_SIZE)
        hero_path = temp_dir / f"{img_path.stem}_hero.jpeg"
        hero_img.save(hero_path, quality=95)
        cut_img, cut_note_tmp = _cutout_on_white(orig, hero_det, size=HERO_SIZE)
        if cut_img is not None:
            hero_cutout_path = temp_dir / f"{img_path.stem}_hero_cutout.jpeg"
            cut_img.save(hero_cutout_path, quality=95)
            cut_note = cut_note_tmp
        if hero_source == "fallback" and cut_note is None:
            cut_note = "fallback_bbox_union"
    else:
        # ultimate fallback: resize full image onto white square
        canvas = Image.new("RGB", (HERO_SIZE, HERO_SIZE), (255, 255, 255))
        scale = min(HERO_SIZE / orig.width, HERO_SIZE / orig.height)
        new_w = max(1, int(round(orig.width * scale)))
        new_h = max(1, int(round(orig.height * scale)))
        resized = orig.resize((new_w, new_h), Image.LANCZOS)
        off_x = (HERO_SIZE - new_w) // 2
        off_y = (HERO_SIZE - new_h) // 2
        canvas.paste(resized, (off_x, off_y))
        hero_path = temp_dir / f"{img_path.stem}_hero.jpeg"
        canvas.save(hero_path, quality=95)

    # Heuristic summary (optional)
    H=orig.size[1]; W=orig.size[0]
    summary=find_best(lines,W,H)
    avg_conf=round(sum(l["conf"] for l in lines)/len(lines),4) if lines else 0.0

    total_area = W * H if W and H else 1
    max_text_area_ratio = 0.0
    total_text_area = 0.0
    headline_len = 0
    top_char_count = 0
    for l in lines or []:
        txt = str(l.get("text","")).strip()
        bbox = l.get("bbox") or {}
        w = max(0, int(bbox.get("w",0)))
        h = max(0, int(bbox.get("h",0)))
        area = w * h
        total_text_area += area
        ratio = area / total_area if total_area else 0.0
        if ratio > max_text_area_ratio:
            max_text_area_ratio = ratio
        if bbox.get("y",0) < H * 0.45:
            top_char_count += len(txt)
            if len(txt) > headline_len:
                headline_len = len(txt)
    text_metrics = {
        "max_text_area_ratio": max_text_area_ratio,
        "total_text_area_ratio": total_text_area / total_area if total_area else 0.0,
        "headline_len": headline_len,
        "top_char_count": top_char_count,
        "line_count": len(lines)
    }

    if product_area_ratio <= 0.0:
        fallback_area = max_text_area_ratio * 1.4
        total_ratio = text_metrics["total_text_area_ratio"] or 0.0
        fallback_area = max(fallback_area, total_ratio * 1.1)
        product_area_ratio = min(0.7, fallback_area)
    product_area_ratio = max(0.0, min(product_area_ratio, 0.9))

    # Build anchor crop from OCR lines (single crop to keep cost low)
    anchor_crop, anchor_meta = _anchor_crop_from_ocr(orig, lines)
    if anchor_crop is not None:
        anchor_img_path = temp_dir / f"{img_path.stem}_anchor.jpeg"
        anchor_crop.save(anchor_img_path, quality=95)
    else:
        anchor_img_path = None

    anchor_info = _extract_anchor_info(lines)
    if summary.get("expiry") is None:
        summary["expiry"] = anchor_info["dates"].get("expiration") or anchor_info["dates"].get("best_by")

    # JSON dump (raw pipeline output)
    data={
        "file": img_path.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00","Z"),
        "models":{"detector":YOLO_MODEL,"ocr":"easyocr"},
        "timings":{"yolo_s":round(t1-t0,3),"easyocr_s":round(t2-t1,3),"barcode_s":round(t3-t2,3)},
        "detections":{"yolo_world":dets,"classes":YOLO_CLASSES,"yolo_note":yolo_note},
        "ocr":{"line_count":len(lines),"avg_conf":avg_conf,"lines":lines,"ocr_note":ocr_note},
        "barcode":bc,
        "summary":summary,
        "text_metrics": text_metrics,
        "product_area_ratio": product_area_ratio,
        "image_size": {"width": orig.width, "height": orig.height},
        "outputs":{
            "detect_image": str(det_img.resolve()) if det_img else None,
            "ocr_image": str(ocr_img.resolve()),
            "masked_image": str(masked_img_path.resolve()) if masked_img_path else None,
            "anchor_image": str(anchor_img_path.resolve()) if anchor_img_path else None,
            "hero_image": str(hero_path.resolve()) if hero_path else None,
            "hero_cutout_image": str(hero_cutout_path.resolve()) if hero_cutout_path else None
        },
        "notes":{"cutout": cut_note, "anchor": anchor_meta},
        "anchor_extraction": anchor_info
    }
    out_json = temp_dir / f"{img_path.stem}.json"
    with out_json.open("w",encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "data": data,
        "raw_json_path": out_json,
        "img_path": img_path,
        "masked_img_path": masked_img_path,
        "anchor_img_path": anchor_img_path
    }

def _process_product_group(group: dict, bill_summaries: list[dict]):
    product_slug = group["slug"]
    images = group["images"]
    temp_dir = TEMP_DIR / product_slug
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"product={product_slug} images={len(images)}")

    processed=[]
    for idx, img_path in enumerate(images):
        result = _process_single_image(img_path, temp_dir)
        result["order"] = idx
        processed.append(result)

    processed = _prioritize_views(processed)

    if not processed:
        print(f"llm_note={product_slug}: no images processed")
        return

    link_report = None
    if bill_summaries:
        link_report = link_supplier_data(processed, bill_summaries)
        if link_report.get("linked"):
            print(f"supplier_links_added[{product_slug}]={link_report['linked']}")
        if link_report.get("unmatched"):
            unmatched_preview = link_report["unmatched"][:5]
            print(f"unmatched_barcodes[{product_slug}]={len(link_report['unmatched'])} sample={unmatched_preview}")
        if link_report.get("errors"):
            print(f"bill_errors[{product_slug}]={link_report['errors']}")

    items=[]
    final_sequence=[]
    for res in processed:
        view_label = res.get("view_label","other")
        res["data"]["view_label"] = view_label
        _rename_artifacts_for_view(res, view_label, product_slug)
        try:
            with res["raw_json_path"].open("w", encoding="utf-8") as f:
                json.dump(res["data"], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"warn[{product_slug}]=view_label_write_failed file={res['raw_json_path']} err={e}")
        items.append({
            "raw_json_path": res["raw_json_path"],
            "img_path": res["img_path"],
            "view_label": view_label
        })
        hero_primary = res["data"]["outputs"].get("hero_cutout_image") or res["data"]["outputs"].get("hero_image")
        final_sequence.append((view_label, hero_primary))

    if len(items) > 1:
        output_name = f"{product_slug}_combined_structured.json"
    else:
        output_name = f"{product_slug}_structured.json"

    structured_data, err = structurize_with_gpt4o(
        items,
        model=OPENAI_VISION_MODEL,
        output_name=output_name
    )
    if err:
        label = "llm_note_combined" if len(items) > 1 else "llm_note"
        print(f"{label}[{product_slug}]={err}")
        structured_data = None
    else:
        print(f"structured_json[{product_slug}]={'combined' if len(items)>1 else 'single'} generated")

    try:
        for leftover in list(temp_dir.iterdir()):
            if leftover.is_file():
                leftover.unlink(missing_ok=True)
        temp_dir.rmdir()
    except Exception:
        pass

    print(f"final_view_order ({product_slug}):")
    for view_label, path in final_sequence:
        if path:
            print(f"  {view_label}: {path}")
        else:
            print(f"  {view_label}: (no artifact)")

    catalog_path = OUT_DIR / f"{product_slug}_catalog.json"
    combined = build_combined_product(processed, link_report, catalog_path, structured_data, product_id=product_slug)
    if combined:
        ai_report = _generate_ai_enhance_report(combined, structured_data or combined)
        if ai_report:
            combined["ai_enhance_report"] = ai_report
            try:
                with catalog_path.open("w", encoding="utf-8") as f:
                    json.dump(combined, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                print(f"ai_enhance_note=write_failed[{product_slug}]={exc}")
            try:
                with (OUT_DIR / f"{product_slug}_ai_enhance.json").open("w", encoding="utf-8") as f:
                    json.dump(ai_report, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                print(f"ai_enhance_note=report_write_failed[{product_slug}]={exc}")
        _submit_product_to_api(combined)
        update_global_catalog(combined, OUT_DIR / "catalog.json", combined.get("views", {}))
    print(f"catalog_json={catalog_path}")
    print(f"catalog_html={(OUT_DIR / 'catalog.html')}\n")


def main(selected_groups: list[dict]|None = None):
    groups = selected_groups or collect_product_groups()
    bill_summaries = parse_supplier_bills()
    for group in groups:
        _process_product_group(group, bill_summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR + product catalog pipeline")
    parser.add_argument("--apply-overrides", dest="overrides", help="Path to catalog overrides JSON", default=None)
    parser.add_argument("--product-filter", dest="product_filter", help="Comma separated slugs to process", default=None)
    args = parser.parse_args()

    if args.overrides:
        apply_catalog_overrides(Path(args.overrides))
        sys.exit(0)

    if args.product_filter:
        os.environ["PRODUCT_FILTER"] = args.product_filter
        new_filter = {s.strip().lower() for s in args.product_filter.split(",") if s.strip()}
        PRODUCT_FILTER.clear()
        PRODUCT_FILTER.update(new_filter)

    main()
