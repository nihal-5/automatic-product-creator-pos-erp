# pos_worker — photograph a product, get a catalogue draft

> **New here?** `docs/SYSTEM_OVERVIEW.md` in `Retail-intel/POS` (branch
> `ai-platform`) explains how this worker, the POS, and the voice agent fit
> together. This README covers the worker alone.

Reads a product photograph and creates a **draft** entry in the POS
(`Retail-intel/POS`) for a person to review.

    export OPENAI_API_KEY=...
    export POS_BASE_URL=http://localhost:8000

    python -m pos_worker.run photo.jpg --category-id 125 --unit-id 170
    python -m pos_worker.run ./inbox/                    # a whole folder
    python -m pos_worker.run photo.jpg --dry-run         # extract only

## Why this exists alongside `ocr_pipeline.py`

`ocr_pipeline.py` in this repo is the full stack: YOLO for region proposal,
EasyOCR and PaddleOCR for text, a vision model on top. It is more precise on
dense nutrition panels and small print, and it needs PyTorch — roughly two
gigabytes of dependency.

`pos_worker` needs `openai`, `Pillow` and `httpx`. It sends the photograph to a
vision model and asks for the fields the POS actually stores. For a shop owner
photographing a product to create a catalogue entry, that is usually enough, and
it runs on a laptop with no GPU.

**Start here. Move to the full pipeline when measured accuracy on real shelf
photographs says you need to, not before.**

## Why it is a separate service, not part of the POS

The POS serves 116 endpoints, none of which need image libraries. Putting the
vision stack inside it would add gigabytes to that container and slow every cold
start. The two share only an HTTP contract:

    photograph
       -> extract.py         vision model reads the pack
       -> submit.py          POST /v1/products/vision-intake
       -> POS                resolves references, creates a DRAFT

## What it will not do

**It never invents a value.** Unread fields come back `null`, and the POS lists
them in `needs_review`. In particular it never derives a selling price from MRP
— margin is a commercial decision, not something readable off a package.

**It never creates categories or units.** Those vocabularies are curated. An
unrecognised one falls back to `--category-id` / `--unit-id` and is flagged.

**A barcode that fails its EAN-13 check digit is dropped, not used.** OCR
misreads digits; a wrong barcode would attach another manufacturer's product to
this record. The POS assigns a valid in-store code instead and flags it.

## Responses you should expect

| Outcome | Meaning |
|---|---|
| `201` + `needs_review` | Draft created. The listed fields need a person. |
| `409 BARCODE_EXISTS` | Already in the catalogue. Not a failure. |
| `409 POTENTIAL_VARIANT_DETECTED` | Looks like a variant of something stocked — the response names the existing product, so link it rather than duplicating. |
| `422 VISION_INTAKE_NO_NAME` / `NO_BRAND` | Nothing usable was read. Re-photograph. |

Exit codes: `0` all submitted, `1` one or more not created, `2` misconfigured.

## Verified

Against a running POS with `gpt-4o`, on a generated Indian grocery label:
brand, product name, net weight and unit, MRP, HSN code, GST percentage, the
full six-row nutrition panel and the ingredient list were all read correctly.
The deliberately invalid barcode on the test label was dropped and flagged
rather than failing the intake.
