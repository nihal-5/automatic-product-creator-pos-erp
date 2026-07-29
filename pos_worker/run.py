#!/usr/bin/env python3
"""Photograph a product, get a draft catalogue entry in the POS.

    python -m pos_worker.run photo.jpg
    python -m pos_worker.run ./inbox/ --category-id 125 --unit-id 170
    python -m pos_worker.run photo.jpg --dry-run     # extract only, do not submit

Needs OPENAI_API_KEY, and a running POS (POS_BASE_URL, default localhost:8000).

Exit codes: 0 all submitted, 1 one or more failed, 2 misconfigured.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pos_worker.extract import extract  # noqa: E402
from pos_worker.submit import SubmitError, describe_result, submit  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def collect_images(target: Path) -> List[Path]:
    """Return the image files to process, sorted."""
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(
            p for p in target.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    raise FileNotFoundError(target)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path", type=Path, help="an image file or a directory of them")
    parser.add_argument("--base-url", default=os.getenv("POS_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--category-id", type=int, default=None,
                        help="category to fall back to when the read one is unknown")
    parser.add_argument("--unit-id", type=int, default=None,
                        help="unit to fall back to when the read one is unknown")
    parser.add_argument("--dry-run", action="store_true",
                        help="extract and print, submit nothing")
    args = parser.parse_args(argv)

    if not (args.dry_run or os.getenv("POS_BASE_URL") or args.base_url):
        print("No POS base URL. Set POS_BASE_URL or pass --base-url.")
        return 2
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.")
        return 2

    try:
        images = collect_images(args.path)
    except FileNotFoundError:
        print(f"No such file or directory: {args.path}")
        return 2
    if not images:
        print(f"No images in {args.path} (looked for {', '.join(sorted(IMAGE_SUFFIXES))})")
        return 2

    print(f"{len(images)} image(s) to process\n")
    failures = 0

    for image in images:
        print(f"{image.name}")
        try:
            extraction = extract(image)
        except Exception as exc:
            print(f"  extraction failed: {exc}")
            failures += 1
            continue

        read = [k for k, v in extraction.items()
                if not k.startswith("_") and v not in (None, [], "")]
        print(f"  read: {extraction.get('brand')} / {extraction.get('name')}")
        print(f"  fields with values: {len(read)}")

        if args.dry_run:
            print(json.dumps(extraction, indent=2, ensure_ascii=False)[:1200])
            print()
            continue

        try:
            result = submit(
                extraction,
                base_url=args.base_url,
                fallback_category_id=args.category_id,
                fallback_unit_id=args.unit_id,
            )
        except SubmitError as exc:
            # 409 is a real outcome, not a crash: the product already exists, or
            # it resembles one that does. Both name the existing product.
            error = exc.payload.get("error", {}) if isinstance(exc.payload, dict) else {}
            code = error.get("code", exc.status)
            print(f"  not created ({code})")
            if error.get("message"):
                print(f"    {error['message'][:160]}")
            failures += 1
            print()
            continue
        except Exception as exc:
            print(f"  submit failed: {exc}")
            failures += 1
            print()
            continue

        print(f"  {describe_result(result)}")
        print()

    if failures:
        print(f"{len(images) - failures} created, {failures} not created.")
        return 1
    if not args.dry_run:
        print(f"All {len(images)} created as drafts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
