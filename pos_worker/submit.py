"""Send extracted product data to the POS as a draft.

The worker and the POS are separate services that share only an HTTP contract.
That is deliberate: the vision stack needs image libraries and, on the full
path, PyTorch. Putting it inside the POS would add gigabytes to a container
serving 116 endpoints that need none of it.

The POS side is `POST /v1/products/vision-intake`, which resolves brand,
category and unit references, strips anything the model could not read, and
creates the product in DRAFT with `needs_review` listing what a person still has
to supply. Prices are always on that list -- a photograph cannot establish what
you paid for something.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

POS_BASE_URL = os.getenv("POS_BASE_URL", "http://localhost:8000")
INTAKE_PATH = "/v1/products/vision-intake"


class SubmitError(RuntimeError):
    """The POS rejected the draft."""

    def __init__(self, status: int, payload: Any) -> None:
        self.status = status
        self.payload = payload
        super().__init__(f"POS returned {status}: {payload}")


def submit(
    extraction: Dict[str, Any],
    *,
    base_url: str = POS_BASE_URL,
    fallback_category_id: Optional[int] = None,
    fallback_unit_id: Optional[int] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Create a draft product from an extraction.

    Args:
        extraction: Output of `extract.extract()`.
        fallback_category_id: Used when the read category matches nothing. The
            POS never creates categories from a photograph -- that is how a
            catalogue fills up with 'Category 001'.
        fallback_unit_id: Same, for units of measure.

    Returns:
        {"product": {...}, "needs_review": [...]}

    Raises:
        SubmitError: on a non-2xx response. A 409 is normal and meaningful: the
            barcode already exists, or the product looks like a variant of one
            already stocked. Both carry the existing product's id so a reviewer
            can open it instead of creating a near-duplicate.
    """
    body: Dict[str, Any] = {"extraction": extraction}
    if fallback_category_id is not None:
        body["fallback_category_id"] = fallback_category_id
    if fallback_unit_id is not None:
        body["fallback_unit_id"] = fallback_unit_id

    response = httpx.post(
        base_url.rstrip("/") + INTAKE_PATH, json=body, timeout=timeout
    )
    try:
        payload = response.json()
    except ValueError:
        payload = response.text

    if not response.is_success:
        raise SubmitError(response.status_code, payload)
    return payload


def describe_result(result: Dict[str, Any]) -> str:
    """One-line summary of what the POS created."""
    product = result.get("product") or {}
    review = result.get("needs_review") or []
    line = f"{product.get('item_code')}  {product.get('name')}  [{product.get('status')}]"
    if review:
        line += f"\n  needs review: {', '.join(review)}"
    return line
