from __future__ import annotations

import csv
import io
import json
from typing import List
from .models import Supplier, SKU, Location, InventoryRecord, SaleRecord


def load_suppliers(data: bytes | None) -> List[Supplier]:
    if not data:
        return []
    payload = json.loads(data.decode("utf-8"))
    out = []
    for item in payload:
        out.append(
            Supplier(
                supplier_id=item["supplier_id"],
                name=item.get("name", item["supplier_id"]),
                lead_time_days=int(item.get("lead_time_days", 0)),
                min_order_qty=int(item.get("min_order_qty", 0)),
                price_band=item.get("price_band", {}),
            )
        )
    return out


def load_skus(data: bytes | None) -> List[SKU]:
    if not data:
        return []
    payload = json.loads(data.decode("utf-8"))
    out = []
    for item in payload:
        out.append(
            SKU(
                sku=item["sku"],
                supplier_id=item["supplier_id"],
                case_size=int(item.get("case_size", 1)),
            )
        )
    return out


def _load_inventory_json(payload) -> List[InventoryRecord]:
    out = []
    for item in payload:
        out.append(
            InventoryRecord(
                sku=item["sku"],
                location_id=item["location_id"],
                on_hand=int(item.get("on_hand", 0)),
                inbound=int(item.get("inbound", 0)),
            )
        )
    return out


def load_inventory(data: bytes | None) -> List[InventoryRecord]:
    if not data:
        return []
    text = data.decode("utf-8")
    if text.strip().startswith("["):
        payload = json.loads(text)
        return _load_inventory_json(payload)
    reader = csv.DictReader(io.StringIO(text))
    out = []
    for row in reader:
        out.append(
            InventoryRecord(
                sku=row["sku"],
                location_id=row["location_id"],
                on_hand=int(row.get("on_hand", 0)),
                inbound=int(row.get("inbound", 0)),
            )
        )
    return out


def _load_sales_json(payload) -> List[SaleRecord]:
    out = []
    for item in payload:
        out.append(
            SaleRecord(
                sku=item["sku"],
                location_id=item["location_id"],
                qty=int(item.get("qty", 0)),
                days=int(item.get("days", 1)),
            )
        )
    return out


def load_sales(data: bytes | None) -> List[SaleRecord]:
    if not data:
        return []
    text = data.decode("utf-8")
    if text.strip().startswith("["):
        payload = json.loads(text)
        return _load_sales_json(payload)
    reader = csv.DictReader(io.StringIO(text))
    out = []
    for row in reader:
        out.append(
            SaleRecord(
                sku=row["sku"],
                location_id=row["location_id"],
                qty=int(row.get("qty", 0)),
                days=int(row.get("days", 1)),
            )
        )
    return out


def load_locations(data: bytes | None) -> List[Location]:
    if not data:
        return []
    payload = json.loads(data.decode("utf-8"))
    out = []
    for item in payload:
        out.append(
            Location(
                location_id=item["location_id"],
                kind=item.get("kind", "store"),
                capacity=int(item.get("capacity", 0)),
                safety_stock=int(item.get("safety_stock", 0)),
            )
        )
    return out
