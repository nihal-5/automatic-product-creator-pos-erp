from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Supplier:
    supplier_id: str
    name: str
    lead_time_days: int
    min_order_qty: int
    price_band: Dict[str, float]  # sku -> unit price


@dataclass
class SKU:
    sku: str
    supplier_id: str
    case_size: int


@dataclass
class Location:
    location_id: str
    kind: str  # store, dc, online
    capacity: int
    safety_stock: int


@dataclass
class InventoryRecord:
    sku: str
    location_id: str
    on_hand: int
    inbound: int = 0


@dataclass
class SaleRecord:
    sku: str
    location_id: str
    qty: int
    days: int  # over how many days this qty was sold


@dataclass
class PlanLine:
    sku: str
    qty: int
    unit_cost: float


@dataclass
class Allocation:
    location_id: str
    sku: str
    qty: int


@dataclass
class PurchaseOrder:
    supplier_id: str
    status: str
    eta_days: int
    lines: List[PlanLine]
    allocations: List[Allocation]


@dataclass
class PlanResult:
    purchase_orders: List[PurchaseOrder]
    notes: List[str]
