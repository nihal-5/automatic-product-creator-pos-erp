from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple
from .models import (
    Supplier,
    SKU,
    Location,
    InventoryRecord,
    SaleRecord,
    PlanResult,
    PurchaseOrder,
    PlanLine,
    Allocation,
)


def _daily_sales(sales: List[SaleRecord]) -> Dict[Tuple[str, str], float]:
    daily: Dict[Tuple[str, str], float] = {}
    for s in sales:
        key = (s.sku, s.location_id)
        daily[key] = daily.get(key, 0.0) + (s.qty / max(1, s.days))
    return daily


def _inventory_lookup(inv: List[InventoryRecord]) -> Dict[Tuple[str, str], InventoryRecord]:
    lookup: Dict[Tuple[str, str], InventoryRecord] = {}
    for rec in inv:
        lookup[(rec.sku, rec.location_id)] = rec
    return lookup


def plan(
    suppliers: List[Supplier],
    skus: List[SKU],
    locations: List[Location],
    inventory: List[InventoryRecord],
    sales: List[SaleRecord],
    cover_buffer_days: int = 2,
) -> PlanResult:
    supplier_map = {s.supplier_id: s for s in suppliers}
    sku_map = {s.sku: s for s in skus}
    loc_map = {l.location_id: l for l in locations}

    daily = _daily_sales(sales)
    inv_lookup = _inventory_lookup(inventory)

    # Aggregate need per supplier
    supplier_lines: Dict[str, List[PlanLine]] = defaultdict(list)
    supplier_allocations: Dict[str, List[Allocation]] = defaultdict(list)
    notes: List[str] = []

    for sku_id, sku in sku_map.items():
        supplier_id = sku.supplier_id
        supplier = supplier_map.get(supplier_id)
        if not supplier:
            notes.append(f"No supplier for {sku_id}")
            continue

        # Demand per location
        for loc_id, loc in loc_map.items():
            key = (sku_id, loc_id)
            daily_qty = daily.get(key, 0.0)
            inv_rec = inv_lookup.get(key)
            on_hand = inv_rec.on_hand if inv_rec else 0
            inbound = inv_rec.inbound if inv_rec else 0
            target = int(round(daily_qty * (supplier.lead_time_days + cover_buffer_days))) + loc.safety_stock
            net = target - (on_hand + inbound)
            if net <= 0:
                continue
            # round up to case size
            case_qty = sku.case_size
            need = ((net + case_qty - 1) // case_qty) * case_qty
            supplier_allocations[supplier_id].append(
                Allocation(location_id=loc_id, sku=sku_id, qty=need)
            )

    # Collapse allocations into lines and respect MOQ
    for sup_id, allocs in supplier_allocations.items():
        agg: Dict[str, int] = defaultdict(int)
        for a in allocs:
            agg[a.sku] += a.qty
        supplier = supplier_map[sup_id]
        for sku_id, total_qty in agg.items():
            unit_cost = supplier.price_band.get(sku_id, 0.0)
            if total_qty < supplier.min_order_qty:
                total_qty = supplier.min_order_qty
            supplier_lines[sup_id].append(PlanLine(sku=sku_id, qty=total_qty, unit_cost=unit_cost))

    purchase_orders: List[PurchaseOrder] = []
    for sup_id, lines in supplier_lines.items():
        supplier = supplier_map[sup_id]
        allocations = supplier_allocations.get(sup_id, [])
        purchase_orders.append(
            PurchaseOrder(
                supplier_id=sup_id,
                status="Draft",
                eta_days=supplier.lead_time_days,
                lines=lines,
                allocations=allocations,
            )
        )

    return PlanResult(purchase_orders=purchase_orders, notes=notes)
