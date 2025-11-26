# Resource Procurement Automation

A concise, deterministic prototype that generates supplier-ready purchase orders from demand and inventory signals. It ships with sample data, accepts JSON/CSV uploads, and produces a single procurement plan that can be consumed by downstream systems.

## Highlights
- End-to-end loop: suppliers, SKUs, locations, inventory, sales → purchase orders and allocations.
- Deterministic heuristics: lead time + buffer cover, safety stock, case-size rounding, supplier MOQs.
- Dual interface: CLI for quick runs, FastAPI UI for uploads and JSON download.
- Simple Python data models that are easy to extend for a real ERP/OMS integration.

## Quick Start (CLI)
```bash
cd procurement_automation
python -m procurement_automation.run
# Writes output/procurement_plan.json
```

## Run the API + UI
```bash
cd procurement_automation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn procurement_automation.app:app --reload --port 8010
```
Open http://localhost:8010 to upload optional JSON/CSV files or just run with the included sample data.

## Input expectations
- Suppliers: `supplier_id`, `name`, `lead_time_days`, `min_order_qty`, `price_band` (sku → unit price).
- SKUs: `sku`, `supplier_id`, `case_size`.
- Locations: `location_id`, `kind` (`store|dc|online`), `capacity`, `safety_stock`.
- Inventory: `sku`, `location_id`, `on_hand`, `inbound`.
- Sales: `sku`, `location_id`, `qty`, `days` (qty sold over this many days).

Upload JSON arrays or CSVs (for inventory and sales) that match these fields. See `procurement_automation/data.py` for the bundled sample payloads.

## Output shape (excerpt)
```json
{
  "purchase_orders": [
    {
      "supplier_id": "SUP-RED",
      "status": "Draft",
      "eta_days": 5,
      "lines": [
        { "sku": "SKU-RED-TEA", "qty": 420, "unit_cost": 1.1 }
      ],
      "allocations": [
        { "location_id": "DC-EAST", "sku": "SKU-RED-TEA", "qty": 220 }
      ]
    }
  ],
  "notes": []
}
```

## How it plans
1. Compute daily demand per SKU/location from sales history.
2. Derive target stock = (lead time + buffer days) × daily demand + safety stock.
3. Net against on-hand + inbound; round up to case size.
4. Aggregate by supplier, enforcing MOQ and unit costs from the supplier price band.

Tune safety stock, cover buffer, pricing, or MOQ rules in `procurement_automation/data.py` and `procurement_automation/planner.py`.
