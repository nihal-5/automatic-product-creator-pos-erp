# Resource Procurement Automation

A compact, deterministic engine that turns sales and inventory signals into supplier-ready purchase orders. It ships with sample data, accepts JSON/CSV uploads, and emits a single plan that downstream systems can consume without extra wiring.

## What you get
- End-to-end loop: suppliers, SKUs, locations, inventory, sales → purchase orders and allocations.
- Deterministic heuristics: lead-time cover + buffer, safety stock, case-size rounding, supplier MOQs.
- Two ways to run: CLI for quick checks, FastAPI UI for uploads and JSON download.
- Simple Python models that can plug into an ERP/OMS or be tuned for your rules.

## Run it quickly (CLI)
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
Open http://localhost:8010 to upload optional JSON/CSV files or just run with the bundled sample data.

### API endpoints
- `GET /` — minimal UI for uploads and running the plan.
- `POST /api/run` — accepts multipart form-data (files optional); returns the procurement plan JSON.

Example:
```bash
curl -X POST http://localhost:8010/api/run \
  -F "suppliers=@suppliers.json" \
  -F "skus=@skus.json" \
  -F "locations=@locations.json" \
  -F "inventory=@inventory.csv" \
  -F "sales=@sales.csv"
```

## Data contracts
- Suppliers: `supplier_id`, `name`, `lead_time_days`, `min_order_qty`, `price_band` (sku -> unit price).
- SKUs: `sku`, `supplier_id`, `case_size`.
- Locations: `location_id`, `kind` (`store|dc|online`), `capacity`, `safety_stock`.
- Inventory (JSON or CSV): `sku`, `location_id`, `on_hand`, `inbound`.
- Sales (JSON or CSV): `sku`, `location_id`, `qty`, `days` (qty sold over this many days).

Upload JSON arrays or CSVs that match these fields. See `procurement_automation/data.py` for the included sample payloads.

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

## How the plan is built
1. Compute daily demand per SKU/location from sales history.
2. Target stock = (lead time + buffer days) * daily demand + safety stock.
3. Net against on-hand + inbound; round up to the SKU case size.
4. Aggregate by supplier, enforce MOQ, and apply unit costs from each supplier price band.

## Project layout
- `procurement_automation/app.py` — FastAPI app + HTML UI.
- `procurement_automation/run.py` — CLI entry point.
- `procurement_automation/planner.py` — planning logic and heuristics.
- `procurement_automation/data.py` — sample data and tweakable assumptions.
- `procurement_automation/loader.py` — JSON/CSV ingestion.
- `procurement_automation/samples/` — ready-to-upload JSON/CSV files that mirror the sample data.

Tune safety stock, cover buffer, pricing, or MOQ rules in `procurement_automation/data.py` and `procurement_automation/planner.py`.
