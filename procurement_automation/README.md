# Procurement Automation (Sample Playbook)

A self-contained, data-driven prototype that simulates an end-to-end procurement loop:
- Multi-location, omni-channel inventory snapshot (stores, DC/warehouse, online).
- Supplier metadata (lead time, MOQs, price bands).
- Demand signal from recent sales history.
- Reorder proposal: target cover = lead time + buffer; distribute to locations.
- Purchase orders (draft) with planned GRN and downstream allocation plan.
- File upload support (JSON/CSV) to override the sample data.

## What it does
- Loads sample state (suppliers, SKUs, locations, stock, inbound POs, sales history) or accepts uploaded files.
- Computes net need per SKU → target stock per location → aggregated supplier POs.
- Emits a single plan JSON with POs, GRN schedule, and per-location allocation.
- Simple FastAPI UI to upload files and view/download the plan.

## Run it (CLI)
```bash
cd procurement_automation
python -m procurement_automation.run
# plan written to procurement_automation/output/procurement_plan.json
```

## Run the API/UI
```bash
cd procurement_automation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn procurement_automation.app:app --reload --port 8010
```
Open http://localhost:8010 to upload JSON/CSV files or use the sample data.

## Data model (sample)
- Suppliers: lead time days, min order qty, price tiers.
- SKUs: supplier mapping, case size.
- Locations: type (store/DC/online), capacity, safety stock.
- Inventory: on-hand + inbound.
- Sales history: recent daily units (per SKU, per location).

## Output (excerpt)
```json
{
  "plan_generated_at": "...",
  "purchase_orders": [
    {
      "supplier_id": "SUP-RED",
      "status": "Draft",
      "eta_days": 5,
      "lines": [
        { "sku": "SKU-RED-TEA", "qty": 420, "unit_cost": 1.1 }
      ],
      "allocations": [
        { "location_id": "DC-EAST", "sku": "SKU-RED-TEA", "qty": 220 },
        { "location_id": "STORE-NY", "sku": "SKU-RED-TEA", "qty": 140 },
        { "location_id": "ONLINE", "sku": "SKU-RED-TEA", "qty": 60 }
      ]
    }
  ]
}
```

## Notes
- All logic is heuristic and deterministic (no LLMs here).
- Tweak `data.py` to change assumptions (safety stock, cover days, prices, lead times).
- Extend `planner.py` if you want negotiation/confirmation states or to plug into a real ERP.
