from __future__ import annotations

import json
from pathlib import Path
from . import data, planner


def main():
    plan = planner.plan(
        suppliers=data.sample_suppliers(),
        skus=data.sample_skus(),
        locations=data.sample_locations(),
        inventory=data.sample_inventory(),
        sales=data.sample_sales(),
    )
    plan_dict = {
        "purchase_orders": [
            {
                "supplier_id": po.supplier_id,
                "status": po.status,
                "eta_days": po.eta_days,
                "lines": [vars(line) for line in po.lines],
                "allocations": [vars(a) for a in po.allocations],
            }
            for po in plan.purchase_orders
        ],
        "notes": plan.notes,
    }
    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "procurement_plan.json"
    out_file.write_text(json.dumps(plan_dict, indent=2))
    print(f"plan written to {out_file}")


if __name__ == "__main__":
    main()
