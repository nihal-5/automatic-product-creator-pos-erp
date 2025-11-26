from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from . import data, loader, planner
from .models import PlanResult, PurchaseOrder, PlanLine, Allocation


app = FastAPI(title="Procurement Automation", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def plan_to_dict(plan: PlanResult) -> dict:
    return {
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


def _read_bytes(file: Optional[UploadFile]) -> Optional[bytes]:
    if file is None:
        return None
    return file.file.read()


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Procurement Automation</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 1100px; margin: 32px auto; color: #1f2937; }
    h1 { margin-bottom: 6px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin: 12px 0; }
    .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; background: #fff; }
    button { background: #2563eb; color: #fff; border: none; padding: 10px 16px; border-radius: 8px; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    .result { white-space: pre; font-family: ui-monospace, SFMono-Regular, Consolas, Menlo, monospace; background: #f9fafb; padding: 12px; border-radius: 10px; overflow: auto; max-height: 420px; }
    .note { color: #dc2626; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>Procurement Automation</h1>
  <p>Upload data (JSON/CSV) to generate a purchase plan. Leave empty to use sample data.</p>
  <form id="form" enctype="multipart/form-data">
    <div class="grid">
      <div class="card">
        <strong>Suppliers (JSON)</strong><br>
        <input type="file" name="suppliers">
      </div>
      <div class="card">
        <strong>SKUs (JSON)</strong><br>
        <input type="file" name="skus">
      </div>
      <div class="card">
        <strong>Locations (JSON)</strong><br>
        <input type="file" name="locations">
      </div>
      <div class="card">
        <strong>Inventory (JSON or CSV)</strong><br>
        <input type="file" name="inventory">
      </div>
      <div class="card">
        <strong>Sales (JSON or CSV)</strong><br>
        <input type="file" name="sales">
      </div>
    </div>
    <button type="button" onclick="runPlan()">Generate Plan</button>
  </form>
  <div class="note" id="note"></div>
  <h3>Plan</h3>
  <div id="result" class="result">Awaiting input…</div>

  <script>
    async function runPlan() {
      const form = document.getElementById('form');
      const data = new FormData(form);
      const note = document.getElementById('note');
      const result = document.getElementById('result');
      note.textContent = '';
      result.textContent = 'Processing...';
      try {
        const res = await fetch('/api/run', { method: 'POST', body: data });
        if (!res.ok) throw new Error('Request failed');
        const json = await res.json();
        result.textContent = JSON.stringify(json, null, 2);
      } catch (e) {
        note.textContent = 'Failed: ' + e.message;
        result.textContent = 'Awaiting input…';
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/api/run")
async def run_plan(
    suppliers: UploadFile | None = File(default=None),
    skus: UploadFile | None = File(default=None),
    locations: UploadFile | None = File(default=None),
    inventory: UploadFile | None = File(default=None),
    sales: UploadFile | None = File(default=None),
):
    # Load user-provided or sample data
    sup = loader.load_suppliers(_read_bytes(suppliers)) or data.sample_suppliers()
    sku_list = loader.load_skus(_read_bytes(skus)) or data.sample_skus()
    locs = loader.load_locations(_read_bytes(locations)) or data.sample_locations()
    inv = loader.load_inventory(_read_bytes(inventory)) or data.sample_inventory()
    sales_data = loader.load_sales(_read_bytes(sales)) or data.sample_sales()

    plan = planner.plan(
        suppliers=sup,
        skus=sku_list,
        locations=locs,
        inventory=inv,
        sales=sales_data,
    )
    plan_dict = plan_to_dict(plan)

    # Also drop a file to output/ for reference
    out_path = Path(__file__).resolve().parent.parent / "output"
    out_path.mkdir(exist_ok=True, parents=True)
    (out_path / "procurement_plan.json").write_text(json.dumps(plan_dict, indent=2))

    return JSONResponse(plan_dict)
