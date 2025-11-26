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
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f172a;
      --card: #0b1224;
      --panel: #0f172a;
      --border: #1f2937;
      --accent: #f97316;
      --muted: #94a3b8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Manrope', 'Segoe UI', sans-serif;
      background: radial-gradient(circle at 10% 20%, #132040 0, #0b1329 25%, #0f172a 60%);
      color: #e5e7eb;
    }
    .container { max-width: 1120px; margin: 48px auto; padding: 0 20px 32px; }
    .hero {
      background: linear-gradient(120deg, rgba(249, 115, 22, 0.12), rgba(59, 130, 246, 0.12));
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 16px 48px rgba(0, 0, 0, 0.35);
    }
    .eyebrow { text-transform: uppercase; letter-spacing: 0.08em; font-size: 12px; color: var(--muted); margin: 0 0 6px; }
    h1 { margin: 0 0 8px; font-size: 28px; }
    .subhead { margin: 0; color: #cbd5e1; font-size: 15px; line-height: 1.6; }
    form { margin-top: 22px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 14px 36px rgba(0, 0, 0, 0.28);
    }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 12px 0 4px; }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      min-height: 110px;
    }
    .card strong { display: block; margin-bottom: 8px; color: #e2e8f0; }
    .hint { color: var(--muted); font-size: 12px; margin-top: 2px; }
    input[type="file"] { color: #cbd5e1; font-size: 13px; width: 100%; }
    .actions { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-top: 10px; }
    button {
      background: linear-gradient(120deg, #f97316, #fbbf24);
      color: #0f172a;
      border: none;
      padding: 11px 18px;
      border-radius: 10px;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 10px 28px rgba(249, 115, 22, 0.35);
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.65; cursor: not-allowed; transform: none; }
    .badge { background: rgba(249, 115, 22, 0.16); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); padding: 5px 10px; border-radius: 999px; font-size: 12px; }
    h3 { margin: 18px 0 8px; }
    .result {
      white-space: pre;
      font-family: ui-monospace, SFMono-Regular, Consolas, Menlo, monospace;
      background: #0a0f1e;
      padding: 14px;
      border-radius: 12px;
      border: 1px solid var(--border);
      overflow: auto;
      max-height: 460px;
    }
    .note { color: #fbbf24; margin-top: 6px; font-size: 13px; }
    .footer { color: var(--muted); font-size: 12px; margin-top: 10px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <p class="eyebrow">Supplier-ready plan</p>
      <h1>Procurement Automation</h1>
      <p class="subhead">Upload JSON/CSV files or run with the bundled sample set to generate a deterministic procurement plan with supplier-level POs and allocations.</p>
    </div>

    <form id="form" enctype="multipart/form-data" class="panel">
      <div class="actions">
        <span class="badge">Uploads optional — defaults provided</span>
        <button id="run-btn" type="button" onclick="runPlan()">Generate plan</button>
        <span class="hint">We never send data outside this service.</span>
      </div>
      <div class="grid">
        <div class="card">
          <strong>Suppliers</strong>
          <span class="hint">JSON: supplier_id, lead_time_days, min_order_qty, price_band</span>
          <input type="file" name="suppliers">
        </div>
        <div class="card">
          <strong>SKUs</strong>
          <span class="hint">JSON: sku, supplier_id, case_size</span>
          <input type="file" name="skus">
        </div>
        <div class="card">
          <strong>Locations</strong>
          <span class="hint">JSON: location_id, kind, capacity, safety_stock</span>
          <input type="file" name="locations">
        </div>
        <div class="card">
          <strong>Inventory</strong>
          <span class="hint">JSON or CSV: sku, location_id, on_hand, inbound</span>
          <input type="file" name="inventory">
        </div>
        <div class="card">
          <strong>Sales</strong>
          <span class="hint">JSON or CSV: sku, location_id, qty, days</span>
          <input type="file" name="sales">
        </div>
      </div>
    </form>

    <div class="note" id="note">Using bundled sample data by default. Uploads are optional.</div>
    <h3>Plan output</h3>
    <div id="result" class="result">Generating plan...</div>
    <div class="footer">Output is also written to output/procurement_plan.json for convenience.</div>
  </div>

  <script>
    async function runPlan() {
      const form = document.getElementById('form');
      const data = new FormData(form);
      const note = document.getElementById('note');
      const result = document.getElementById('result');
      const btn = document.getElementById('run-btn');
      note.textContent = 'Processing...';
      btn.disabled = true;
      try {
        const res = await fetch('/api/run', { method: 'POST', body: data });
        if (!res.ok) throw new Error('Request failed');
        const json = await res.json();
        result.textContent = JSON.stringify(json, null, 2);
        if (Array.isArray(json.notes) && json.notes.length > 0) {
          note.textContent = 'Notes: ' + json.notes.join(' | ');
        } else {
          note.textContent = 'Plan generated successfully.';
        }
      } catch (e) {
        note.textContent = 'Failed: ' + e.message;
        result.textContent = 'Submit files or run with defaults to view the plan.';
      } finally {
        btn.disabled = false;
      }
    }

    window.addEventListener('DOMContentLoaded', () => {
      runPlan();
    });
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
