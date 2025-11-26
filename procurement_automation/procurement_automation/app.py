from __future__ import annotations

import io
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fpdf import FPDF
from . import data, loader, planner
from .models import PlanResult, PurchaseOrder, PlanLine, Allocation, Supplier, SKU, InventoryRecord, Location


app = FastAPI(title="Procurement Automation", version="0.2.0")
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


def _po_to_dict(po: PurchaseOrder) -> dict:
    return {
        "supplier_id": po.supplier_id,
        "status": po.status,
        "eta_days": po.eta_days,
        "lines": [vars(line) for line in po.lines],
        "allocations": [vars(a) for a in po.allocations],
    }


def _read_bytes(file: Optional[UploadFile]) -> Optional[bytes]:
    if file is None:
        return None
    return file.file.read()


# Simple in-memory state to reuse data across requests
current_data: Dict[str, List] = {
    "suppliers": data.sample_suppliers(),
    "skus": data.sample_skus(),
    "locations": data.sample_locations(),
    "inventory": data.sample_inventory(),
    "sales": data.sample_sales(),
}
current_plan: Optional[PlanResult] = None


def _set_state(
    suppliers: List[Supplier],
    skus: List[SKU],
    locations: List[Location],
    inventory: List[InventoryRecord],
    sales: List,
    plan: PlanResult,
):
    global current_data, current_plan
    current_data = {
        "suppliers": suppliers,
        "skus": skus,
        "locations": locations,
        "inventory": inventory,
        "sales": sales,
    }
    current_plan = plan


def _ensure_plan() -> PlanResult:
    global current_plan
    if current_plan is None:
        current_plan = planner.plan(
            suppliers=current_data["suppliers"],
            skus=current_data["skus"],
            locations=current_data["locations"],
            inventory=current_data["inventory"],
            sales=current_data["sales"],
        )
    return current_plan


def _inventory_map(inv: List[InventoryRecord]) -> Dict[tuple[str, str], InventoryRecord]:
    lookup: Dict[tuple[str, str], InventoryRecord] = {}
    for rec in inv:
        lookup[(rec.sku, rec.location_id)] = rec
    return lookup


def _supplier_summary(supplier_id: str):
    plan = _ensure_plan()
    suppliers = {s.supplier_id: s for s in current_data["suppliers"]}
    skus = [s for s in current_data["skus"] if s.supplier_id == supplier_id]
    inv_lookup = _inventory_map(current_data["inventory"])
    locations = {l.location_id: l for l in current_data["locations"]}

    po = None
    for p in plan.purchase_orders:
        if p.supplier_id == supplier_id:
            po = p
            break

    alloc_lookup: Dict[tuple[str, str], int] = defaultdict(int)
    alloc_by_loc: Dict[str, Dict] = {}
    if po:
        for alloc in po.allocations:
            alloc_lookup[(alloc.sku, alloc.location_id)] += alloc.qty
            loc = locations.get(alloc.location_id)
            entry = alloc_by_loc.setdefault(
                alloc.location_id,
                {
                    "location_id": alloc.location_id,
                    "kind": loc.kind if loc else "store",
                    "total_allocated": 0,
                    "skus": [],
                },
            )
            entry["total_allocated"] += alloc.qty
            entry["skus"].append({"sku": alloc.sku, "qty": alloc.qty})

    sku_blocks = []
    for sku in skus:
        locs = []
        planned_total = 0
        for loc_id, loc in locations.items():
            rec = inv_lookup.get((sku.sku, loc_id))
            planned_in = alloc_lookup.get((sku.sku, loc_id), 0)
            planned_total += planned_in
            locs.append(
                {
                    "location_id": loc_id,
                    "kind": loc.kind,
                    "capacity": loc.capacity,
                    "safety_stock": loc.safety_stock,
                    "on_hand": rec.on_hand if rec else 0,
                    "inbound": rec.inbound if rec else 0,
                    "planned_in": planned_in,
                    "low": (rec.on_hand if rec else 0) < loc.safety_stock,
                }
            )
        sku_blocks.append(
            {
                "sku": sku.sku,
                "case_size": sku.case_size,
                "locations": locs,
                "planned_total": planned_total,
            }
        )

    return {
        "supplier": suppliers.get(supplier_id),
        "skus": sku_blocks,
        "purchase_order": po,
        "allocation_summary": list(alloc_by_loc.values()),
    }


def _po_pdf_bytes(supplier: Supplier, po: PurchaseOrder, skus: List[SKU]) -> bytes:
    sku_map = {s.sku: s for s in skus}
    loc_kind_map = {l.location_id: l.kind for l in current_data["locations"]}
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Purchase Order - {supplier.name}", ln=1)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Supplier ID: {supplier.supplier_id}", ln=1)
    pdf.cell(0, 8, f"Lead time (days): {supplier.lead_time_days}", ln=1)
    pdf.cell(0, 8, f"Status: {po.status}", ln=1)
    pdf.cell(0, 8, f"ETA (days): {po.eta_days}", ln=1)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(60, 8, "SKU", border=1)
    pdf.cell(30, 8, "Case size", border=1)
    pdf.cell(30, 8, "Qty", border=1)
    pdf.cell(30, 8, "Unit cost", border=1)
    pdf.cell(40, 8, "Line total", border=1, ln=1)
    pdf.set_font("Helvetica", "", 12)
    grand = 0.0
    for line in po.lines:
        case = sku_map.get(line.sku).case_size if line.sku in sku_map else ""
        line_total = float(line.qty) * float(line.unit_cost)
        grand += line_total
        pdf.cell(60, 8, line.sku, border=1)
        pdf.cell(30, 8, str(case), border=1)
        pdf.cell(30, 8, str(line.qty), border=1)
        pdf.cell(30, 8, f"{line.unit_cost:.2f}", border=1)
        pdf.cell(40, 8, f"{line_total:.2f}", border=1, ln=1)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(150, 8, "Total", border=1)
    pdf.cell(40, 8, f"{grand:.2f}", border=1, ln=1)
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "Allocations to DCs/branches")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(50, 8, "Location", border=1)
    pdf.cell(30, 8, "Kind", border=1)
    pdf.cell(50, 8, "SKU", border=1)
    pdf.cell(40, 8, "Qty", border=1, ln=1)
    alloc_totals: Dict[str, int] = defaultdict(int)
    for alloc in po.allocations:
        alloc_totals[alloc.location_id] += alloc.qty
        pdf.cell(50, 8, alloc.location_id, border=1)
        pdf.cell(30, 8, loc_kind_map.get(alloc.location_id, ""), border=1)
        pdf.cell(50, 8, alloc.sku, border=1)
        pdf.cell(40, 8, str(alloc.qty), border=1, ln=1)
    if alloc_totals:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Allocation totals", ln=1)
        pdf.set_font("Helvetica", "", 10)
        for loc_id, qty in alloc_totals.items():
            pdf.cell(80, 8, loc_id, border=1)
            pdf.cell(40, 8, loc_kind_map.get(loc_id, ""), border=1)
            pdf.cell(40, 8, str(qty), border=1, ln=1)
    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return str(out).encode("latin1")


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
      --ok: #22c55e;
      --warn: #facc15;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Manrope', 'Segoe UI', sans-serif;
      background: radial-gradient(circle at 10% 20%, #132040 0, #0b1329 25%, #0f172a 60%);
      color: #e5e7eb;
    }
    .container { max-width: 1520px; margin: 32px auto 48px; padding: 0 28px 40px; }
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
    form { margin: 0; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 14px 36px rgba(0, 0, 0, 0.28);
    }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; margin-top: 18px; }
    @media (max-width: 1280px) { .two-col { grid-template-columns: 1fr; } }
    .wide-panel { margin-top: 16px; }
    .plan-actions { display:flex; gap:10px; align-items:center; justify-content: space-between; flex-wrap:wrap; }
    .status-list { list-style: none; padding: 0; margin: 10px 0 0; display: flex; flex-wrap: wrap; gap: 8px; }
    .status-item {
      padding: 8px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--card);
      color: #e2e8f0;
      font-size: 13px;
    }
    .status-item.ok { border-color: rgba(34, 197, 94, 0.4); color: var(--ok); }
    .status-item.pending { border-color: rgba(250, 204, 21, 0.4); color: var(--warn); }
    .status-item.info { border-color: rgba(148, 163, 184, 0.5); color: #cbd5e1; }
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
    .supplier-card { background: #0a0f1e; border: 1px solid var(--border); border-radius: 12px; padding: 12px; margin-top: 10px; }
    .tag { display: inline-block; padding: 4px 10px; border-radius: 999px; background: rgba(148,163,184,0.12); color: #cbd5e1; margin-right: 6px; font-size: 12px; }
    .sku-card { border: 1px solid var(--border); border-radius: 10px; padding: 10px; margin-top: 8px; background: #0d1426; }
    .sku-locs { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 6px; margin-top: 6px; }
    .loc-chip { background: rgba(148,163,184,0.08); border: 1px solid var(--border); border-radius: 8px; padding: 6px 8px; color: #cbd5e1; font-size: 12px; }
    .pill { background: rgba(249, 115, 22, 0.18); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); padding: 4px 8px; border-radius: 10px; font-size: 12px; }
    .warn { color: #fbbf24; }
    .po-line { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
    .po-line button { padding: 6px 10px; font-size: 13px; }
    .po-line input { width: 90px; padding: 6px 8px; border-radius: 8px; border: 1px solid var(--border); background: #0a0f1e; color: #e2e8f0; }
    .note { color: #fbbf24; margin-top: 6px; font-size: 13px; }
    .footer { color: var(--muted); font-size: 12px; margin-top: 10px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <p class="eyebrow">Supplier-ready plan</p>
      <h1>Procurement Automation</h1>
      <p class="subhead">Upload JSON/CSV files or run with the bundled multi-supplier sample set to generate a deterministic procurement plan with supplier-level POs, branch allocations, and ready-to-send PDFs.</p>
    </div>

    <div class="two-col">
      <div class="panel">
        <div class="actions">
          <strong>Supplier view</strong>
          <select id="supplier-select" style="padding:8px 10px;border-radius:10px;border:1px solid var(--border);background:var(--card);color:#e2e8f0;">
            <option value="">Loading suppliers...</option>
          </select>
          <button id="regen-btn" type="button" onclick="regeneratePO()">Generate PO</button>
          <button id="pdf-btn" type="button" onclick="downloadPdf()">Download PO PDF</button>
          <span class="hint">Auto-refreshes from the latest data/plan and shows branch stock per SKU.</span>
        </div>
        <div id="supplier-info" class="supplier-card">Select a supplier to view linked products, branch stock, and PO lines.</div>
      </div>

      <div>
        <form id="form" enctype="multipart/form-data" class="panel">
          <div class="actions">
            <span class="badge">Uploads optional — defaults provided</span>
            <button id="run-btn" type="button" onclick="regeneratePO()">Generate PO</button>
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

        <div class="panel" style="margin-top:12px;">
          <div id="linked-skus" class="supplier-card">Select a supplier to view linked SKUs and stock.</div>
        </div>
      </div>
    </div>

    <div class="panel" style="margin-top:16px; max-width: 880px;">
      <div class="plan-actions">
        <div>
          <strong>Plan output</strong>
          <span class="hint">Latest JSON emitted by the planner. Auto-updates after Generate PO.</span>
        </div>
        <div class="note" id="note">Using bundled sample data by default. Uploads are optional.</div>
      </div>
      <div id="result" class="result" style="max-height: 280px;">Generating plan...</div>
      <div class="footer">Output is also written to output/procurement_plan.json for convenience.</div>
    </div>
  </div>

  <script>
    function pushStatus(text, state = 'info') {
      const list = document.getElementById('status-list');
      if (!list) return;
      const li = document.createElement('li');
      li.className = `status-item ${state}`;
      li.textContent = text;
      list.appendChild(li);
    }

    function renderSupplierSummary(summary) {
      const panel = document.getElementById('supplier-info');
      if (!summary || summary.error) {
        panel.textContent = 'Supplier not found.';
        return;
      }
      const supplier = summary.supplier;
      const po = summary.purchase_order;
      const skus = summary.skus || [];
      const allocations = summary.allocation_summary || [];
      window.currentSupplier = supplier;
      window.currentPoLines = po ? (po.lines || []) : [];

      const stats = [
        `<span class="pill">Lead time ${supplier.lead_time_days}d</span>`,
        `<span class="pill">MOQ ${supplier.min_order_qty}</span>`
      ];
      if (po) {
        stats.push(`<span class="pill">Lines ${po.lines.length}</span>`);
        stats.push(`<span class="pill">ETA ${po.eta_days}d</span>`);
      }

      const poLines = po && po.lines.length
        ? po.lines.map(l => `<div class="loc-chip">${l.sku}: qty ${l.qty} @ ${l.unit_cost}</div>`).join('')
        : '<span class="hint">No draft PO lines for this supplier.</span>';

      const allocHtml = allocations.length
        ? allocations.map(a => `<div class="loc-chip">${a.location_id} (${a.kind}) → ${a.total_allocated} units</div>`).join('')
        : '<span class="hint">No allocations planned for this supplier.</span>';

      panel.innerHTML = `
        <div class="tag">Supplier</div>
        <div><strong>${supplier.name}</strong> (${supplier.supplier_id})</div>
        <div style="margin:6px 0; display:flex; gap:6px; flex-wrap:wrap;">${stats.join('')}</div>
        <div style="margin:6px 0;"><strong>Allocations (DC/branches)</strong></div>
        <div class="sku-locs">${allocHtml}</div>
        <div style="margin:6px 0;"><strong>Draft PO lines</strong></div>
        <div class="sku-locs">${poLines}</div>
        <div style="margin:10px 0 6px;"><strong>Edit PO quantities</strong></div>
        <div id="po-editor">
          ${
            po && po.lines.length
              ? po.lines
                  .map(
                    (l) => `
                    <div class="sku-card">
                      <div><strong>${l.sku}</strong> <span class="tag">unit ${l.unit_cost}</span></div>
                      <div class="po-line">
                        <button type="button" class="po-btn" data-action="minus" data-sku="${l.sku}">-</button>
                        <input class="po-line-input" data-sku="${l.sku}" type="number" min="0" value="${l.qty}">
                        <button type="button" class="po-btn" data-action="plus" data-sku="${l.sku}">+</button>
                      </div>
                    </div>
                  `
                  )
                  .join('')
              : '<span class="hint">No PO lines to edit.</span>'
          }
        </div>
      `;

      const linked = document.getElementById('linked-skus');
      if (linked) {
        const skuCards = skus.map((sku) => {
          const totalOnHand = sku.locations.reduce((acc, l) => acc + (l.on_hand || 0), 0);
          const totalInbound = sku.locations.reduce((acc, l) => acc + (l.inbound || 0), 0);
          const locs = sku.locations.map(l => {
            const badge = l.low ? '<span class="warn">• low</span>' : '';
            const planned = l.planned_in && l.planned_in > 0 ? ` | planned in ${l.planned_in}` : '';
            const cap = l.capacity ? ` | cap ${l.capacity}` : '';
            return `<div class="loc-chip">${l.location_id} (${l.kind}) — on hand ${l.on_hand}, inbound ${l.inbound}${planned}${cap} ${badge}</div>`;
          }).join('');
          return `
            <div class="sku-card">
              <div><strong>${sku.sku}</strong> <span class="tag">case ${sku.case_size}</span> <span class="tag">on hand ${totalOnHand} | inbound ${totalInbound}</span> <span class="tag">planned ${sku.planned_total}</span></div>
              <div class="sku-locs">${locs}</div>
            </div>
          `;
        }).join('');
        linked.innerHTML = skuCards || '<span class="hint">No SKUs for this supplier.</span>';
      }
    }

    async function loadSuppliers(preferId = null) {
      const sel = document.getElementById('supplier-select');
      sel.innerHTML = '<option value=\"\">Loading suppliers...</option>';
      try {
        const res = await fetch('/api/state');
        if (!res.ok) throw new Error('Failed to load suppliers');
        const json = await res.json();
        sel.innerHTML = '';
        json.suppliers.forEach((s, idx) => {
          const opt = document.createElement('option');
          opt.value = s.supplier_id;
          opt.textContent = `${s.name} (${s.supplier_id})`;
          sel.appendChild(opt);
        });
        if (json.suppliers.length > 0) {
          const match = preferId && json.suppliers.find(s => s.supplier_id === preferId);
          sel.value = match ? preferId : json.suppliers[0].supplier_id;
          loadSupplierSummary(sel.value);
          pushStatus(`Loaded ${json.suppliers.length} suppliers`, 'ok');
        }
      } catch (e) {
        sel.innerHTML = '<option value=\"\">Failed to load suppliers</option>';
        pushStatus('Failed to load suppliers', 'pending');
      }
    }

    async function loadSupplierSummary(id) {
      const panel = document.getElementById('supplier-info');
      if (!id) {
        panel.textContent = 'Select a supplier to view linked products and stock.';
        return;
      }
      panel.textContent = 'Loading supplier details...';
      try {
        const res = await fetch(`/api/supplier/${encodeURIComponent(id)}/summary`);
        if (!res.ok) throw new Error('Failed to load summary');
        const json = await res.json();
        renderSupplierSummary(json);
        pushStatus(`Viewing ${json.supplier.name}`, 'ok');
      } catch (e) {
        panel.textContent = 'Failed to load supplier details.';
      }
    }

    async function runPlan(preferId = null) {
      const form = document.getElementById('form');
      const data = new FormData(form);
      const note = document.getElementById('note');
      const result = document.getElementById('result');
      const btn = document.getElementById('run-btn');
      const regen = document.getElementById('regen-btn');
      note.textContent = 'Processing...';
      pushStatus('Planning purchase orders…', 'pending');
      btn.disabled = true;
      if (regen) regen.disabled = true;
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
        loadSuppliers(preferId);
        pushStatus(`Plan ready with ${json.purchase_orders.length} supplier POs`, 'ok');
      } catch (e) {
        note.textContent = 'Failed: ' + e.message;
        result.textContent = 'Submit files or run with defaults to view the plan.';
        pushStatus('Plan failed', 'pending');
      } finally {
        btn.disabled = false;
        if (regen) regen.disabled = false;
      }
    }

    function downloadPdf() {
      const sel = document.getElementById('supplier-select');
      if (!sel.value) return;
      const lines = getEditedLines();
      fetch(`/api/supplier/${encodeURIComponent(sel.value)}/po.pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lines }),
      })
        .then(res => res.blob())
        .then(blob => {
          const url = window.URL.createObjectURL(blob);
          window.open(url, '_blank');
        });
    }

    function getEditedLines() {
      const inputs = Array.from(document.querySelectorAll('.po-line-input'));
      return inputs.map(input => {
        const sku = input.dataset.sku;
        const qty = parseInt(input.value || '0', 10);
        const base = (window.currentPoLines || []).find(l => l.sku === sku);
        return {
          sku,
          qty: isNaN(qty) ? 0 : qty,
          unit_cost: base ? base.unit_cost : 0,
        };
      });
    }

    document.addEventListener('change', (ev) => {
      if (ev.target && ev.target.id === 'supplier-select') {
        loadSupplierSummary(ev.target.value);
      }
      if (ev.target && ev.target.classList.contains('po-line-input')) {
        // values are read on download
      }
    });

    document.addEventListener('click', (ev) => {
      if (ev.target && ev.target.classList.contains('po-btn')) {
        const sku = ev.target.dataset.sku;
        const input = document.querySelector(`.po-line-input[data-sku="${sku}"]`);
        if (!input) return;
        const current = parseInt(input.value || '0', 10) || 0;
        if (ev.target.dataset.action === 'plus') input.value = current + 1;
        if (ev.target.dataset.action === 'minus') input.value = Math.max(0, current - 1);
      }
    });

    function regeneratePO() {
      const sel = document.getElementById('supplier-select');
      const current = sel ? sel.value : null;
      runPlan(current);
    }

    window.addEventListener('DOMContentLoaded', () => {
      pushStatus('Running sample plan…', 'pending');
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
    _set_state(
        suppliers=sup,
        skus=sku_list,
        locations=locs,
        inventory=inv,
        sales=sales_data,
        plan=plan,
    )
    plan_dict = plan_to_dict(plan)

    # Also drop a file to output/ for reference
    out_path = Path(__file__).resolve().parent.parent / "output"
    out_path.mkdir(exist_ok=True, parents=True)
    (out_path / "procurement_plan.json").write_text(json.dumps(plan_dict, indent=2))

    return JSONResponse(plan_dict)


@app.get("/api/state")
async def api_state():
    plan = _ensure_plan()
    suppliers = [
        {
            "supplier_id": s.supplier_id,
            "name": s.name,
            "lead_time_days": s.lead_time_days,
            "min_order_qty": s.min_order_qty,
        }
        for s in current_data["suppliers"]
    ]
    return {"suppliers": suppliers, "purchase_orders": plan_to_dict(plan)["purchase_orders"]}


@app.get("/api/supplier/{supplier_id}/summary")
async def supplier_summary(supplier_id: str):
    summary = _supplier_summary(supplier_id)
    if not summary["supplier"]:
        return JSONResponse({"error": "Supplier not found"}, status_code=404)
    sup = summary["supplier"]
    po = summary["purchase_order"]
    return {
        "supplier": {
            "supplier_id": sup.supplier_id,
            "name": sup.name,
            "lead_time_days": sup.lead_time_days,
            "min_order_qty": sup.min_order_qty,
        },
        "purchase_order": _po_to_dict(po) if po else None,
        "skus": summary["skus"],
        "allocation_summary": summary.get("allocation_summary", []),
    }


@app.get("/api/supplier/{supplier_id}/po.pdf")
async def supplier_po_pdf(supplier_id: str):
    summary = _supplier_summary(supplier_id)
    supplier = summary["supplier"]
    po = summary["purchase_order"]
    if not supplier or not po:
        return JSONResponse({"error": "No purchase order for this supplier"}, status_code=404)
    pdf_bytes = _po_pdf_bytes(
        supplier=supplier,
        po=po,
        skus=[s for s in current_data["skus"] if s.supplier_id == supplier_id],
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="po_{supplier_id}.pdf"'},
    )


@app.post("/api/supplier/{supplier_id}/po.pdf")
async def supplier_po_pdf_custom(supplier_id: str, request: Request):
    summary = _supplier_summary(supplier_id)
    supplier = summary["supplier"]
    base_po: PurchaseOrder | None = summary["purchase_order"]
    if not supplier or not base_po:
        return JSONResponse({"error": "No purchase order for this supplier"}, status_code=404)
    body = await request.json()
    lines_payload = body.get("lines", [])
    lines: List[PlanLine] = []
    for item in lines_payload:
        try:
            lines.append(
                PlanLine(
                    sku=item["sku"],
                    qty=int(item.get("qty", 0)),
                    unit_cost=float(item.get("unit_cost", 0)),
                )
            )
        except Exception:
            continue
    custom_po = PurchaseOrder(
        supplier_id=base_po.supplier_id,
        status=base_po.status,
        eta_days=base_po.eta_days,
        lines=lines if lines else base_po.lines,
        allocations=base_po.allocations,
    )
    pdf_bytes = _po_pdf_bytes(
        supplier=supplier,
        po=custom_po,
        skus=[s for s in current_data["skus"] if s.supplier_id == supplier_id],
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="po_${supplier_id}_custom.pdf"'},
    )
