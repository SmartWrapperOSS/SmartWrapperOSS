# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
output/dashboard.py

Generates a single, self-contained HTML file that renders a run's results
as an interactive table — expandable rows, sortable columns — with no
server, no database, and no external files. The data is embedded directly
in the HTML as a JS array, so the file works by just double-clicking it
open in a browser, even offline.

DESIGN NOTE FOR FUTURE CONTRIBUTORS (e.g. adding delete support):
The embedded RESULTS array (see _build_html below) is the single source
of truth the page renders from. Any new interactive feature — deleting a
row, filtering, re-sorting — should operate on that in-memory array and
call render() again, rather than manipulating the DOM directly in
multiple places. This keeps the page's behavior predictable as more
features are added. A delete button, when added, should remove an entry
from RESULTS and call render() — see the renderTable() function and the
comment marking where a delete button would attach.

IMPORTANT: deleting a row in the dashboard only affects what's shown in
THIS already-generated HTML file, in your browser's memory — it does not
modify the original results.json on disk, and the change is lost if you
reload the page. There is no database here by design — see ENHANCEMENTS.md
if persistent, shared run history is ever wanted; that's intentionally
out of scope for this file.
"""

import json
from typing import List
from core.types import EvalResult


def save_html(results: List[EvalResult], path: str, workflow_name: str = ""):
    """Write a self-contained HTML dashboard for `results` to `path`."""
    data = _results_to_json_safe(results)
    html = _build_html(data, workflow_name)

    with open(path, "w") as f:
        f.write(html)

    print(f"Dashboard saved to {path} (open it directly in a browser)")


def _results_to_json_safe(results: List[EvalResult]) -> list:
    """Convert EvalResult objects into plain dicts the HTML's JS can embed."""
    data = []
    for i, r in enumerate(results):
        data.append({
            "id": f"{r.framework}__{r.model_id}__{i}",  # stable per-row id, used by expand/delete
            "framework": r.framework,
            "model_id": r.model_id,
            "composite_score": r.composite_score,
            "dimensions": r.score_result.dimensions,
            "output": r.task_result.output,
            "latency_ms": r.task_result.latency_ms,
            "cost_usd": r.task_result.cost_usd,
            "extra": r.task_result.extra,  # e.g. tool_calls, for the tool-use workflow
        })
    return data


def _build_html(data: list, workflow_name: str) -> str:
    # json.dumps with indent=2 keeps the embedded data human-readable if
    # someone opens this file in a text editor, not just a browser.
    #
    # SAFETY: if any model output contains the literal string "</script>",
    # naively embedding it would prematurely close our <script> tag in the
    # browser's HTML parser (this happens BEFORE any JS ever runs, so
    # escapeHtml() in JS can't help here — it's an HTML-parsing issue, not
    # a JS one). Escaping "</" as "<\\/" inside the JSON neutralizes this
    # without changing the parsed JSON value at all (JSON treats \/ and /
    # as identical) — see https://mathiasbynens.be/notes/etago for why
    # this exact escape is the standard fix for this class of bug.
    embedded_json = json.dumps(data, indent=2).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartWrapperOSS — Results{f' ({workflow_name})' if workflow_name else ''}</title>
<style>
{_CSS}
</style>
</head>
<body>
  <div class="page">
    <header class="page-header">
      <div class="brand">
        <span class="brand-mark">&gt;_</span>
        <span class="brand-name">SmartWrapperOSS</span>
      </div>
      <div class="meta">
        <span class="meta-label">workflow</span>
        <span class="meta-value">{workflow_name or 'unknown'}</span>
        <span class="meta-sep">·</span>
        <span class="meta-label">runs</span>
        <span class="meta-value" id="run-count">{len(data)}</span>
      </div>
    </header>

    <main>
      <table id="results-table">
        <thead id="table-head"></thead>
        <tbody id="table-body"></tbody>
      </table>
      <p class="empty-state" id="empty-state" style="display:none;">
        No results to show. Every row was removed from this view.
      </p>
    </main>

    <footer class="page-footer">
      <p class="footer-note">
        <strong>About Cost Score and Latency Score:</strong> these columns
        are not a fixed scale — they rank the rows in this table against
        each other. The cheapest/fastest run shown here gets a score of
        100; the most expensive/slowest run shown here gets a 0. A score
        of 100 means "cheapest among the rows in this comparison," not
        "free" — adding or removing a model from the comparison can change
        these scores even though the underlying cost or latency didn't
        change. Click a row to see the actual dollar cost and latency in
        seconds.
      </p>
      <p>
        Generated by SmartWrapperOSS. This is a static snapshot of one run —
        it does not update live and does not write back to results.json.
      </p>
    </footer>
  </div>

<script>
{_JS.replace("__EMBEDDED_RESULTS__", embedded_json)}
</script>
</body>
</html>
"""


_CSS = """
  :root {
    --bg: #0d1117;
    --bg-raised: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --accent: #7ee787;
    --accent-dim: #2ea04326;
    --mono: ui-monospace, "SF Mono", "Cascadia Code", "Fira Code", Consolas, monospace;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 14px;
    line-height: 1.5;
  }

  .page {
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 24px 64px;
  }

  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
    gap: 12px;
  }

  .brand { display: flex; align-items: baseline; gap: 8px; }
  .brand-mark { color: var(--accent); font-weight: 700; }
  .brand-name { font-weight: 600; letter-spacing: 0.02em; }

  .meta { color: var(--text-dim); font-size: 13px; }
  .meta-label { color: var(--text-dim); }
  .meta-value { color: var(--text); }
  .meta-sep { margin: 0 8px; opacity: 0.4; }

  table { width: 100%; border-collapse: collapse; }

  thead th {
    text-align: left;
    font-weight: 600;
    color: var(--text-dim);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }
  thead th:hover { color: var(--text); }
  thead th .sort-arrow { opacity: 0.5; margin-left: 4px; font-size: 10px; }

  tbody tr.result-row {
    border-bottom: 1px solid var(--border);
    cursor: pointer;
  }
  tbody tr.result-row:hover { background: var(--bg-raised); }

  tbody td {
    padding: 10px 12px;
    white-space: nowrap;
  }

  td.framework { color: var(--accent); font-weight: 600; }
  td.score { font-weight: 700; }

  tr.detail-row td {
    background: var(--bg-raised);
    padding: 16px;
    white-space: normal;
    border-bottom: 1px solid var(--border);
  }
  tr.detail-row { display: none; }
  tr.detail-row.open { display: table-row; }

  .detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }

  .detail-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px 12px;
  }
  .detail-card-label {
    color: var(--text-dim);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
  }
  .detail-card-score { font-size: 18px; font-weight: 700; color: var(--accent); }
  .detail-card-reason { color: var(--text-dim); font-size: 12px; margin-top: 4px; }

  .detail-output {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px;
    white-space: pre-wrap;
    font-size: 13px;
    color: var(--text);
    max-height: 240px;
    overflow-y: auto;
  }
  .detail-output-label {
    color: var(--text-dim);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
  }

  .row-actions { float: right; }
  .row-actions button {
    background: none;
    border: 1px solid var(--border);
    color: var(--text-dim);
    font-family: var(--mono);
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 3px;
    cursor: pointer;
  }
  .row-actions button:hover { color: var(--text); border-color: var(--text-dim); }

  .empty-state {
    color: var(--text-dim);
    text-align: center;
    padding: 48px 0;
  }

  .page-footer {
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    color: var(--text-dim);
    font-size: 12px;
  }

  .footer-note {
    margin-bottom: 12px;
    line-height: 1.6;
  }
  .footer-note strong { color: var(--text); }

  .raw-value {
    color: var(--text-dim);
    font-size: 12px;
  }

  .expand-caret {
    display: inline-block;
    margin-right: 6px;
    color: var(--text-dim);
    transition: transform 0.15s ease;
  }
  tr.result-row.expanded .expand-caret { transform: rotate(90deg); }
"""

_JS = """
  // The full dataset for this run, embedded at generation time by
  // output/dashboard.py. This is the single source of truth the page
  // renders from — see the module docstring in dashboard.py for why.
  let RESULTS = __EMBEDDED_RESULTS__;

  let sortKey = "composite_score";
  let sortDir = -1; // -1 = descending (highest score first, matches CLI default)
  const expandedIds = new Set();

  function dimensionNames() {
    if (RESULTS.length === 0) return [];
    return Object.keys(RESULTS[0].dimensions);
  }

  function formatDimensionValue(dim, key) {
    if (!dim) return "—";
    // latency/cost dimensions carry a more readable raw value in `reason`
    // (e.g. "1.8s", "$0.0098") — same convention as output/formatter.py.
    // Show both: the normalized score (for sorting/comparison) and the
    // raw value (so a viewer isn't left guessing what "100" means).
    if ((key === "cost" || key === "latency") && dim.reason) {
      return `${dim.score} <span class="raw-value">(${escapeHtml(dim.reason)})</span>`;
    }
    return dim.score;
  }

  function sortResults() {
    RESULTS.sort((a, b) => {
      let av = sortKey === "composite_score" ? a.composite_score
             : sortKey in a ? a[sortKey]
             : (a.dimensions[sortKey] ? a.dimensions[sortKey].score : 0);
      let bv = sortKey === "composite_score" ? b.composite_score
             : sortKey in b ? b[sortKey]
             : (b.dimensions[sortKey] ? b.dimensions[sortKey].score : 0);
      if (typeof av === "string") return sortDir * av.localeCompare(bv);
      return sortDir * (av - bv);
    });
  }

  function setSort(key) {
    if (sortKey === key) {
      sortDir *= -1;
    } else {
      sortKey = key;
      sortDir = -1;
    }
    render();
  }

  function toggleExpand(id) {
    if (expandedIds.has(id)) {
      expandedIds.delete(id);
    } else {
      expandedIds.add(id);
    }
    render();
  }

  // --- Where a future "delete row" feature would hook in ---
  // function deleteResult(id) {
  //   RESULTS = RESULTS.filter(r => r.id !== id);
  //   expandedIds.delete(id);
  //   render();
  // }
  // A delete button in renderTable() below would call deleteResult(r.id).
  // Remember: this only removes the row from THIS browser view — it does
  // not modify the results.json file the dashboard was generated from.

  // Some dimension keys need a clearer label than a simple underscore
  // replace would give — "cost" and "latency" are 0-100 scores normalized
  // *within this batch* (100 = cheapest/fastest row shown, not an
  // absolute scale), which isn't obvious from the column alone. See the
  // explanatory footer note in the static HTML (search _build_html for
  // "About Cost Score").
  const DIMENSION_LABELS = {
    cost: "Cost Score",
    latency: "Latency Score",
  };

  function dimensionLabel(key) {
    return DIMENSION_LABELS[key] || key.replace(/_/g, " ");
  }

  function renderHead() {
    const dims = dimensionNames();
    const head = document.getElementById("table-head");
    const cols = [
      { key: "framework", label: "Framework" },
      { key: "model_id", label: "Model" },
      ...dims.map(d => ({ key: d, label: dimensionLabel(d) })),
      { key: "composite_score", label: "Score" },
    ];
    head.innerHTML = "<tr>" + cols.map(c => {
      const arrow = sortKey === c.key ? (sortDir === -1 ? "▼" : "▲") : "";
      return `<th onclick="setSort('${c.key}')">${escapeHtml(c.label)}<span class="sort-arrow">${arrow}</span></th>`;
    }).join("") + "</tr>";
  }

  function renderTable() {
    const dims = dimensionNames();
    const body = document.getElementById("table-body");
    const emptyState = document.getElementById("empty-state");
    document.getElementById("run-count").textContent = RESULTS.length;

    if (RESULTS.length === 0) {
      body.innerHTML = "";
      emptyState.style.display = "block";
      return;
    }
    emptyState.style.display = "none";

    let rowsHtml = "";
    RESULTS.forEach(r => {
      const isOpen = expandedIds.has(r.id);
      const dimCells = dims.map(d => `<td>${formatDimensionValue(r.dimensions[d], d)}</td>`).join("");

      rowsHtml += `
        <tr class="result-row ${isOpen ? 'expanded' : ''}" onclick="toggleExpand('${r.id}')">
          <td class="framework"><span class="expand-caret">▶</span>${escapeHtml(r.framework)}</td>
          <td>${escapeHtml(r.model_id)}</td>
          ${dimCells}
          <td class="score">${r.composite_score}</td>
        </tr>
        <tr class="detail-row ${isOpen ? 'open' : ''}" id="detail-${r.id}">
          <td colspan="${3 + dims.length}">
            <div class="detail-grid">
              ${dims.map(d => {
                const dim = r.dimensions[d] || {};
                return `
                  <div class="detail-card">
                    <div class="detail-card-label">${escapeHtml(dimensionLabel(d))}</div>
                    <div class="detail-card-score">${dim.score ?? "—"}</div>
                    <div class="detail-card-reason">${escapeHtml(dim.reason ?? "")}</div>
                  </div>`;
              }).join("")}
            </div>
            <div class="detail-output-label">Output</div>
            <div class="detail-output">${escapeHtml(r.output || "(empty)")}</div>
          </td>
        </tr>
      `;
    });
    body.innerHTML = rowsHtml;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function render() {
    sortResults();
    renderHead();
    renderTable();
  }

  render();
"""
