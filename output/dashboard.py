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


# --- Aggregate (multi-run) dashboard -----------------------------------------
#
# Same philosophy as save_html above: one self-contained file, no server,
# works by double-clicking. Differences from the single-run dashboard:
#
#   - The headline visual is an SVG ERROR-BAR CHART of composite scores
#     (dot = mean, whiskers = 95% CI). This is deliberately the first
#     thing on the page: error bars are the whole point of multi-run
#     mode, and this chart is the screenshot people share.
#   - Rows are (framework, model) AGGREGATES, not individual runs.
#     Expanding a row shows per-dimension stats and a per-run breakdown.
#   - Mostly rendered server-side (Python builds the HTML); JS is only
#     used for expand/collapse. Aggregates don't need client-side
#     sorting/deleting the way exploratory single-run views do.

import html as _html
from core.stats import ComboAggregate


def save_aggregate_html(aggs: List[ComboAggregate], records: list,
                        path: str, workflow_name: str = ""):
    """Write a self-contained aggregate dashboard for `aggs` to `path`.

    `records` are the raw run dicts (from RunStore.load_all()) used to
    render the per-run breakdown inside each expanded row."""
    out = _build_aggregate_html(aggs, records, workflow_name)
    with open(path, "w") as f:
        f.write(out)
    print(f"Aggregate dashboard saved to {path} (open it directly in a browser)")


def _esc(s) -> str:
    return _html.escape(str(s), quote=True)


def _svg_ci_chart(aggs: List[ComboAggregate]) -> str:
    """Horizontal error-bar chart: composite mean (dot) with 95% CI
    (whiskers) per (framework, model) combination."""
    if not aggs:
        return ""

    width, row_h, pad_top = 760, 40, 34
    label_w, value_w = 250, 84
    plot_x0, plot_x1 = label_w + 12, width - value_w
    height = pad_top + row_h * len(aggs) + 30

    lows = [a.composite.ci_low for a in aggs]
    highs = [a.composite.ci_high for a in aggs]
    span = max(highs) - min(lows) or 1.0
    d0, d1 = min(lows) - 0.08 * span, max(highs) + 0.08 * span

    def x(v):
        return plot_x0 + (v - d0) / (d1 - d0) * (plot_x1 - plot_x0)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="Composite score with 95% confidence intervals">',
        f'<text x="{plot_x0}" y="18" class="svg-axis">composite score '
        f'(dot = mean over n runs, whiskers = 95% CI)</text>',
    ]
    # light vertical gridlines at round values
    step = max(1, round(span / 4))
    v = int(d0) - int(d0) % step
    while v <= d1:
        if v >= d0:
            parts.append(f'<line x1="{x(v):.1f}" y1="{pad_top - 6}" x2="{x(v):.1f}" '
                         f'y2="{height - 22}" class="svg-grid"/>')
            parts.append(f'<text x="{x(v):.1f}" y="{height - 8}" class="svg-axis" '
                         f'text-anchor="middle">{v}</text>')
        v += step

    for i, a in enumerate(aggs):
        cy = pad_top + row_h * i + row_h / 2
        c = a.composite
        cls = "svg-best" if i == 0 else "svg-other"
        label = f"{a.framework} + {a.model_id}"
        parts.append(f'<text x="{label_w}" y="{cy + 4:.1f}" text-anchor="end" '
                     f'class="svg-label">{_esc(label)}</text>')
        if a.n > 1:
            parts.append(f'<line x1="{x(c.ci_low):.1f}" y1="{cy:.1f}" '
                         f'x2="{x(c.ci_high):.1f}" y2="{cy:.1f}" class="{cls} svg-ci"/>')
            for vv in (c.ci_low, c.ci_high):  # whisker end ticks
                parts.append(f'<line x1="{x(vv):.1f}" y1="{cy - 6:.1f}" '
                             f'x2="{x(vv):.1f}" y2="{cy + 6:.1f}" class="{cls} svg-ci"/>')
        parts.append(f'<circle cx="{x(c.mean):.1f}" cy="{cy:.1f}" r="5" class="{cls} svg-dot"/>')
        parts.append(f'<text x="{plot_x1 + 10}" y="{cy + 4:.1f}" class="svg-value">'
                     f'{c.mean:.1f} ±{c.std:.1f}</text>')

    parts.append("</svg>")
    return "".join(parts)


def _overlap_note(aggs: List[ComboAggregate]) -> str:
    if len(aggs) < 2 or aggs[0].n < 2 or aggs[1].n < 2:
        return ""
    a, b = aggs[0], aggs[1]
    if a.composite.overlaps(b.composite):
        return (f'<p class="overlap-note">Note: the 95% CIs of the top two combinations '
                f'overlap — this ranking is <strong>not statistically distinguishable</strong> '
                f'at n={a.n}. Add runs to tighten the intervals.</p>')
    return (f'<p class="overlap-note">The 95% CIs of the top two combinations do not '
            f'overlap — the lead of <strong>{_esc(a.framework)} + {_esc(a.model_id)}</strong> '
            f'holds at this sample size.</p>')


def _build_aggregate_html(aggs: List[ComboAggregate], records: list,
                          workflow_name: str) -> str:
    dims = list(aggs[0].dimensions.keys()) if aggs else []
    n_values = sorted({a.n for a in aggs}) or [0]
    n_desc = f"n={n_values[0]}" if len(n_values) == 1 else f"n={n_values[0]}..{n_values[-1]}"

    # group raw runs per combo for the expandable per-run breakdown
    runs_by_combo = {}
    for r in records:
        runs_by_combo.setdefault((r["framework"], r["model_id"]), []).append(r)

    head_cells = "".join(
        f"<th>{_esc(d.replace('_', ' ').title())}</th>" for d in dims)
    rows = []
    for i, a in enumerate(aggs):
        rid = f"agg-{i}"
        dim_cells = "".join(
            f"<td>{a.dimensions[d].mean:.1f} <span class='raw-value'>"
            f"±{a.dimensions[d].std:.1f}</span></td>" for d in dims)
        c = a.composite
        rows.append(f"""
        <tr class="result-row" onclick="toggleExpand('{rid}')">
          <td class="framework"><span class="expand-caret">&#9654;</span>{_esc(a.framework)}</td>
          <td>{_esc(a.model_id)}</td>
          {dim_cells}
          <td>{a.latency_ms_median / 1000:.1f}s <span class="raw-value">/ {a.latency_ms_p95 / 1000:.1f}s</span></td>
          <td>${a.cost_usd_mean:.4f}</td>
          <td class="score">{c.mean:.1f} <span class="raw-value">±{c.std:.1f}</span></td>
        </tr>""")

        dim_cards = "".join(f"""
              <div class="detail-card">
                <div class="detail-card-label">{_esc(d.replace('_', ' ').title())}</div>
                <div class="detail-card-score">{a.dimensions[d].mean:.1f} <span class="raw-value">±{a.dimensions[d].std:.1f}</span></div>
                <div class="detail-card-reason">95% CI [{a.dimensions[d].ci_low:.1f}, {a.dimensions[d].ci_high:.1f}] ·
                range [{a.dimensions[d].min:.1f}, {a.dimensions[d].max:.1f}]</div>
              </div>""" for d in dims)

        run_rows = "".join(
            f"<tr><td>run {r.get('run_index', '?')}</td>"
            f"<td>{float(r['composite_score']):.1f}</td>"
            f"<td>{float(r['latency_ms']) / 1000:.1f}s</td>"
            f"<td>${float(r['cost_usd']):.4f}</td>"
            f"<td class='raw-value'>{_esc(r.get('saved_at', ''))}</td></tr>"
            for r in sorted(runs_by_combo.get((a.framework, a.model_id), []),
                            key=lambda r: r.get("run_index", 0)))

        rows.append(f"""
        <tr class="detail-row" id="detail-{rid}">
          <td colspan="{5 + len(dims)}">
            <div class="detail-grid">{dim_cards}
              <div class="detail-card">
                <div class="detail-card-label">Composite</div>
                <div class="detail-card-score">{c.mean:.1f} <span class="raw-value">±{c.std:.1f}</span></div>
                <div class="detail-card-reason">95% CI [{c.ci_low:.1f}, {c.ci_high:.1f}] ·
                range [{c.min:.1f}, {c.max:.1f}] · n={a.n} · total cost ${a.cost_usd_total:.4f}</div>
              </div>
            </div>
            <div class="detail-output-label">Individual runs (full outputs in the runs/ directory)</div>
            <table class="runs-table">
              <thead><tr><th>Run</th><th>Composite</th><th>Latency</th><th>Cost</th><th>Saved</th></tr></thead>
              <tbody>{run_rows}</tbody>
            </table>
          </td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartWrapperOSS — Aggregate Results{f' ({workflow_name})' if workflow_name else ''}</title>
<style>
{_CSS}
{_AGG_CSS}
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
        <span class="meta-value">{_esc(workflow_name) or 'unknown'}</span>
        <span class="meta-sep">·</span>
        <span class="meta-label">aggregated over</span>
        <span class="meta-value">{n_desc} runs per combination</span>
      </div>
    </header>

    <main>
      <div class="chart-wrap">{_svg_ci_chart(aggs)}</div>
      {_overlap_note(aggs)}
      <table id="results-table">
        <thead><tr>
          <th>Framework</th><th>Model</th>{head_cells}
          <th>Lat med/p95</th><th>Cost mean</th><th>Composite</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </main>

    <footer class="page-footer">
      <p class="footer-note">
        <strong>How to read this:</strong> score cells are mean ± sample std
        dev over the runs shown; the chart's whiskers are 95% confidence
        intervals (Student's t). Latency is reported as median / p95 of raw
        seconds and cost as mean dollars per run — their normalized 0–100
        scores are relative to a single comparison batch and are therefore
        never averaged across runs. Overlapping CIs mean the ranking is not
        established at this sample size; they do not mean the combinations
        are equal.
      </p>
      <p>
        Generated by SmartWrapperOSS. Static snapshot — raw per-run records
        (including full model outputs) live in the runs/ directory and can be
        used to independently re-derive every number on this page.
      </p>
    </footer>
  </div>

<script>
  function toggleExpand(id) {{
    const row = document.getElementById("detail-" + id);
    const trigger = row.previousElementSibling;
    row.classList.toggle("open");
    trigger.classList.toggle("expanded");
  }}
</script>
</body>
</html>
"""


_AGG_CSS = """
  .chart-wrap {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 12px 8px;
    margin-bottom: 20px;
  }
  .chart-wrap svg { width: 100%; height: auto; display: block; }
  .svg-axis  { fill: var(--text-dim); font-family: var(--mono); font-size: 11px; }
  .svg-label { fill: var(--text); font-family: var(--mono); font-size: 12px; }
  .svg-value { fill: var(--text-dim); font-family: var(--mono); font-size: 12px; }
  .svg-grid  { stroke: var(--border); stroke-width: 1; }
  .svg-ci    { stroke-width: 2; }
  .svg-best.svg-ci  { stroke: var(--accent); }
  .svg-best.svg-dot { fill: var(--accent); }
  .svg-other.svg-ci  { stroke: var(--text-dim); }
  .svg-other.svg-dot { fill: var(--text-dim); }

  .overlap-note {
    color: var(--text-dim);
    font-size: 13px;
    margin: 0 0 20px;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    background: var(--accent-dim);
  }
  .overlap-note strong { color: var(--text); }

  .runs-table { width: auto; min-width: 60%; }
  .runs-table th, .runs-table td { padding: 6px 14px 6px 0; font-size: 12px; }
  .runs-table thead th { cursor: default; }
"""
