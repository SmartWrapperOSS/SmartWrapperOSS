# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/guidance_extraction/task.py

What a guidance-extraction benchmark task looks like, how it's loaded from
YAML, and how an agent's raw text answer is parsed back into JSON.

A task holds two (or more) earnings-call excerpts, a closed vocabulary of
metrics with units and tolerances, and an answer key: the guidance items
each call contains and how guidance changed between calls. See
benchmarks/guidance_extraction/northwind_q2_q3_fy2026.yaml for a worked
example. Everything is in the YAML, so a new company, quarter pair, or
metric set is a new YAML file, not new code.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

CHANGE_TYPES = ["raised", "lowered", "maintained", "narrowed", "widened", "initiated", "withdrawn"]
PERIOD_RE = re.compile(r"^(Q[1-4])?FY\d{4}$")


def norm_period(p: Any) -> str:
    """'Q4 FY2026', 'q4-fy2026' and 'Q4FY2026' all become 'Q4FY2026'."""
    return re.sub(r"[\s_\-]", "", str(p or "")).upper()


def norm_metric(m: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(m or "").strip().lower())


@dataclass
class Document:
    id: str
    label: str
    text: str


@dataclass
class Metric:
    name: str
    unit: str
    tol: float
    description: str = ""


@dataclass
class GuidanceItem:
    source: str
    metric: str
    period: str
    low: Optional[float]
    high: Optional[float]


@dataclass
class ExpectedChange:
    metric: str
    period: str
    type: str
    accept: List[str]
    from_: Optional[Dict[str, Optional[float]]]
    to: Optional[Dict[str, Optional[float]]]


@dataclass
class Trap:
    source: str
    metric: str
    period: str
    value: Optional[float]
    description: str


@dataclass
class GuidanceTask:
    task_id: str
    company: str
    documents: List[Document]
    metrics: Dict[str, Metric]
    expected_guidance: List[GuidanceItem]
    expected_changes: List[ExpectedChange]
    traps: List[Trap] = field(default_factory=list)

    @property
    def doc_ids(self) -> List[str]:
        return [d.id for d in self.documents]


def _num(v, where: str) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where}: expected a number or null, got {v!r}")
    return float(v)


def _bounds(d, where: str) -> Optional[Dict[str, Optional[float]]]:
    if d is None:
        return None
    return {"low": _num(d.get("low"), where), "high": _num(d.get("high"), where)}


def load_task(yaml_path: str) -> GuidanceTask:
    """Load and sanity-check a task, so a typo in an answer key fails
    loudly at load time instead of silently skewing every score."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    documents = [Document(d["id"], d.get("label", d["id"]), d["text"]) for d in data["documents"]]
    if len(documents) < 2:
        raise ValueError(f"{yaml_path}: needs at least two documents to compare")
    doc_ids = {d.id for d in documents}

    metrics = {
        norm_metric(name): Metric(norm_metric(name), m["unit"], float(m.get("tol", 0)), m.get("description", ""))
        for name, m in data["metrics"].items()
    }

    def check(metric, period, where):
        if norm_metric(metric) not in metrics:
            raise ValueError(f"{where}: metric {metric!r} is not in the task's metrics list")
        if not PERIOD_RE.match(norm_period(period)):
            raise ValueError(f"{where}: period {period!r} must look like FY2026 or Q4FY2026")

    guidance = []
    for i, g in enumerate(data["expected_guidance"]):
        where = f"{yaml_path}: expected_guidance[{i}]"
        check(g["metric"], g["period"], where)
        if g["source"] not in doc_ids:
            raise ValueError(f"{where}: unknown source {g['source']!r}")
        guidance.append(GuidanceItem(g["source"], norm_metric(g["metric"]), norm_period(g["period"]),
                                     _num(g.get("low"), where), _num(g.get("high"), where)))

    changes = []
    for i, c in enumerate(data["expected_changes"]):
        where = f"{yaml_path}: expected_changes[{i}]"
        check(c["metric"], c["period"], where)
        accepted = [c["type"]] + list(c.get("accept", []))
        bad = [t for t in accepted if t not in CHANGE_TYPES]
        if bad:
            raise ValueError(f"{where}: unknown change type(s) {bad}; use one of {CHANGE_TYPES}")
        changes.append(ExpectedChange(norm_metric(c["metric"]), norm_period(c["period"]), c["type"],
                                      accepted, _bounds(c.get("from"), where), _bounds(c.get("to"), where)))

    traps = []
    for i, t in enumerate(data.get("traps", [])):
        where = f"{yaml_path}: traps[{i}]"
        check(t["metric"], t["period"], where)
        traps.append(Trap(t["source"], norm_metric(t["metric"]), norm_period(t["period"]),
                          _num(t.get("value"), where), t.get("description", "trap")))

    return GuidanceTask(
        task_id=data["task_id"],
        company=data.get("company", ""),
        documents=documents,
        metrics=metrics,
        expected_guidance=guidance,
        expected_changes=changes,
        traps=traps,
    )


# --- Shared prompt pieces ----------------------------------------------------
#
# Both runners describe the output format with the same text, so a score
# difference between them reflects orchestration, not a different spec.

def metrics_block(task: GuidanceTask) -> str:
    return "\n".join(f"- {m.name}: {m.description} (unit: {m.unit})" for m in task.metrics.values())


def documents_block(task: GuidanceTask, ids: Optional[List[str]] = None) -> str:
    docs = [d for d in task.documents if ids is None or d.id in ids]
    return "\n\n".join(f"[document id: {d.id}] {d.label}\n{d.text.strip()}" for d in docs)


RULES = """RULES
- Guidance means a forward-looking expectation stated BY THE COMPANY for a
  future period. Reported results, analyst or Street consensus, and industry
  or market forecasts are NOT guidance. Leave them out.
- Report only the metrics listed above, using the metric names exactly as
  written and numbers in the stated unit (e.g. $4.1 billion with unit
  "USD billions" is 4.1; 42% with unit "percent" is 42).
- Periods use the form FY2026 for a fiscal year or Q4FY2026 for a quarter.
- A range gives low and high. A single number ("approximately $300 million")
  gives low == high. An open bound ("at least $500 million") gives the known
  side and null for the other.
- Compare guidance for the same metric and period between the earlier and
  the later document. Change types: raised, lowered, maintained, narrowed,
  widened, initiated (new in the later document), withdrawn (the later
  document says it is no longer provided). Do not report a change for a
  period that has already ended by the later document."""

OUTPUT_FORMAT = """Respond with ONLY a JSON object, no prose and no Markdown fences:
{
  "guidance": [
    {"source": "<document id>", "metric": "<metric>", "period": "<period>",
     "low": <number or null>, "high": <number or null>}
  ],
  "changes": [
    {"metric": "<metric>", "period": "<period>", "type": "<change type>",
     "from": {"low": <number or null>, "high": <number or null>} or null,
     "to": {"low": <number or null>, "high": <number or null>} or null}
  ]
}"""


def parse_output(text: str) -> Optional[dict]:
    """Best-effort JSON extraction: strips ```json fences and surrounding
    prose, then parses the outermost {...}. Returns None if nothing parses.
    Being lenient here is deliberate: the schema_validity dimension already
    scores format discipline, so a stray fence shouldn't zero every other
    dimension too."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
