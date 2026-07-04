# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
core/runstore.py

Persists every raw run to disk as one JSON file per run, and reads them
back for aggregation.

WHY THIS EXISTS (the single most important design decision of multi-run
support): collection and analysis are SEPARATE steps. Every run —
completion text, token counts, latency, cost, scores — is written to
disk the moment it's scored. Aggregation then reads *from disk*, not
from memory. Consequences:

  - Resumability: if run 4 of 5 dies on a rate limit, re-running the
    same command only executes the missing runs. You never pay twice.
  - Incremental n: run with --runs 3 today, --runs 10 next week; the
    aggregate quietly grows to n=10.
  - Auditability: anyone can re-derive (or dispute) your published
    numbers from the raw JSON. Publish the runs/ directory alongside
    results and the benchmark becomes independently verifiable.

LAYOUT
    <root>/<workflow>/<task_tag>/<framework>__<model>__run003.json

The task_tag (benchmark filename stem, or document basename for
summarization) keeps runs of *different tasks* from being aggregated
together as if they were repeats of the same experiment.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

_FNAME_RE = re.compile(r"^(?P<fw>.+)__(?P<model>.+)__run(?P<idx>\d{3})\.json$")


def _safe(s: str) -> str:
    """Make a string safe for use in a filename (model ids can contain
    '/', ':' etc. depending on provider naming)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s)


class RunStore:
    def __init__(self, root: str, workflow_name: str, task_tag: str):
        self.dir = Path(root) / _safe(workflow_name) / _safe(task_tag)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _fname(self, framework: str, model_id: str, run_index: int) -> Path:
        return self.dir / f"{_safe(framework)}__{_safe(model_id)}__run{run_index:03d}.json"

    def completed_indices(self, framework: str, model_id: str) -> Set[int]:
        """Which run indices already exist on disk for this combo?
        This is what makes `--runs N` resumable: the caller only
        schedules indices not in this set."""
        prefix = f"{_safe(framework)}__{_safe(model_id)}__run"
        done = set()
        for p in self.dir.glob(f"{prefix}*.json"):
            m = _FNAME_RE.match(p.name)
            if m:
                done.add(int(m.group("idx")))
        return done

    def save(self, framework: str, model_id: str, run_index: int,
             record: Dict[str, Any]) -> Path:
        record = dict(record)
        record.setdefault("framework", framework)
        record.setdefault("model_id", model_id)
        record["run_index"] = run_index
        record["saved_at"] = datetime.now(timezone.utc).isoformat()
        path = self._fname(framework, model_id, run_index)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(record, f, indent=2, default=str)
        tmp.replace(path)  # atomic-ish: never leaves a half-written run
        return path

    def load_all(self) -> List[Dict[str, Any]]:
        """All persisted runs for this (workflow, task), across every
        session. Aggregation always operates on this, so previously
        completed runs contribute to the stats automatically."""
        records = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                with open(p) as f:
                    records.append(json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                print(f"  warning: skipping unreadable run file {p.name}: {e}")
        return records
