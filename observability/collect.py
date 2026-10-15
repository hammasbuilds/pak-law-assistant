"""JSON Lines in, panels out. No server, no I/O beyond reading the file.

Everything here is a pure function of the events, so a panel can be checked without
starting anything, and `serve.py` holds no arithmetic of its own - a console that
computes its numbers in the page is a console whose numbers cannot be tested.

Two decisions worth stating:

**Statuses are counted, never enumerated.** `refusal_status` and the citation statuses
are grouped by whatever value appears in the file. A fixed list here would have been
wrong within the hour: it was written from a grep of the source that found eleven
statuses, and the first real call produced `subject_not_in_corpus`, which was not among
them. A console that silently drops a status it has not been told about is worse than one
that shows an unfamiliar name.

**A percentile over very few calls is not a percentile.** `p95` of three calls is the
largest of three. Each one carries the count it was computed from, and the console prints
that count beside it, so nobody reads a number off six samples as if it were a service
level.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

#: Below this many samples a percentile is reported with a warning rather than withheld -
#: withholding it hides the traffic that exists, and this is a console, not a contract.
THIN = 20


def read(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Every readable event, oldest first. A damaged line is skipped, not fatal.

    A log is appended to by a live process, so the last line can be half-written at the
    moment it is read. One truncated line must not empty the console.
    """
    events: list[dict[str, Any]] = []
    for candidate in (path.with_suffix(path.suffix + ".1"), path):
        if not candidate.exists():
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                events.append(event)
    events.sort(key=lambda e: e.get("at") or 0)
    return events[-limit:] if limit else events


def within(events: Iterable[dict], seconds: float | None) -> list[dict]:
    """The events from the last `seconds`, or all of them when seconds is None."""
    if not seconds:
        return list(events)
    cutoff = time.time() - seconds
    return [e for e in events if (e.get("at") or 0) >= cutoff]


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank, which needs no interpolation and no numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


def summarise(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Every panel the console draws."""
    durations = [float(e.get("ms") or 0) for e in events]
    outcomes = Counter(str(e.get("outcome") or "unknown") for e in events)

    by_tool: dict[str, dict[str, Any]] = {}
    for event in events:
        tool = str(event.get("tool") or "unknown")
        row = by_tool.setdefault(
            tool,
            {"tool": tool, "calls": 0, "ok": 0, "refused": 0, "errors": 0, "rejected": 0, "ms": []},
        )
        row["calls"] += 1
        outcome = str(event.get("outcome") or "unknown")
        if outcome == "ok":
            row["ok"] += 1
        elif outcome == "refused":
            row["refused"] += 1
        elif outcome == "rejected":
            row["rejected"] += 1
        else:
            row["errors"] += 1
        row["ms"].append(float(event.get("ms") or 0))

    tools = []
    for row in by_tool.values():
        times = row.pop("ms")
        row["p50"] = round(_percentile(times, 0.50), 2)
        row["p95"] = round(_percentile(times, 0.95), 2)
        row["max"] = round(max(times), 2) if times else 0.0
        row["thin"] = len(times) < THIN
        row["refusal_rate"] = round(row["refused"] / row["calls"], 3) if row["calls"] else 0.0
        tools.append(row)
    tools.sort(key=lambda r: (-r["calls"], r["tool"]))

    # The signature panel: why this server declined to answer. Grouped by whatever
    # status the file holds, never by a list typed here.
    refusals = Counter(
        str(e["refusal_status"]) for e in events if e.get("refused") and e.get("refusal_status")
    )

    # Citation statuses across every check_citations call, summed.
    citations: Counter[str] = Counter()
    for event in events:
        statuses = event.get("statuses")
        if isinstance(statuses, dict):
            for status, count in statuses.items():
                try:
                    citations[str(status)] += int(count)
                except (TypeError, ValueError):
                    continue

    errors = Counter(
        str(e.get("error") or "unnamed")
        for e in events
        if str(e.get("outcome")) == "internal_error"
    )

    sampled = [bool(e["sample"]) for e in events if "sample" in e]

    return {
        "calls": len(events),
        "first": min((e.get("at") or 0) for e in events) if events else None,
        "last": max((e.get("at") or 0) for e in events) if events else None,
        "outcomes": dict(outcomes.most_common()),
        "refused": outcomes.get("refused", 0),
        "refusal_rate": round(outcomes.get("refused", 0) / len(events), 3) if events else 0.0,
        "p50": round(_percentile(durations, 0.50), 2),
        "p95": round(_percentile(durations, 0.95), 2),
        "max": round(max(durations), 2) if durations else 0.0,
        "thin": len(durations) < THIN,
        "thin_at": THIN,
        "tools": tools,
        "refusals": dict(refusals.most_common()),
        "citation_statuses": dict(citations.most_common()),
        "errors": dict(errors.most_common()),
        # `None` rather than False when no call recorded it: "every call used the sample
        # corpus" and "no call said which corpus it used" are different facts.
        "sample_corpus": (all(sampled) if sampled else None),
        "recent": [
            {k: v for k, v in e.items() if k in ("at", "tool", "outcome", "ms", "refusal_status")}
            for e in events[-25:][::-1]
        ],
    }
