"""One line per tool call, appended to a file you name — and nothing at all otherwise.

OFF unless `PAKLAW_TELEMETRY` holds a path. That default is not timidity: this package
ships with `dependencies = []` and a server that touches no file it was not asked about,
and a build that starts writing somewhere the moment it is installed would be a different
promise than the one on the badge. Set the variable and it writes exactly one file, the
one you named.

The format is JSON Lines: one object per line, append-only, no schema migration, no
database. `tail -f` works on it, `grep` works on it, `wc -l` counts your calls, and
deleting the file is a complete uninstall. `observability/` reads it; nothing else does.

Three rules this module keeps, because telemetry that breaks the thing it measures is
worse than no telemetry:

  * It never raises into the caller. Every failure path here ends in `return` - a full
    disk, a read-only directory, a path that is a directory, a surrogate that will not
    encode. The tool call carries on and answers.
  * It never slows a call down by more than a line write. No network, no lock file, no
    flush-and-fsync, no background thread to join at exit.
  * It records what the call DID, not what was asked. Arguments can hold a draft someone
    pasted; a question is a person's question. What goes in the file is the tool name,
    how long it took, whether it refused and under which status - the shape of the
    traffic, not its content.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

#: The variable that switches this on, and the file it names.
ENV = "PAKLAW_TELEMETRY"

#: Rotate past this, so a long-running server cannot fill a disk. One generation is
#: kept: `calls.jsonl` becomes `calls.jsonl.1` and a new file starts. Two files bounded
#: at 8 MB each is 16 MB worst case, which is a few hundred thousand calls.
MAX_BYTES = 8 * 1024 * 1024

#: Fields copied out of a result, when the result is a dict and carries them. Deliberately
#: short: these are the ones the console has a panel for. `refusal_reason` is prose and is
#: NOT copied - it can quote the question back, and the point of this file is the shape of
#: the traffic rather than its content.
_RESULT_FIELDS = ("refused", "refusal_status", "sample")


def destination() -> Path | None:
    """The file to append to, or None when telemetry is off."""
    raw = (os.environ.get(ENV) or "").strip()
    return Path(raw) if raw else None


def _rotate(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size >= MAX_BYTES:
            path.replace(path.with_suffix(path.suffix + ".1"))
    except OSError:
        # A rotation that cannot happen is not a reason to drop the event: the next
        # append will simply land in a file that is larger than intended.
        pass


def record(event: dict[str, Any]) -> None:
    """Append one event, if telemetry is on. Never raises."""
    path = destination()
    if path is None:
        return
    try:
        event.setdefault("at", time.time())
        line = json.dumps(event, ensure_ascii=False, default=str)
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate(path)
        # `newline=""` so a CRLF platform does not turn every record into two line
        # endings, which would make the file disagree with itself between hosts.
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(line + "\n")
    except (OSError, TypeError, ValueError, UnicodeError):
        # Every reachable failure, swallowed on purpose. A disk that filled up during a
        # citation check must not turn into a failed citation check.
        return


def from_result(result: Any) -> dict[str, Any]:
    """The few fields the console has a panel for, if the result carries them.

    `check_citations` returns a list of citations rather than one verdict, so the
    statuses are counted here instead of copied - the panel wants "how many of each",
    and the citations themselves are the draft someone pasted.
    """
    out: dict[str, Any] = {}
    if not isinstance(result, dict):
        return out
    for field in _RESULT_FIELDS:
        if field in result:
            out[field] = result[field]
    citations = result.get("citations")
    if isinstance(citations, list):
        statuses: dict[str, int] = {}
        for citation in citations:
            if isinstance(citation, dict):
                status = citation.get("status")
                if isinstance(status, str):
                    statuses[status] = statuses.get(status, 0) + 1
        if statuses:
            out["statuses"] = statuses
    return out
