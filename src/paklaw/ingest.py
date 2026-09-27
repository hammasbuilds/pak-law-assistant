"""Building and maintaining a temporal corpus without hand-writing JSON.

Two jobs, both of which are where a real corpus goes wrong:

**Splitting an Act into provisions.** Official texts lay each section out as a number, a
heading and a body — ``20. Offences against dignity of a natural person.—(1) Whoever``.
The number is the only reliable boundary, and it is also the thing most easily confused
with a numbered clause or a year inside a body. So a candidate heading is accepted only
if its number comes *after* the previous one in statute order; anything else stays body
text and is reported, not silently swallowed or silently split.

**Recording amendments.** A substitution is two edits that must agree: the live version
ends on a date and the new one begins on the same date. Done by hand, one of them gets
forgotten and the corpus holds two live versions or a gap — which `Corpus.validate`
catches, but only after the fact. These operations make both edits together and refuse
the ones that cannot be right (substituting a provision that is not in force, or with
effect from before its current version began).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import tempfile
from pathlib import Path

from .audit import provision_order
from .citation import normalise_number, parse, statute_aliases
from .corpus import Corpus, CorpusError, Provision

_HEADER = re.compile(
    r"^[ \t]*(?P<number>\d{1,4}(?:-?[A-Z]{1,3})?)\.[ \t]+"
    r"(?P<heading>[A-Z\[][^\n]{1,240}?)"
    # The heading ends at ".—", ":—", ".-", a bare em/en dash, or ": ".
    r"(?:[ \t]*[.:][ \t]*[—–-]+|[ \t]*[—–]|:[ \t])"
    r"[ \t]*(?P<rest>.*)$",
    re.M,
)
_CHAPTER = re.compile(r"^[ \t]*(CHAPTER|PART)[ \t]+([IVXLC\d]+[A-Z]?)\b[ \t.:—–-]*(.*)$", re.M)
_OMITTED_LINE = re.compile(
    r"^[ \t]*(?P<number>\d{1,4}(?:-?[A-Z]{1,3})?)\.[ \t]*"
    r"\[[ \t]*(?:omitted|repealed)\b[^\]\n]*\]\.?[ \t]*$",
    re.M | re.I,
)
# Schedules follow the last section and are not provisions. Missed, the whole schedule
# becomes the tail of the last section's text.
_SCHEDULE = re.compile(
    r"^[ 	]*(?:THE[ 	]+)?(?:[A-Z]+[ 	]+)?SCHEDULE[ 	]*(?:[-—–:.(].*)?$", re.M
)
_DROPPED = re.compile(r"^\[?\s*(omitted|repealed)\b", re.I)


def split_act(
    text: str,
    *,
    statute: str,
    in_force_from: str,
    unit: str = "section",
) -> tuple[list[dict], dict]:
    """Rows for `corpus.build`, and a report of everything that needs a human look."""
    dt.date.fromisoformat(in_force_from)  # fail before doing any work
    schedules = ""
    first_header = _HEADER.search(text)
    schedule = _SCHEDULE.search(text, first_header.end() if first_header else 0)
    if schedule:
        schedules = text[schedule.start() :]
        text = text[: schedule.start()]
    # "4. [Omitted by ...]" has no heading dash, so it is not a heading — but it is still
    # a boundary. Missed, it is swallowed into the end of section 3.
    candidates = {m.start(): (m, False) for m in _HEADER.finditer(text)}
    candidates.update({m.start(): (m, True) for m in _OMITTED_LINE.finditer(text)})

    headers: list[tuple[re.Match, bool]] = []
    rejected = []
    last = None
    for _, (match, omitted) in sorted(candidates.items()):
        order = provision_order(normalise_number(match.group("number")))
        if last is not None and order <= last:
            rejected.append(
                f"'{match.group(0).strip()[:60]}' looks like a heading but its number does "
                "not follow the previous provision; kept as body text"
            )
            continue
        headers.append((match, omitted))
        last = order

    chapters = [
        (m.start(), f"{m.group(1).title()} {m.group(2)} {m.group(3)}".strip())
        for m in _CHAPTER.finditer(text)
    ]

    rows: list[dict] = []
    dropped: list[str] = []
    for index, (match, omitted) in enumerate(headers):
        number = normalise_number(match.group("number"))
        if omitted:
            dropped.append(f"{unit} {number}: '{match.group(0).strip()[:60]}' — record its dates")
            continue
        end = headers[index + 1][0].start() if index + 1 < len(headers) else len(text)
        body = match.group("rest") + text[match.end() : end]
        # Chapter headings sit between sections; they belong to the next one, not the body.
        body = _CHAPTER.split(body)[0] if _CHAPTER.search(body) else body
        body = re.sub(r"\s+", " ", body).strip()
        heading = match.group("heading").strip().rstrip(".")

        if _DROPPED.match(heading) or _DROPPED.match(body):
            dropped.append(f"{unit} {number}: '{heading[:50]}' — record its dates by hand")
            continue

        chapter = next((name for start, name in reversed(chapters) if start < match.start()), "")
        rows.append(
            {
                "statute": statute,
                "unit": unit,
                "number": number,
                "heading": heading,
                "text": body,
                "in_force_from": in_force_from,
                **({"chapter": chapter} if chapter else {}),
            }
        )

    report = {
        "provisions": len(rows),
        "empty": [f"{unit} {r['number']}" for r in rows if not r["text"]],
        "omitted_or_repealed": dropped,
        "rejected_headings": rejected,
        "not_imported": (
            [f"schedule text ({len(schedules)} characters) after the last section"]
            if schedules.strip()
            else []
        ),
    }
    return rows, report


# ---- amendments --------------------------------------------------------------------------


def _key(citation_text: str, rows: list[dict]) -> tuple[str, str, str]:
    aliases = statute_aliases({r.get("statute") for r in rows})
    found = [c for c in parse(citation_text, statutes=aliases) if c.kind == "statutory"]
    if not found:
        raise CorpusError(f"no statutory citation in {citation_text!r}")
    c = found[0]
    if not c.statute:
        raise CorpusError(f"{c.pretty()!r} names no Act")
    if "(" in c.provision:
        # A corpus holds whole provisions; an amended subsection is a new text of the
        # section it belongs to, or lookups would find the stale section beside it.
        raise CorpusError(
            f"{c.pretty()!r} is a subdivision; amend the whole provision with its full text"
        )
    return c.statute, c.unit, c.provision


def _live(rows: list[dict], key: tuple[str, str, str]) -> dict | None:
    return next(
        (
            r
            for r in rows
            if (r["statute"], r["unit"], r["number"]) == key and not r.get("in_force_to")
        ),
        None,
    )


def _close(row: dict, on: str, manner: str, by: str) -> None:
    if dt.date.fromisoformat(on) <= dt.date.fromisoformat(row["in_force_from"]):
        raise CorpusError(
            f"{row['unit']} {row['number']} {row['statute']}: its current version began "
            f"{row['in_force_from']}, so it cannot end on {on}"
        )
    row.update(in_force_to=on, manner=manner, amended_by=by)


def substitute(
    rows: list[dict], citation: str, *, on: str, by: str, text: str, heading: str | None = None
) -> list[dict]:
    key = _key(citation, rows)
    live = _live(rows, key)
    if live is None:
        raise CorpusError(f"{citation!r} has no version in force to substitute")
    _close(live, on, "substituted", by)
    live["superseded_by"] = f"{key[1]} {key[2]} {key[0]} as substituted {on}"
    rows.append(
        {
            **{k: v for k, v in live.items() if k in ("statute", "unit", "number", "chapter")},
            "heading": heading or live["heading"],
            "text": text,
            "in_force_from": on,
            "enacted_by": by,
        }
    )
    return rows


def repeal(
    rows: list[dict], citation: str, *, on: str, by: str, manner: str = "repealed"
) -> list[dict]:
    live = _live(rows, _key(citation, rows))
    if live is None:
        raise CorpusError(f"{citation!r} has no version in force to repeal")
    _close(live, on, manner, by)
    return rows


def insert(
    rows: list[dict], citation: str, *, on: str, by: str, text: str, heading: str
) -> list[dict]:
    statute, unit, number = _key(citation, rows)
    if _live(rows, (statute, unit, number)) is not None:
        raise CorpusError(f"{citation!r} is already in force; substitute it instead")
    dt.date.fromisoformat(on)
    rows.append(
        {
            "statute": statute,
            "unit": unit,
            "number": number,
            "heading": heading,
            "text": text,
            "in_force_from": on,
            "enacted_by": by,
        }
    )
    return rows


# ---- files -------------------------------------------------------------------------------

_FIELDS = set(Provision.__dataclass_fields__)
_REQUIRED = {"statute", "unit", "number", "heading", "text", "in_force_from"}


def read_rows(path: str | Path) -> list[dict]:
    """A JSON array of provisions, or JSON Lines with one provision per line."""
    text = Path(path).read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path}: line {number} is not JSON: {exc.msg}") from exc
    if not isinstance(rows, list):
        raise CorpusError(f"{path}: expected a JSON array or JSON Lines of provisions")
    return rows


def build_checked(rows: list[dict], *, source: str = "corpus") -> Corpus:
    """`corpus.build`, with the row at fault named instead of a bare TypeError."""
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise CorpusError(f"{source}: provision {index} is not an object")
        unknown = set(row) - _FIELDS
        if unknown:
            raise CorpusError(f"{source}: provision {index} has unknown fields {sorted(unknown)}")
        missing = _REQUIRED - set(row)
        if missing:
            raise CorpusError(f"{source}: provision {index} is missing {sorted(missing)}")
    corpus = Corpus()
    for index, row in enumerate(rows, 1):
        try:
            corpus.add(Provision(**row))
        except (TypeError, ValueError) as exc:
            where = f"{row.get('statute')} {row.get('unit')} {row.get('number')}"
            raise CorpusError(f"{source}: provision {index} ({where}): {exc}") from exc
    return corpus


def check(rows: list[dict], *, source: str = "corpus") -> list[str]:
    """Structural problems, or [] — the same gate the server applies at start."""
    return build_checked(rows, source=source).validate()


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    """Atomically: a failed write must not leave half a statute book behind."""
    path = Path(path)
    ordered = sorted(
        rows,
        key=lambda r: (r["statute"], r["unit"], provision_order(r["number"]), r["in_force_from"]),
    )
    handle, temporary = tempfile.mkstemp(dir=path.parent or ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as out:
            for row in ordered:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


# ---- command line ------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from .citation import normalise_statute

    parser = argparse.ArgumentParser(
        prog="paklaw-corpus",
        description="Build and amend a temporal statute corpus (JSON Lines) for paklaw-mcp.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    imp = commands.add_parser("import", help="split an Act's text into provisions")
    imp.add_argument("text_file", help="the Act as plain text (UTF-8)")
    imp.add_argument("--statute", required=True, help="PPC, PECA, CrPC, ... or a full name")
    imp.add_argument("--in-force-from", required=True, help="commencement date, YYYY-MM-DD")
    imp.add_argument("--unit", default="section", choices=["section", "article"])
    imp.add_argument("-o", "--output", required=True, help="corpus file; appended if it exists")

    for name, help_text in (
        ("substitute", "replace a provision's text with effect from a date"),
        ("insert", "add a new provision with effect from a date"),
        ("repeal", "end a provision with effect from a date"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("corpus")
        sub.add_argument("citation", help="e.g. 'section 20 PECA'")
        sub.add_argument("--on", required=True, help="effective date, YYYY-MM-DD")
        sub.add_argument("--by", required=True, help="the amending instrument")
        if name != "repeal":
            sub.add_argument("--text-file", required=True, help="the full new text")
            sub.add_argument("--heading", required=name == "insert")
        else:
            sub.add_argument("--omitted", action="store_true", help="record as 'omitted'")

    chk = commands.add_parser("check", help="validate a corpus file")
    chk.add_argument("corpus")

    args = parser.parse_args(argv)

    def fail(message: str) -> int:
        print(f"paklaw-corpus: {message}", file=sys.stderr)
        return 2

    try:
        if args.command == "check":
            rows = read_rows(args.corpus)
            problems = check(rows, source=args.corpus)
            for problem in problems:
                print(problem)
            print(f"{len(rows)} versions, {len(problems)} problem(s)")
            return 1 if problems else 0

        if args.command == "import":
            statute = normalise_statute(args.statute) or args.statute.strip().upper()
            text = Path(args.text_file).read_text(encoding="utf-8-sig")
            new, report = split_act(
                text, statute=statute, in_force_from=args.in_force_from, unit=args.unit
            )
            if not new:
                return fail("no provisions found; is the text laid out as '1. Heading.— ...'?")
            output = Path(args.output)
            existing = read_rows(output) if output.exists() else []
            clash = {(r["statute"], r["unit"], r["number"]) for r in existing} & {
                (r["statute"], r["unit"], r["number"]) for r in new
            }
            if clash:
                return fail(f"{len(clash)} provision(s) of {statute} already in {output}")
            rows = existing + new
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            rows = read_rows(args.corpus)
            output = Path(args.corpus)
            common = {"on": args.on, "by": args.by}
            if args.command == "repeal":
                repeal(
                    rows, args.citation, manner="omitted" if args.omitted else "repealed", **common
                )
            else:
                text = re.sub(r"\s+", " ", Path(args.text_file).read_text("utf-8-sig")).strip()
                operation = substitute if args.command == "substitute" else insert
                operation(rows, args.citation, text=text, heading=args.heading, **common)

        problems = check(rows, source=str(output))
        if problems:
            # Nothing is written: a corpus the server would refuse is not saved.
            return fail("refusing to write an invalid corpus:\n  " + "\n  ".join(problems))
        write_jsonl(rows, output)
        print(f"wrote {len(rows)} versions to {output}", file=sys.stderr)
        return 0
    except (OSError, ValueError, CorpusError) as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
