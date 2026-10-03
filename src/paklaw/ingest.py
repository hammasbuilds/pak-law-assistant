"""Building and maintaining a temporal corpus without hand-writing JSON.

Two jobs, both of which are where a real corpus goes wrong.

**Splitting an Act into provisions** is `split.split_act`, which accounts for every
character it does not import; cleaning a source's PDF or HTML first is `sources`.

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
from .citation import parse, statute_aliases
from .corpus import Corpus, CorpusError, Provision
from .split import split_act

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


META_KEY = "_corpus"

# Instruments numbered in articles. Stored as sections, "Article 6" finds nothing and
# every citation to them is reported as unknown.
ARTICLE_STATUTES = {"CONST": "article", "QSO": "article"}


def read_corpus(path: str | Path) -> tuple[list[dict], dict]:
    """Provisions, and the corpus's own description if its first entry is one.

    A description is an object with the single key ``_corpus``: when the corpus was last
    brought up to date (``as_at``), its sources and their terms. It is what lets an
    answer about a later date say that a later amendment would not be in it.
    """
    rows = read_rows(path, keep_meta=True)
    meta: dict = {}
    if rows and isinstance(rows[0], dict) and set(rows[0]) == {META_KEY}:
        meta = rows.pop(0)[META_KEY]
        if not isinstance(meta, dict):
            raise CorpusError(f"{path}: {META_KEY} must be an object")
    if any(isinstance(r, dict) and META_KEY in r for r in rows):
        raise CorpusError(f"{path}: {META_KEY} may appear only as the first entry")
    return rows, meta


def read_rows(path: str | Path, *, keep_meta: bool = False) -> list[dict]:
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
    if not keep_meta and rows and isinstance(rows[0], dict) and set(rows[0]) == {META_KEY}:
        rows = rows[1:]
    return rows


def build_checked(rows: list[dict], *, source: str = "corpus", meta: dict | None = None) -> Corpus:
    """`corpus.build`, with the row at fault named instead of a bare TypeError."""
    meta = meta or {}
    try:
        as_at = dt.date.fromisoformat(meta["as_at"]) if meta.get("as_at") else None
    except (TypeError, ValueError) as exc:
        raise CorpusError(f"{source}: {META_KEY}.as_at is not a date: {exc}") from exc
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise CorpusError(f"{source}: provision {index} is not an object")
        unknown = set(row) - _FIELDS
        if unknown:
            raise CorpusError(f"{source}: provision {index} has unknown fields {sorted(unknown)}")
        missing = _REQUIRED - set(row)
        if missing:
            raise CorpusError(f"{source}: provision {index} is missing {sorted(missing)}")
        if ARTICLE_STATUTES.get(row["statute"]) == "article" and row["unit"] == "section":
            raise CorpusError(
                f"{source}: provision {index} is {row['statute']} section {row['number']}, but "
                f"{row['statute']} is numbered in articles, so citations to it would never be "
                'found; set "unit": "article" (paklaw-corpus import now does this itself)'
            )
    corpus = Corpus(as_at=as_at, meta=meta)
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


def write_jsonl(rows: list[dict], path: str | Path, *, meta: dict | None = None) -> None:
    """Atomically: a failed write must not leave half a statute book behind."""
    path = Path(path)
    ordered = sorted(
        rows,
        key=lambda r: (r["statute"], r["unit"], provision_order(r["number"]), r["in_force_from"]),
    )
    handle, temporary = tempfile.mkstemp(dir=path.parent or ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as out:
            if meta:
                out.write(json.dumps({META_KEY: meta}, ensure_ascii=False) + "\n")
            for row in ordered:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


# ---- command line ------------------------------------------------------------------------


def _read_source(path: Path, source: str) -> str:
    from .sources import pakistan_code, pakistani_org, strip_stars

    if source == "pakistani-org":
        raw = path.read_bytes()
        # Pages on the site are UTF-8 or ISO-8859-1 depending on their age.
        try:
            page = raw.decode("utf-8")
        except UnicodeDecodeError:
            page = raw.decode("latin-1")
        return pakistani_org(page)[0]
    text = path.read_text(encoding="utf-8-sig")
    if source == "pakistan-code":
        return strip_stars(pakistan_code(text).text)
    return text


def summarise(report: dict) -> str:
    """One paragraph a person reads before trusting an import."""
    lost = sum(span["characters"] for span in report["not_imported"])
    lines = [
        f"imported {report['provisions']} provision(s): {report['imported_characters']:,} of "
        f"{report['characters']:,} characters; {lost:,} characters in "
        f"{len(report['not_imported'])} span(s) were not imported (listed in the report)"
    ]
    for key, what in (
        ("omitted_or_repealed", "omitted or repealed, dates to record by hand"),
        ("missing_from_contents", "listed in the contents but not found in the text"),
        ("rejected_headings", "heading-like lines kept as body text"),
        ("too_long", "provision(s) suspiciously long: check for a missed heading"),
    ):
        if report.get(key):
            lines.append(f"  {len(report[key])} {what}")
    return "\n".join(lines)


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
    imp.add_argument("text_file", help="the Act: plain text (UTF-8), or a source named by --source")
    imp.add_argument(
        "--source",
        default="text",
        choices=["text", "pakistan-code", "pakistani-org"],
        help="text: plain text as is; pakistan-code: `pdftotext -layout` output of a "
        "pakistancode.gov.pk PDF (page footers, footnotes and amendment markers removed); "
        "pakistani-org: a saved pakistani.org statute page (HTML)",
    )
    imp.add_argument("--statute", required=True, help="PPC, PECA, CrPC, ... or a full name")
    imp.add_argument("--in-force-from", required=True, help="commencement date, YYYY-MM-DD")
    imp.add_argument(
        "--unit",
        choices=["section", "article"],
        help="default: article for the Constitution and the Qanun-e-Shahadat Order, "
        "section for everything else",
    )
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
    # Windows consoles default to the ANSI code page; headings carry em dashes.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    def fail(message: str) -> int:
        print(f"paklaw-corpus: {message}", file=sys.stderr)
        return 2

    try:
        meta: dict = {}
        if args.command == "check":
            rows, meta = read_corpus(args.corpus)
            problems = build_checked(rows, source=args.corpus, meta=meta).validate()
            for problem in problems:
                print(problem)
            print(f"{len(rows)} versions, {len(problems)} problem(s)")
            return 1 if problems else 0

        if args.command == "import":
            statute = normalise_statute(args.statute) or args.statute.strip().upper()
            unit = args.unit or ARTICLE_STATUTES.get(statute, "section")
            text = _read_source(Path(args.text_file), args.source)
            new, report = split_act(
                text, statute=statute, in_force_from=args.in_force_from, unit=unit
            )
            if not new:
                hint = (
                    " If this is pdftotext output of a pakistancode.gov.pk PDF, add "
                    "--source pakistan-code; for a saved pakistani.org page, "
                    "--source pakistani-org."
                    if args.source == "text"
                    else ""
                )
                return fail(
                    "no provisions found; is the text laid out as '1. Heading.- ...'?" + hint
                )
            output = Path(args.output)
            existing, meta = read_corpus(output) if output.exists() else ([], {})
            clash = {(r["statute"], r["unit"], r["number"]) for r in existing} & {
                (r["statute"], r["unit"], r["number"]) for r in new
            }
            if clash:
                return fail(f"{len(clash)} provision(s) of {statute} already in {output}")
            rows = existing + new
            print(json.dumps(report, indent=2, ensure_ascii=False))
            print(summarise(report), file=sys.stderr)
        else:
            rows, meta = read_corpus(args.corpus)
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
        write_jsonl(rows, output, meta=meta)
        print(f"wrote {len(rows)} versions to {output}", file=sys.stderr)
        return 0
    except (OSError, ValueError, CorpusError) as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
