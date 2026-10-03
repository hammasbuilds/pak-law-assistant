"""Questions asked *about* the statute book rather than answered from it.

  **check_citations**    every citation in a draft, checked against the law on a date
  **compare_versions**   what an amendment actually changed, word by word
  **contents**           the provisions of one statute in force on a date
  **changes_between**    every commencement, substitution and repeal in a period

The first is the one that matters most. A draft written by a person or a model cites
provisions from memory, and memory does not know about the 2022 substitution. Checking
each citation against a temporal corpus is cheap, mechanical and catches the error that
reads as correct — so the check never guesses: a citation it cannot verify is reported
as unverifiable, not as fine.
"""

from __future__ import annotations

import datetime as dt
import difflib
import re

from .citation import Citation, locate, statute_aliases
from .corpus import Corpus, Provision

# Statuses, in the order a reader should worry about them.
REPEALED = "not_in_force"  # existed, ceased before the date
NOT_YET = "not_yet_in_force"  # exists, commenced after the date
UNKNOWN = "unknown_provision"  # statute loaded, provision absent
NO_ACT = "no_act_named"  # "section 9" — of what?
NOT_LOADED = "statute_not_loaded"  # the corpus cannot say anything either way
NOT_CHECKED = "not_checkable"  # case law and SROs: not what this corpus holds
# Dated before the corpus's record of the statute begins: it may well have been in force.
BEFORE_RECORD = "before_record"
IN_FORCE = "in_force"
# Good law on the date, but amended since. The citation is right; the words quoted from
# it may be the later ones. A draft about 2019 conduct quoting "up to five years" cites a
# valid section and states the 2022 penalty.
AMENDED_SINCE = "amended_since"

PROBLEMS = (REPEALED, NOT_YET, UNKNOWN, NO_ACT)
REVIEW = (AMENDED_SINCE,)
UNVERIFIED = (NOT_LOADED, NOT_CHECKED, BEFORE_RECORD)


def _date(value: str | dt.date) -> dt.date:
    return dt.date.fromisoformat(value) if isinstance(value, str) else value


def _nearest(versions: list[Provision], date: dt.date) -> Provision:
    """The version a reader asking about `date` most needs described."""
    return next((v for v in reversed(versions) if v.in_force_from <= date), versions[0])


def check_citations(
    corpus: Corpus,
    text: str,
    *,
    as_of: str | dt.date,
    default_statute: str | None = None,
) -> dict:
    """Every citation in `text`, with whether it was good law on `as_of`."""
    date = _date(as_of)
    loaded = {p.statute for p in corpus}
    results: list[dict] = []

    for offset, citation in locate(text, statutes=statute_aliases(loaded)):
        if default_statute and citation.kind == "statutory" and not citation.statute:
            citation = Citation(**{**citation.__dict__, "statute": default_statute})
        # Where it is, so a reader or an editor can go straight to it.
        results.append(_check_one(corpus, citation, date, loaded) | {"offset": offset})

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    problems = [r for r in results if r["status"] in PROBLEMS]
    review = [r for r in results if r["status"] in REVIEW]
    unverified = [r for r in results if r["status"] in UNVERIFIED]
    return {
        "as_of": date.isoformat(),
        "citations": results,
        "counts": counts,
        "problems": len(problems),
        "review": len(review),
        "unverified": len(unverified),
        # Clean only when every citation was checked and passed. Nothing checked is not
        # the same as nothing wrong.
        "verdict": (
            "problems"
            if problems
            else "review"
            if review
            else "unverified"
            if unverified or not results
            else "all_in_force"
        ),
    }


def _check_one(corpus: Corpus, c: Citation, date: dt.date, loaded: set[str]) -> dict:
    entry = {"citation": c.pretty(), "raw": c.raw, "kind": c.kind}

    if c.kind != "statutory":
        what = "case law" if c.kind == "reported" else "subordinate legislation"
        return entry | {"status": NOT_CHECKED, "note": f"{what} is not held in this corpus"}
    if not c.statute:
        return entry | {
            "status": NO_ACT,
            "note": "no Act named; the provision cannot be checked until one is",
        }
    if c.statute not in loaded:
        return entry | {
            "status": NOT_LOADED,
            "note": f"{c.statute} is not in this corpus, so this citation is unverified",
        }

    cited_key = c.key
    key = corpus.resolve(cited_key)
    versions = corpus.versions(key)
    if not versions:
        return entry | {"status": UNKNOWN, "note": f"{c.statute} has no such provision here"}

    if key != cited_key:
        # Only the section is checkable; whether subsection (9) exists is not recorded.
        entry["checked_as"] = versions[0].citation().pretty()
        entry["subdivision_checked"] = False

    live = corpus.version_on(key, date)
    if live is None:
        nearest = _nearest(versions, date)
        if date < versions[0].in_force_from:
            status = NOT_YET if versions[0].start_known else BEFORE_RECORD
        else:
            status = REPEALED
        entry |= {"status": status, "note": nearest.status_note(date)}
        current = corpus.current(key)
        if current is not None and status == REPEALED:
            # Repealed as at the date, but a later version exists: the citation may be
            # right about today and wrong about the date asked.
            entry["note"] += f"; a later version has been in force since {current.in_force_from}"
        entry["history"] = corpus.history(key)
        return entry

    entry |= {"status": IN_FORCE, "heading": live.heading}
    later = [v for v in versions if v.in_force_from > date]
    if later:
        entry["status"] = AMENDED_SINCE
        entry["note"] = (
            f"in force on {date}, but amended with effect from {later[0].in_force_from}"
            " — check the draft quotes the words in force on its date"
        )
    elif live.in_force_to:
        entry["note"] = f"in force on {date}; ceased {live.in_force_to}"
    return entry


# ---- what changed ------------------------------------------------------------------------

_WORD = re.compile(r"\S+")


def word_diff(before: str, after: str) -> list[dict]:
    """Changed passages only, with their surrounding words cut away.

    Word-level rather than character-level: "three years" → "five years" is the fact a
    lawyer needs, and a character diff renders it as "thr" → "fiv".
    """
    a, b = _WORD.findall(before), _WORD.findall(after)
    changes = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        changes.append(
            {
                "change": {"replace": "replaced", "delete": "removed", "insert": "added"}[op],
                "before": " ".join(a[i1:i2]) or None,
                "after": " ".join(b[j1:j2]) or None,
            }
        )
    return changes


def compare_versions(
    corpus: Corpus, key: str, *, before: str | dt.date, after: str | dt.date
) -> dict:
    """The provision as it stood on two dates, and exactly what differs."""
    key = corpus.resolve(key)
    versions = corpus.versions(key)
    if not versions:
        return {"error": "the cited provision is not in this corpus"}

    d1, d2 = _date(before), _date(after)
    if d1 > d2:
        d1, d2 = d2, d1
    v1, v2 = corpus.version_on(key, d1), corpus.version_on(key, d2)
    citation = versions[0].citation().pretty()

    def describe(version: Provision | None, date: dt.date) -> dict:
        if version is None:
            return {
                "date": date.isoformat(),
                "in_force": False,
                "status": _nearest(versions, date).status_note(date),
            }
        return {
            "date": date.isoformat(),
            "in_force": True,
            "in_force_from": version.in_force_from.isoformat(),
            "in_force_to": version.in_force_to.isoformat() if version.in_force_to else None,
            "text": version.text,
        }

    result = {"citation": citation, "before": describe(v1, d1), "after": describe(v2, d2)}
    if v1 is None or v2 is None:
        result["summary"] = "not in force on both dates; nothing to compare"
        return result
    if v1 is v2:
        result["summary"] = "the same version was in force on both dates"
        result["changes"] = []
        return result

    between = [v for v in versions if d1 < v.in_force_from <= d2]
    # Who made the change is recorded on the version that ceased.
    result["amended_by"] = [
        v.amended_by for v in versions if v.in_force_to and d1 < v.in_force_to <= d2
    ]
    result["versions_between"] = len(between)
    result["changes"] = word_diff(v1.text, v2.text)
    if v1.heading != v2.heading:
        result["heading_changed"] = {"before": v1.heading, "after": v2.heading}
    result["summary"] = (
        f"{len(result['changes'])} change(s) across {len(between)} amendment(s)"
        if result["changes"]
        else "text identical; only the version record differs"
    )
    return result


# ---- browsing ----------------------------------------------------------------------------

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman(numeral: str) -> int:
    total = 0
    for current, following in zip(numeral, numeral[1:] + " ", strict=False):
        value = _ROMAN.get(current, 0)
        total += -value if _ROMAN.get(following, 0) > value else value
    return total


def provision_order(number: str) -> tuple:
    """Statute order, not string order: 2 < 10 < 10A < 11, and Order IX before Order X."""
    match = re.match(r"(\d+)(.*)", number)
    if match:
        return (0, int(match.group(1)), match.group(2))
    match = re.match(r"([IVXLCDM]+)/(\d+)", number)
    if match:
        return (1, _roman(match.group(1)), int(match.group(2)))
    return (2, 0, number)


def contents(corpus: Corpus, statute: str, *, as_of: str | dt.date) -> dict:
    """The table of contents of one statute as it stood on a date."""
    date = _date(as_of)
    live = sorted(
        (p for p in corpus.as_of(date) if p.statute == statute),
        key=lambda p: (p.unit, provision_order(p.number)),
    )
    everything = {p.key for p in corpus if p.statute == statute}
    return {
        "statute": statute,
        "as_of": date.isoformat(),
        "in_force": len(live),
        "not_in_force_on_this_date": len(everything) - len({p.key for p in live}),
        "provisions": [
            {
                "citation": p.citation().pretty(),
                "heading": p.heading,
                "chapter": p.chapter or None,
                "in_force_from": p.in_force_from.isoformat(),
                "amended": len(corpus.versions(p.key)) > 1,
            }
            for p in live
        ],
    }


def changes_between(
    corpus: Corpus,
    start: str | dt.date,
    end: str | dt.date,
    *,
    statute: str | None = None,
) -> dict:
    """Every commencement and cessation in [start, end], oldest first.

    A substitution is one event, not a repeal plus an unrelated commencement: the
    version that ceased and the one that began on the same day are reported together.
    """
    d1, d2 = sorted((_date(start), _date(end)))
    events: list[dict] = []
    for key in sorted({p.key for p in corpus if not statute or p.statute == statute}):
        versions = corpus.versions(key)
        citation = versions[0].citation().pretty()
        for index, v in enumerate(versions):
            if v.in_force_to and d1 <= v.in_force_to <= d2:
                successor = next(
                    (n for n in versions[index + 1 :] if n.in_force_from == v.in_force_to), None
                )
                events.append(
                    {
                        "date": v.in_force_to.isoformat(),
                        "citation": citation,
                        "event": v.manner or ("substituted" if successor else "repealed"),
                        "by": v.amended_by or None,
                        "heading": v.heading,
                    }
                )
            first = index == 0
            replaces = not first and versions[index - 1].in_force_to == v.in_force_from
            if d1 <= v.in_force_from <= d2 and not replaces:
                events.append(
                    {
                        "date": v.in_force_from.isoformat(),
                        "citation": citation,
                        "event": "commenced" if first else "reinstated",
                        "by": v.enacted_by or None,
                        "heading": v.heading,
                    }
                )
    events.sort(key=lambda e: (e["date"], e["citation"]))
    return {
        "from": d1.isoformat(),
        "to": d2.isoformat(),
        "statute": statute,
        "events": events,
    }
