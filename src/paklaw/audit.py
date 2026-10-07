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

from dataclasses import dataclass

from .citation import (
    Citation,
    is_foreign_report,
    locate,
    resolve_bare,
    statute_aliases,
)
from .corpus import Corpus, Provision

# Statuses, in the order a reader should worry about them.
REPEALED = "not_in_force"  # existed, ceased before the date
NOT_YET = "not_yet_in_force"  # exists, commenced after the date
UNKNOWN = "unknown_provision"  # statute loaded, provision absent
NO_ACT = "no_act_named"  # "section 9" — of what?
NOT_LOADED = "statute_not_loaded"  # the corpus cannot say anything either way
NOT_CHECKED = "not_checkable"  # case law and SROs: not what this corpus holds
# An Act was named and is not one this corpus knows - "the Indian Penal Code". Its own
# status, because reading it as "no Act named" is how a foreign provision ends up
# attributed to a Pakistani one with the same number.
FOREIGN_ACT = "act_not_recognised"
# Dated before the corpus's record of the statute begins: it may well have been in force.
BEFORE_RECORD = "before_record"
IN_FORCE = "in_force"
# Good law on the date, but amended since. The citation is right; the words quoted from
# it may be the later ones. A draft about 2019 conduct quoting "up to five years" cites a
# valid section and states the 2022 penalty.
AMENDED_SINCE = "amended_since"

PROBLEMS = (REPEALED, NOT_YET, UNKNOWN, NO_ACT, FOREIGN_ACT)
REVIEW = (AMENDED_SINCE,)
UNVERIFIED = (NOT_LOADED, NOT_CHECKED, BEFORE_RECORD)

#: Every status a citation can carry. Declared here and read by the MCP schema, so the
#: two cannot drift: the schema said `{"type": "string"}` and the README listed nine of
#: these, which is how `act_not_recognised` - the one the README calls the most serious -
#: went undocumented with a validator running over every tool's output on every path.
ALL_STATUSES = (IN_FORCE, *PROBLEMS, *REVIEW, *UNVERIFIED)


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
        if default_statute:
            # One implementation of "apply a default statute", in citation.py, which
            # also refuses to apply one over an Act the text named. Reimplementing the
            # rule here is how the two drifted: the audit learned to refuse a foreign
            # Act and the exported helper did not.
            (citation,) = resolve_bare([citation], default_statute=default_statute)
        # Where it is, so a reader or an editor can go straight to it.
        results.append(_check_one(corpus, citation, date, loaded) | {"offset": offset})

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    # Every occurrence is reported, with its offset, because an editor has to find each
    # one. But a draft that cites one repealed section in five places has one problem to
    # fix, not five, and a count that cannot tell those apart overstates the damage.
    distinct_problems = len({r["citation"] for r in results if r["status"] in PROBLEMS})

    problems = [r for r in results if r["status"] in PROBLEMS]
    review = [r for r in results if r["status"] in REVIEW]
    unverified = [r for r in results if r["status"] in UNVERIFIED]
    return {
        "as_of": date.isoformat(),
        "citations": results,
        "counts": counts,
        "problems": len(problems),
        "distinct_problems": distinct_problems,
        "distinct_citations": len({r["citation"] for r in results}),
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
        if c.kind == "reported" and is_foreign_report(c.report):
            # Said out loud, for the same reason a foreign Act is: a reader checking a
            # draft needs to know a cited authority is from another jurisdiction, and
            # "not held in this corpus" invites them to assume it would be if loaded.
            return entry | {
                "status": NOT_CHECKED,
                "note": f"{c.report} is a foreign law report; this corpus holds Pakistani "
                "statutes only and cannot speak to it either way",
            }
        what = "case law" if c.kind == "reported" else "subordinate legislation"
        return entry | {"status": NOT_CHECKED, "note": f"{what} is not held in this corpus"}
    if not c.statute:
        if c.named_statute:
            return entry | {
                "status": FOREIGN_ACT,
                "note": f"the text names {c.named_statute!r}, which is not an Act this "
                "corpus knows; it was not read as a Pakistani provision of the same number",
            }
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


#: Sentence endings, which is how the text is cut before any word is compared. A
#: provision has tens of sentences, so matching them is free whatever the words look
#: like.
_SENTENCE = re.compile(r"(?<=[.;:])\s+")

#: The most pairwise word comparisons this will spend on PRECISE diffing across one
#: whole call. Precise is `autojunk=False`, which keeps every word in play - the
#: heuristic it disables drops any element in more than 1% of a long sequence, which in
#: a statute is "shall", "the" and "section", the words a legal diff has to line up on.
#:
#: Measured, on word lists, with the shape named because a figure without one cannot be
#: reproduced:
#:
#:      words   alternating same/unique   statute prose, 1 word in 50 changed
#:        500        1.25s                      0.03s
#:      1,000        8.67s                      0.24s
#:      2,000       84.59s                      2.17s
#:      6,000           -                      58.89s
#:
#: Far worse than quadratic, and the cost is in the content rather than the length: the
#: same 1,000 words are 8.67s or 0.24s depending on how they alternate. So the budget is
#: in comparisons - the product of the two lengths - and not in words, which is what a
#: length cap got wrong. 250,000 of them is about 1.3s at the worst shape above.
#:
#: ONE budget for the whole call. Per passage it was no bound at all: total work was the
#: cap times the number of passages.
MAX_DIFF_WORK = 250_000

#: What a passage over the remaining budget gets instead: `autojunk=True`, which
#: compares the pathological 100,000-word case in 0.115s. Coarser - it reports a longer
#: run as one change - and a true statement about what differs, which a refusal to
#: compare is not. There is no "not compared" outcome.
COARSE_NOTE = (
    "compared with the fast heuristic rather than word by word: this passage is "
    "{a:,} and {b:,} words and the precise comparison is far worse than quadratic. "
    "Runs of change are reported together rather than individually; nothing here is "
    "wrong, only coarse. Read both texts with `get_provision` at each date."
)

#: The most changed passages one diff reports. A reply is a protocol message, and
#: 1,460,000 characters of corpus text produced 5.8 MB in one of them.
MAX_DIFF_CHANGES = 500

#: And the same bound in characters, because 500 rows of a 20,000-word passage each is
#: still megabytes. The row count alone let a 10 MB reply through.
MAX_DIFF_CHARS = 200_000

#: The most characters of either side of ONE row. A row count cannot bound a reply
#: whose rows are unbounded, and "all of this was replaced by all of that" is a single
#: row carrying both texts entire - 4.38 MB of it, measured, on an input of 1.4 MB.
#: A diff says which words changed, and 2,000 characters of one change is already past
#: what anyone reads; the row states the full length and `get_provision` serves it.
MAX_ROW_CHARS = 2_000

#: Every value `word_diff` can put in a row's `change` field. Exported because the MCP
#: tool's output schema declares this as an enum, and declared it as the three ordinary
#: ones while this module emitted two more - so the validator that checks every tool's
#: output on every path could not have passed a bounded diff. The same defect the
#: refusal statuses had, for the same reason: a list of values written out by hand in
#: the file that does not define them.
ALL_CHANGE_KINDS = ("added", "removed", "replaced", "approximate", "truncated")


def word_diff(before: str, after: str) -> list[dict]:
    """Changed passages only, with their surrounding words cut away.

    Word-level rather than character-level: "three years" -> "five years" is the fact a
    lawyer needs, and a character diff renders it as "thr" -> "fiv".

    Bounded on three axes, and each bound reports itself rather than truncating quietly -
    a diff that stopped early and does not say so is worse than no diff. Precise
    comparison draws on one `MAX_DIFF_WORK` budget for the whole call and passages past
    it are compared coarsely, with a row saying so; the row and character counts stop
    the reply growing without limit, with a row saying how many changes were not listed.

    Nothing is ever left uncompared.
    """
    budget = _Work(MAX_DIFF_WORK)
    changes: list[dict] = []
    characters = 0
    passages = _changed_passages(before, after)
    for position, (before_part, after_part) in enumerate(passages):
        if _full(changes, characters):
            changes.append(_truncated(len(passages) - position))
            return changes
        for row in _word_level(before_part, after_part, budget):
            if _full(changes, characters):
                changes.append(_truncated(len(passages) - position))
                return changes
            row["before"] = _abridge(row["before"])
            row["after"] = _abridge(row["after"])
            changes.append(row)
            characters += len(row["before"] or "") + len(row["after"] or "")
    return changes


def _full(changes: list[dict], characters: int) -> bool:
    return len(changes) >= MAX_DIFF_CHANGES or characters >= MAX_DIFF_CHARS


def _truncated(remaining: int) -> dict:
    return {
        "change": "truncated",
        "before": None,
        "after": (
            f"{remaining:,} further changed passage(s) are not listed: a diff is capped "
            f"at {MAX_DIFF_CHANGES:,} changes and {MAX_DIFF_CHARS:,} characters. These "
            "two versions have little text in common."
        ),
    }


def _abridge(text: str | None) -> str | None:
    """One side of one row, cut to `MAX_ROW_CHARS` and saying so if it was cut."""
    if text is None or len(text) <= MAX_ROW_CHARS:
        return text
    return (
        text[:MAX_ROW_CHARS].rstrip()
        + f" ... [abridged: {len(text):,} characters in all. Read the full text with "
        "`get_provision` at each date.]"
    )


@dataclass
class _Work:
    """The precise-comparison budget, drawn down across one call.

    A passage asks for what it would cost; it gets precision if the budget covers it and
    a coarse comparison otherwise. Held in an object rather than returned, because the
    point is that the passages share it.
    """

    remaining: int

    def afford(self, cost: int) -> bool:
        if cost > self.remaining:
            return False
        self.remaining -= cost
        return True


#: How much of two sentences must be shared vocabulary before they are treated as the
#: same sentence amended, rather than two different sentences that happen to sit at the
#: same index. An amended sentence keeps nearly all of its words ("three years" -> "five
#: years" is 0.9); the sentences either side of a struck clause share little but "the"
#: (0.2 in the case that found this). `quick_ratio` counts shared words without
#: aligning them, which is all this decision needs and is linear.
_PAIR_RATIO = 0.5


def _corresponds(before: str, after: str) -> bool:
    """Whether these two sentences are one sentence amended."""
    return (
        difflib.SequenceMatcher(a=before.split(), b=after.split()).quick_ratio()
        >= _PAIR_RATIO
    )


#: A run of unchanged sentences is trusted to mark a correspondence between the two
#: texts only if it carries at least this many words. A statute numbers its clauses, and
#: `C.` or `(3)` survives a renumbering exactly because it is a numbering - so the
#: matcher pinned those as equal and lined up the shifted clauses either side of them
#: against each other, reporting a sentence present in both texts as replaced. One word
#: is not evidence of anything; a sentence of real text is.
_ANCHOR_WORDS = 4


def _changed_regions(a: list[str], b: list[str]) -> list[tuple[list[str], list[str]]]:
    """The runs of sentences that differ, as (before, after) lists of sentences.

    Changes separated by nothing but clause numbers are one region: see `_ANCHOR_WORDS`.
    A substantial unchanged run does split them, which is what keeps the word comparison
    below working on short inputs.
    """
    regions: list[tuple[list[str], list[str]]] = []
    pending_a: list[str] = []
    pending_b: list[str] = []

    def flush() -> None:
        if pending_a or pending_b:
            regions.append((list(pending_a), list(pending_b)))
        pending_a.clear()
        pending_b.clear()

    # autojunk ON here, and OFF in `_word_level` below. The two passes want opposite
    # things from it: this one is only locating the regions that differ, so treating a
    # sentence that repeats throughout the text as uninteresting costs nothing and is
    # exactly what keeps it fast - the alternation that made the word pass cubic sits in
    # the 1%-popular elements autojunk drops. The fine pass is where accuracy matters
    # and where the input is a passage, so it keeps every word in play.
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if op != "equal":
            pending_a.extend(a[i1:i2])
            pending_b.extend(b[j1:j2])
            continue
        run = a[i1:i2]
        if sum(len(sentence.split()) for sentence in run) >= _ANCHOR_WORDS:
            flush()
        elif pending_a or pending_b:
            # Not an anchor, and a change is open: the run belongs to it. It is equal
            # text on both sides, so the word comparison will report it as equal.
            pending_a.extend(run)
            pending_b.extend(run)
    flush()
    return regions


def _changed_passages(before: str, after: str) -> list[tuple[str, str]]:
    """(before, after) of each run of sentences that differs. Equal runs are dropped.

    The cheap half. A provision has tens of sentences, so this comparison costs nothing
    whatever the words inside them look like - which is the whole point, because the
    word comparison below is far worse than quadratic on the text an amending Act
    produces.
    """
    a = [s for s in _SENTENCE.split(before) if s.strip()]
    b = [s for s in _SENTENCE.split(after) if s.strip()]
    out: list[tuple[str, str]] = []
    for region_a, region_b in _changed_regions(a, b):
        if (
            len(region_a) > 1
            and len(region_a) == len(region_b)
            and all(_corresponds(x, y) for x, y in zip(region_a, region_b, strict=True))
        ):
            # The same number of sentences, each one recognisably the other amended -
            # which is what an amending Act does. Compared one for one, so each word
            # comparison sees one sentence instead of the whole region: joining the
            # region made a 13,000-character provision of repeated sentences exceed
            # the precise-comparison budget and be diffed coarsely, which is a worse
            # answer than a slow one.
            #
            # `_corresponds`, because equal counts are not correspondence. Striking one
            # clause and renumbering the rest also produces a region of equal length,
            # offset by one.
            out.extend(zip(region_a, region_b, strict=True))
            continue
        out.append((" ".join(region_a), " ".join(region_b)))
    return out


def _word_level(before: str, after: str, budget: _Work) -> list[dict]:
    """The word diff of one changed passage.

    Precise if the budget covers it, coarse if not, and never skipped.
    """
    a, b = _WORD.findall(before), _WORD.findall(after)

    # The common prefix and suffix, which are exact, linear and most of an amendment:
    # "three years" -> "five years" inside a 5,000-word provision leaves a core of two
    # words, and the comparison below is the one whose cost is super-quadratic in what
    # it is handed. Stripping them first is the difference between a precise diff and a
    # coarse one for every realistic amendment.
    head = 0
    while head < min(len(a), len(b)) and a[head] == b[head]:
        head += 1
    tail = 0
    while tail < min(len(a), len(b)) - head and a[len(a) - 1 - tail] == b[len(b) - 1 - tail]:
        tail += 1
    core_a = a[head : len(a) - tail]
    core_b = b[head : len(b) - tail]
    if not core_a and not core_b:
        return []

    precise = budget.afford(len(core_a) * len(core_b))
    changes: list[dict] = []
    if not precise:
        changes.append(
            {
                "change": "approximate",
                "before": None,
                "after": COARSE_NOTE.format(a=len(core_a), b=len(core_b)),
            }
        )
    matcher = difflib.SequenceMatcher(a=core_a, b=core_b, autojunk=not precise)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        changes.append(
            {
                "change": {"replace": "replaced", "delete": "removed", "insert": "added"}[op],
                "before": " ".join(core_a[i1:i2]) or None,
                "after": " ".join(core_b[j1:j2]) or None,
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
        return {
            "error": "the cited provision is not in this corpus",
            "refusal_status": "unknown_provision",
        }

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
        # The condition is "or" and the message used to say "both", so the common and
        # interesting case - it did not exist then and does now - was summarised as
        # though it had never been in force at all. `summary` is the field a model
        # paraphrases to a reader, and a wrong statement about whether a provision was
        # in force on a date is the harm this repository is built around.
        if v1 is None and v2 is None:
            result["summary"] = "in force on neither date; nothing to compare"
        else:
            absent, present = (d1, d2) if v1 is None else (d2, d1)
            result["summary"] = (
                f"not in force on {absent.isoformat()}, in force on "
                f"{present.isoformat()}; there is no earlier text to diff against"
            )
        # Present on every path, so a client reads one shape rather than branching.
        result["changes"] = []
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


def sequence_within(number: str) -> tuple | None:
    """The host's own position in the numbering that a buried heading would continue.

    `provision_order` exists to SORT unlike numbers, so it tags them by scheme and a
    section always sorts before an Order. That tag is meaningless as a comparison, and
    two detectors in `split.py` were using it as one: a buried heading is always a
    digit-and-letter number, every one of those is tag 0, and tag 0 is below everything,
    so inside a provision numbered `IX/3` or `Schedule` every candidate was skipped and
    the safeguard was blind.

    What they need is the number a buried `4.` would be later than:

      * `302A` -> `(302, "A")`. A section's own number.
      * `IX/3` -> `(3, "")`. Order IX rule 3: the next rule is rule 4, and the Roman
        numeral is which Order, not which rule.
      * `Schedule` -> None. Nothing to be later than, so the ascending-run rule stands
        on its own - any run of increasing numbers each followed by a capitalised title.
    """
    order = provision_order(number)
    if order[0] == 0:
        return (order[1], order[2])
    if order[0] == 1:
        return (order[2], "")
    return None


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
