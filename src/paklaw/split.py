"""Splitting an Act's text into provisions, and accounting for every character.

Real statute text comes in more than one layout, and the importer has to survive all of
them without silently losing law:

    20. Offences against dignity of a natural person.—(1) Whoever ...   (one line, dash)
    52. "Good faith." Nothing is said to be done ...                    (Pakistan Code PDF)
    302.                                                                 (pakistani.org:
    Punishment of qatl-i-amd:                                             number alone, then
    Whoever commits qatl-e-amd shall ...                                  a colon heading)

**The number is the boundary, and the number is also the easiest thing to fake.** A
numbered clause inside a body ("3. Thirty days: ..."), a footnote reference glued to an
article number ("11. The Republic" for article 1 with footnote 1), a note quoting an
omitted section after the last one: each looks like a heading. Every line that *could*
open a provision is a candidate, and the provisions are the heaviest chain of candidates
whose numbers rise in statute order — a longest-increasing-subsequence, not a greedy
walk. Greedy acceptance lets one spurious high number reject every real heading after it
(the importer this replaced read the whole Penal Code as a single section 237).

**When the Act carries its own table of contents, the contents decide.** Pakistan Code
PDFs open with one. Its numbers are the only numbers accepted and its headings are the
headings, which also settles where a heading ends when the only separator is a full
stop ("Officers, etc. deemed to be public servants.").

**Nothing is dropped without a line in the report.** Text before the first provision,
the table of contents, chapter and group headings, schedules or notes after the last
provision, omitted provisions: each is counted in characters, so a report that says
"1 provision" for a 450 KB Act can no longer look like a success.
"""

from __future__ import annotations

import datetime as dt
import difflib
import re
from dataclasses import dataclass

from .audit import provision_order
from .citation import normalise_number

# A line that could open a provision: "20.", "302.", "2A.", "489-F.", optionally inside
# a marker bracket "[20." left by a source, followed by a space or the end of the line.
_CANDIDATE = re.compile(
    r"^[ \t]*(?P<glued>(?:\d{1,3}(?:,[ \t]*|[ \t]+(?=\d)))+)?(?P<open>[\[({])?"
    r"(?P<number>\d{1,5}(?:[- ]?[A-Z]{1,3})?)(?P<sep>[.,)])(?=[ \t]|$|_|[A-Z][a-z])[ \t]*"
    r"(?P<rest>.*)$",
    re.M,
)
# Where a heading ends. Strong: a dash, underscores, or a colon. Weak: a full stop.
_STRONG_END = re.compile(
    r"[ \t]*(?:[.:][ \t]*(?:_{1,}|[—–]+|-{1,2})|_{2,}|[—–]|:(?=[ \t]|$))[ \t]*"
)
_WEAK_END = re.compile(r"\.[”\"']?(?=[ \t]+\S|[ \t]*$)[ \t]*")
# What may follow a known heading before the body begins: a closing quote, a full stop
# or colon, and a dash of any of the kinds the sources use.
_AFTER_HEADING = re.compile(r"(?:[”\"'’.:][ \t]*)*(?:_+|[—–]+|-{1,2})?[ \t]*")
_CHAPTER = re.compile(
    r"^[ \t]*(?P<kind>CHAPTER|PART)[ \t.-]*(?P<id>[IVXLC\d]+(?:[ \t]?[A-Z](?![a-z]))?)\b"
    r"[ \t.:—–_-]*(?P<title>.*)$",
    re.M,
)
_CONTENTS = re.compile(r"^[ \t]*(?:TABLE OF )?CONTENTS[ \t]*$", re.M)
# After the last provision: schedules, forms, appendices, or a site's notes.
_TRAILER = re.compile(
    r"^[ \t]*(?:(?:THE[ \t]+)?(?:[A-Z]+[ \t]+)?SCHEDULE\b.*|APPENDIX\b.*|Notes|Footnotes|"
    r"Sources?[ \t]*::.*|FORMS?[ \t]*)$",
    re.M,
)
_DROPPED = re.compile(r"^\[?\s*(?:omitted|repealed|rep\.\s*by|deleted)\b", re.I)
# "56. [Sentence of Europeans ...] Rep. by the Criminal Law (...) Act, 1949": a heading in
# brackets followed by the instrument that removed it.
_REMOVED_BY = re.compile(r"^\[[^\]]{0,200}\]\.?\s*(?:rep\.|(?:repealed|omitted|deleted)\b)", re.I)
_LISTED_AS_GONE = re.compile(r"\b(?:omitted|repealed|deleted)\b", re.I)
_ONLY_MARKS = re.compile(r"^[\s*\[\].,;:_—–-]*$")
_SPACE = re.compile(r"\s+")

HEADING_WINDOW = 240
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-"})


def _number(raw: str) -> str:
    """'462 I' and '489-F' as the corpus writes them: '462I', '489F'."""
    return normalise_number(raw.replace(" ", ""))


def _same_heading(line: str, listed: str) -> bool:
    """Whether a line opens with the heading the contents list, allowing a misprint
    ("Power of District Magistrate" where the contents say "Power to")."""
    got, wanted = _fold(line)[:30], _fold(listed)[:30]
    if not wanted:
        return False
    if got.startswith(wanted[:12]):
        return True
    return difflib.SequenceMatcher(a=got[: len(wanted)], b=wanted).ratio() >= 0.85


def _fold(text: str) -> str:
    """Lower case, one space, straight quotes: for comparing a heading with its contents line."""
    return _SPACE.sub(" ", text.translate(_QUOTES)).strip().lower()


TOO_LONG = 20_000


@dataclass
class _Candidate:
    start: int  # offset of the line
    body_start: int  # offset just after the heading
    number: str
    heading: str
    weight: float
    order: tuple


def _heading(
    rest: str, following: str, tail: str, toc_heading: str | None
) -> tuple[str, int, float]:
    """(heading, characters of `tail` the heading occupies, weight).

    `rest` is the rest of the header line, `tail` the text from the same point on (a
    heading can wrap), and `following` the next non-empty line, for the layout where the
    number stands alone.
    """
    if toc_heading is not None:
        words = re.findall(r"\w+", toc_heading.translate(_QUOTES))
        if words:
            found = re.match(
                r"[\W_]*" + r"[\W_]+".join(map(re.escape, words)), tail.translate(_QUOTES), re.I
            )
            if found:
                # Measured against the known heading, so a full stop inside it ("etc.")
                # is not taken as its end.
                end = _AFTER_HEADING.match(tail, found.end())
                return toc_heading.strip().rstrip("."), end.end(), 3.0
    if not rest.strip():
        return _heading_below(tail)
    window = rest[:HEADING_WINDOW]
    strong = _STRONG_END.search(window)
    if strong and strong.start() > 0:
        return rest[: strong.start()].strip().rstrip("."), strong.end(), 2.0
    weak = _WEAK_END.search(window)
    if weak and weak.start() > 0 and (rest[:1].isupper() or rest[:1] in "\"'“‘"):
        return rest[: weak.start()].strip().rstrip("."), weak.end(), 1.0
    return rest.strip().rstrip("."), len(rest), 0.5


def _heading_below(tail: str) -> tuple[str, int, float]:
    """The heading of a provision whose number stands alone on its line.

    pakistani.org prints "457." and then the heading, which can wrap over two lines
    before its colon ("Lurking house-trespass ... offence / punishable with
    imprisonment:"). Up to three lines are joined until one ends a heading.
    """
    lines: list[str] = []
    position = 0
    for match in re.finditer(r"[^\n]*(?:\n|$)", tail):
        if not match.group(0):
            break
        line = match.group(0).strip()
        position = match.end()
        if not line:
            if lines:
                break
            continue
        if _CANDIDATE.match(line) or _CHAPTER.match(line):
            break
        lines.append(line)
        joined = " ".join(lines)
        if len(joined) > HEADING_WINDOW:
            break
        # A wrapped heading ends in a colon or dash; a full stop ends only a one-line one,
        # or the first sentence of the body would be read into the heading.
        if joined.endswith((":", "—", "-", "_")) or (len(lines) == 1 and joined.endswith(".")):
            heading = re.sub(r"[ \t]*[.:]?[ \t]*(?:[—–]+|-{1,2}|_+)?$", "", joined)
            return heading, position, 2.0
        if len(lines) == 3:
            break
    if lines and len(lines[0]) <= HEADING_WINDOW:
        # No line ended like a heading: take the first line alone, and say it is weak.
        first = re.search(r"[^\n]*\S[^\n]*(?:\n|$)", tail)
        return lines[0], first.end() if first else 0, 1.0
    return "", 0, 0.5


def _contents(text: str) -> tuple[dict[str, str], list[str], int, int]:
    """The Act's own table of contents: headings by number, their order, and its span."""
    found = _CONTENTS.search(text[: max(20_000, len(text) // 20)])
    if not found:
        return {}, [], 0, 0
    headings: dict[str, str] = {}
    order: list[str] = []
    end = found.end()
    first_heading = "\x00"
    for match in _CANDIDATE.finditer(text, found.end()):
        number = _number(match.group("number"))
        if order and (
            number == order[0]
            # the body restarts at the first provision, perhaps with a footnote number
            # glued to its own ("11. The Republic" for article 1, note 1)
            or (len(order) > 5 and _fold(match.group("rest")).startswith(first_heading))
        ):
            break
        if not order:
            first_heading = _fold(match.group("rest"))[:20] or "\x00"
        if number not in headings:
            headings[number] = match.group("rest").strip()
            order.append(number)
        end = match.end()
    if len(order) < 3:
        return {}, [], 0, 0
    return headings, order, found.start(), end


def _heaviest_rising_chain(candidates: list[_Candidate]) -> list[int]:
    """Indexes of the maximum-weight subsequence whose order keys strictly increase.

    O(n log n) with a Fenwick tree over order ranks, since a large Act has thousands of
    candidate lines.
    """
    ranks = {key: i + 1 for i, key in enumerate(sorted({c.order for c in candidates}))}
    size = len(ranks)
    tree_best = [0.0] * (size + 1)
    tree_at = [-1] * (size + 1)
    best = [0.0] * len(candidates)
    back = [-1] * len(candidates)

    def query(position: int) -> tuple[float, int]:
        value, index = 0.0, -1
        while position > 0:
            if tree_best[position] > value:
                value, index = tree_best[position], tree_at[position]
            position -= position & -position
        return value, index

    def update(position: int, value: float, index: int) -> None:
        while position <= size:
            if value > tree_best[position]:
                tree_best[position], tree_at[position] = value, index
            position += position & -position

    for i, candidate in enumerate(candidates):
        rank = ranks[candidate.order]
        previous, at = query(rank - 1)
        best[i], back[i] = previous + candidate.weight, at
        update(rank, best[i], i)

    if not candidates:
        return []
    i = max(range(len(candidates)), key=lambda k: best[k])
    chain = []
    while i != -1:
        chain.append(i)
        i = back[i]
    return chain[::-1]


def _next_line(text: str, position: int) -> tuple[str, int]:
    """The next non-empty line at or after `position`, and the offset after it."""
    while position < len(text):
        end = text.find("\n", position)
        end = len(text) if end == -1 else end
        line = text[position:end]
        if line.strip():
            return line, min(end + 1, len(text))
        position = end + 1
    return "", len(text)


def _cut_labels(body: str) -> tuple[str, list[tuple[str, str]], int]:
    """Remove chapter/part blocks from a body. Returns (body, chapters, characters cut)."""
    chapters: list[tuple[str, str]] = []
    cut = 0
    out: list[str] = []
    lines = body.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _CHAPTER.match(line)
        if not match:
            out.append(line)
            index += 1
            continue
        block = [line]
        title = match.group("title").strip()
        index += 1
        # The title sits on the following capitalised lines.
        while index < len(lines) and (
            not lines[index].strip()
            or (lines[index].strip().upper() == lines[index].strip() and len(block) < 4)
        ):
            if lines[index].strip():
                title = f"{title} {lines[index].strip()}".strip()
            block.append(lines[index])
            index += 1
        label = f"{match.group('kind').title()} {match.group('id').replace(' ', '')}"
        chapters.append((label, title))
        cut += len("\n".join(block).strip())
    return "\n".join(out), chapters, cut


def _trailing_heading(body: str) -> tuple[str, str]:
    """Split off a group heading that sits just before the next provision.

    "Of Kidnapping, Abduction, Slavery and Forced Labour" or "A.__ Classes of Criminal
    Courts" labels the provisions after it; left in place it becomes the last words of
    the provision before.
    """
    lines = body.rstrip().split("\n")
    if len(lines) < 3:
        return body, ""
    last = lines[-1].strip()
    if (
        not lines[-2].strip()  # set apart by a blank line, not the end of a paragraph
        and len(last.split()) <= 14
        and last
        and len(last) <= 100
        and last[0].isupper()
        and not last.endswith((".", ";", ":", ",", "—", "-", ")", "]", "__"))
        and not re.search(r"\d", last[:3])
    ):
        return "\n".join(lines[:-1]), last
    return body, ""


@dataclass
class _Layout:
    toc: dict[str, str]
    toc_order: list[str]
    toc_start: int
    toc_end: int
    candidates: list[_Candidate]
    chain: list[_Candidate]
    unlisted: list[str]


def provision_spans(text: str, *, include_repealed: bool = False) -> list[tuple[str, int, int]]:
    """(number, start, end) of each provision found in `text`, header included.

    For build scripts that need to know which provision an offset falls in, such as the
    provision an amendment marker belongs to.

    By default this agrees with `split_act`: a heading the source lists as `[Repealed]`
    carries no provision, so it is not reported. It used to be reported, and nothing
    noticed - the function was exported, had no caller and no test, and had quietly
    drifted one provision apart from the importer it has to agree with. On the real
    Penal Code fixture it returned 14 spans where `split_act` imported 13, the extra one
    being s.56 "[Repealed]". A caller attributing amendment markers by offset would have
    attributed some to a provision the importer never created.

    `include_repealed=True` asks for every heading in the layout, which is the right
    answer for "what does this offset sit under" and the wrong one for "which provision
    is this". The caller has to say which it means.
    """
    chain = _layout(text).chain
    spans = [
        (c.number, c.start, chain[i + 1].start if i + 1 < len(chain) else len(text))
        for i, c in enumerate(chain)
    ]
    if include_repealed:
        return spans
    kept = []
    for number, start, end in spans:
        segment = text[start:end]
        # The same test the importer applies, applied to the same text.
        # The importer tests these AFTER stripping the leading number, so the same
        # text has to be presented the same way or the two disagree. s.56 of the Penal
        # Code reads "56. [Sentence of Europeans and Americans to penal servitude.]
        # Rep. by the Criminal Law ... Act, 1949", and `Rep. by` is only at the start
        # once "56." is gone.
        rest = re.sub(r"^\s*\d+[A-Z]*\.\s*", "", segment.replace("\n", " ")).strip()
        if _REMOVED_BY.match(rest) or _DROPPED.match(rest):
            continue
        kept.append((number, start, end))
    return kept


def _layout(text: str) -> _Layout:
    toc, toc_order, toc_start, toc_end = _contents(text)
    toc_rank = {number: i for i, number in enumerate(toc_order)}

    # Numbers the contents do not list are placed just after the listed number before
    # them, so a heading the contents misprint ("362N" for 462N) still has a place.
    toc_by_position = sorted(toc_order, key=provision_order)

    def order_of(number: str) -> tuple:
        if number in toc_rank:
            return (toc_rank[number], 0)
        before = [n for n in toc_by_position if provision_order(n) < provision_order(number)]
        return (toc_rank[before[-1]] if before else -1, 1, provision_order(number))

    candidates: list[_Candidate] = []
    unlisted: list[str] = []
    for match in _CANDIDATE.finditer(text, toc_end):
        raw_number = _number(match.group("number"))
        rest = match.group("rest")
        following, _ = _next_line(text, match.end())
        # "(2E) Removal", "26A, Punishment" and "7, 8112. Order" (two footnote numbers
        # glued to section 112) are misprints a source really has; they are accepted only
        # where the contents confirm the heading.
        irregular = (
            bool(match.group("glued"))
            or match.group("sep") != "."
            or match.group("open") in ("(", "{")
            or len(match.group("number")) > 4
            and match.group("number")[:5].isdigit()
        )
        number = raw_number
        confirmed = False
        if toc:
            # A footnote number glued to the provision number ("11." for article 1
            # with footnote 1) is recognised by the heading that follows it.
            options = [raw_number] + [raw_number[k:] for k in (1, 2) if len(raw_number) > k]
            for option in options:
                if option in toc and _same_heading(rest or following, toc[option]):
                    number, confirmed = option, True
                    break
        if irregular and not confirmed:
            continue
        tail = text[match.start("rest") : match.start("rest") + 2 * HEADING_WINDOW]
        heading, consumed, weight = _heading(rest, following, tail, toc.get(number))
        body_start = match.start("rest") + consumed
        if toc and not confirmed:
            if number in toc:
                weight = min(weight, 0.5)
            elif weight >= 2.0:
                weight = 1.0  # a clear heading the contents do not list
                unlisted.append(number)
            else:
                continue
        order = order_of(number) if toc else provision_order(number)
        candidates.append(_Candidate(match.start(), body_start, number, heading, weight, order))

    chain = [candidates[i] for i in _heaviest_rising_chain(candidates)]
    return _Layout(toc, toc_order, toc_start, toc_end, candidates, chain, unlisted)


def split_act(
    text: str,
    *,
    statute: str,
    in_force_from: str,
    unit: str = "section",
    too_long: int = TOO_LONG,
) -> tuple[list[dict], dict]:
    """Rows for `corpus.build`, and a report of everything that needs a human look."""
    dt.date.fromisoformat(in_force_from)  # fail before doing any work
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    layout = _layout(text)
    toc, toc_order, toc_start, toc_end = (
        layout.toc,
        layout.toc_order,
        layout.toc_start,
        layout.toc_end,
    )
    candidates, chain, unlisted = layout.candidates, layout.chain, layout.unlisted
    accepted = {id(c) for c in chain}
    rejected = [
        f"'{text[c.start : c.start + 60].strip()}' looks like a heading but its number does "
        "not fit the sequence; kept as body text"
        for c in candidates
        if id(c) not in accepted and c.weight >= 1.0
    ]

    not_imported: list[dict] = []

    def note(what: str, span: str) -> None:
        stripped = span.strip()
        if stripped:
            not_imported.append(
                {
                    "what": what,
                    "characters": len(stripped),
                    "starts": _SPACE.sub(" ", stripped)[:80],
                }
            )

    if toc:
        note("table of contents", text[toc_start:toc_end])
    first = chain[0].start if chain else len(text)
    preamble, preamble_chapters, _ = _cut_labels(text[toc_end:first])
    note("text before the first provision", preamble)

    rows: list[dict] = []
    dropped: list[str] = []
    too_long_found: list[dict] = []
    chapter = (
        f"{preamble_chapters[-1][0]} {preamble_chapters[-1][1]}".strip()
        if preamble_chapters
        else ""
    )
    label_characters = 0
    labels_cut = 0
    headings_between: list[str] = []

    for index, candidate in enumerate(chain):
        end = chain[index + 1].start if index + 1 < len(chain) else len(text)
        body = text[candidate.body_start : end]
        if index + 1 == len(chain):
            trailer = _TRAILER.search(body)
            if trailer:
                note(
                    "text after the last provision (schedules, forms or notes)",
                    body[trailer.start() :],
                )
                body = body[: trailer.start()]
        body, chapters, cut = _cut_labels(body)
        label_characters += cut
        labels_cut += len(chapters)
        body, group = _trailing_heading(body)
        if group:
            headings_between.append(group)
            label_characters += len(group)
        body = _SPACE.sub(" ", body).strip()
        heading = candidate.heading
        number = candidate.number
        this_chapter = chapter
        if chapters:
            chapter = f"{chapters[-1][0]} {chapters[-1][1]}".strip()

        removed = _REMOVED_BY.match(f"{heading} {body}")
        if removed or _DROPPED.match(heading) or _DROPPED.match(body) or _ONLY_MARKS.match(body):
            if not body or _ONLY_MARKS.match(body):
                what = heading or "no text"
            else:
                what = f"{heading} {body}".strip()
            dropped.append(
                f"{unit} {number}: '{what[:60]}' ({len(body)} characters) — record its dates"
            )
            continue
        if len(body) > too_long:
            too_long_found.append({"provision": f"{unit} {number}", "characters": len(body)})
        rows.append(
            {
                "statute": statute,
                "unit": unit,
                "number": number,
                "heading": heading,
                "text": body,
                "in_force_from": in_force_from,
                **({"chapter": this_chapter} if this_chapter else {}),
            }
        )

    if labels_cut or headings_between:
        not_imported.append(
            {
                "what": f"{labels_cut} chapter/part heading(s) and {len(headings_between)} group "
                "heading(s) between provisions (kept as chapter labels, not text)",
                "characters": label_characters,
                "starts": "; ".join(headings_between[:3]),
            }
        )

    seen = {c.number for c in chain}
    report = {
        "provisions": len(rows),
        "characters": len(text),
        "imported_characters": sum(len(r["text"]) + len(r["heading"]) for r in rows),
        "not_imported": not_imported,
        "empty": [f"{unit} {r['number']}" for r in rows if not r["text"]],
        "omitted_or_repealed": dropped,
        "rejected_headings": rejected,
        "missing_from_contents": [
            f"{unit} {n}" for n in toc_order if n not in seen and not _LISTED_AS_GONE.search(toc[n])
        ],
        "imported_but_not_in_contents": [f"{unit} {n}" for n in unlisted if n in seen],
        "listed_as_repealed_in_contents": [
            f"{unit} {n}" for n in toc_order if n not in seen and _LISTED_AS_GONE.search(toc[n])
        ],
        "too_long": too_long_found,
    }
    return rows, report
