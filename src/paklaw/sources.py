"""Turning the two public sources of Pakistani statute text into plain text.

**The Pakistan Code** (pakistancode.gov.pk, Ministry of Law and Justice) publishes each
Act as a PDF. Run through ``pdftotext -layout`` it is a page at a time: body text, then a
block of numbered footnotes ("2Subs. by Act XLIV of 2016, s. 2."), then "Page 40 of 179".
Amended words carry a marker that points at a footnote on the same page — ``2[three
thousand rupees]`` — and omitted text is ``1[* * *]``. Left in, the footnotes become the
tail of whichever section happens to end the page, and every marker digit becomes a
stray number in the law ("fine which may extend to 2 three thousand rupees").

**pakistani.org** publishes some Acts as HTML, with the same idea in a different shape:
``133[:] 133`` in the text and a *Notes* section at the end that often keeps the words an
amendment replaced ("Substituted ... for: "fourteen years""), which is where the history
of a provision can be recovered from.

Both functions keep what they remove: a footnote is evidence of an amendment, so it is
returned beside the clean text rather than thrown away.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

# "Page 40 of 179", and the consolidation stamp "Dated: 30-11-2025", "RGN Date:
# 10-09-2026" or "RGN: Dated 07-02-2025", which can share a line with a footnote.
_PAGE_FOOTER = re.compile(r"^\s*Page \d+ of \d+\s*$")
_STAMP = re.compile(
    r"[ \t]*(?:RGN\s*:?\s*)?(?:Date|Dated)\s*:?\s*(\d{1,2})-(\d{1,2})-(\d{4})[ \t]*$", re.I
)
# A footnote starts at the left margin with its number glued to the vocabulary of an
# amendment note: "2Subs. by", "1 Omitted by", "2ss.292B and 292C omitted", "3subs. by",
# "*An offence ...". The vocabulary matters: a wrapped note line such as "23rd March,
# 1956)." also starts with a number and a letter, and read as note 23 it breaks the count.
_NOTE_WORDS = (
    r"(?:subs|sub\b|ins\b|ins\.|inserted|substituted|omitted|omitt|added|rep\b|rep\.|"
    r"repealed|renumber|re-number|numbered|the\b|ss?\.|ibid|proviso|words?\b|comma|colon|"
    r"full|semi|explanation|clause|section|new\b|original|for\b|declared|now\b|vide|see\b|"
    r"brought|amended|existing|figure|paragraph|deleted|art\.|article|certain|this\b|"
    r"these\b|an?\b|by\b|in\b|entry|item|part\b|chapter|schedule|sub-|marginal|heading|"
    r"short|w\.e\.f|with\b|re-?lettered|lettered|illustration|1ns\b|\"|“|')"
)
_FOOTNOTE_START = re.compile(r"^\s{0,3}(\d{1,3}|l|\*)\s?(?=" + _NOTE_WORDS + ")", re.I)
_LOOSE_START = re.compile(r"^\s{0,3}(\d{1,3})\s?(?=[A-Z\"“'(])")
# A note number printed on a line of its own, its text on the next line.
_BARE_NUMBER = re.compile(r"^\s{0,3}(\d{1,3})\s*$")
# A marker opens an amended span, "2[", or stands for words omitted in place, "4* * *";
# "*[" is the unnumbered form. The last two alternatives are the brackets the markers
# must be matched against.
_MARKER = re.compile(r"(?:(?<![\w\]])(\d{1,3})(\[|(?=\*))|\*(\[))|\[|\]")
_STARS = re.compile(r"\*(?:[ \t]*\*)+")
# Characters the PDFs' fonts map oddly: a Greek question mark for every semicolon, and a
# soft hyphen where a printed hyphen was meant ("Governor­General").
_PDF_CHARACTERS = str.maketrans({";": ";", "­": "-", "‐": "-", "‑": "-"})
_STARRED_NUMBER = re.compile(r"(?m)^([ \t]*)\*(?=\d)")
STAR = 0  # the number given to an unnumbered "*" note


@dataclass
class Footnote:
    page: int
    number: int
    text: str


@dataclass
class Amended:
    """One marked span in the clean text: the words an amendment put there."""

    start: int
    end: int
    page: int
    number: int
    footnote: str = ""


@dataclass
class SourceText:
    text: str
    footnotes: list[Footnote] = field(default_factory=list)
    amended: list[Amended] = field(default_factory=list)
    stamp: str | None = None  # the consolidation date printed on the pages, ISO


def _note_number(token: str) -> int:
    return STAR if token == "*" else 1 if token == "l" else int(token)


def _merged(lines: list[str]) -> list[tuple[int, str]]:
    """Lines as (original index, text), with a bare note number joined to its text."""
    merged: list[tuple[int, str]] = []
    skip = -1
    for index, line in enumerate(lines):
        if index <= skip:
            continue
        bare = _BARE_NUMBER.match(line)
        if bare:
            ahead = next(
                (k for k in range(index + 1, min(index + 4, len(lines))) if lines[k].strip()),
                None,
            )
            if ahead is not None and re.match(r"\s*" + _NOTE_WORDS, lines[ahead], re.I):
                merged.append((index, bare.group(1) + lines[ahead].strip()))
                skip = ahead
                continue
        merged.append((index, line))
    return merged


def _footnote_block(merged: list[tuple[int, str]]) -> int:
    """Position in `merged` of the first footnote line on a page, or len(merged).

    Footnotes are numbered from 1 on every page and run to the foot of the page, so the
    block starts at a note numbered 1 (or an unnumbered "*" note) after which the notes
    count on without a break. A note-like line whose number does not continue the count
    is a wrapped line of the note before it.
    """
    starts = [
        (position, _note_number(match.group(1)))
        for position, (_, line) in enumerate(merged)
        if (match := _FOOTNOTE_START.match(line))
    ]
    for first, (position, number) in enumerate(starts):
        if number not in (1, STAR):
            continue
        expected = number
        consistent = True
        for _, following in starts[first + 1 :]:
            # One missing number is tolerated: a note whose first word is not in the
            # vocabulary must not throw away every note after it.
            if following in (expected + 1, expected + 2) or (expected == STAR and following == 1):
                expected = following
            elif following == 1:
                consistent = False  # a second count: this "1" was body text
                break
        if consistent:
            return position
    return len(merged)


def pakistan_code(raw: str) -> SourceText:
    """Clean ``pdftotext -layout`` output of a Pakistan Code PDF."""
    stamp = None
    body_parts: list[tuple[int, str]] = []
    footnotes: list[Footnote] = []

    raw = raw.replace("\r\n", "\n").translate(_PDF_CHARACTERS)
    for page_number, page in enumerate(raw.split("\f"), 1):
        kept = []
        for line in page.split("\n"):
            if _PAGE_FOOTER.match(line):
                continue
            found = _STAMP.search(line)
            if found:
                if stamp is None:
                    day, month, year = found.groups()
                    stamp = f"{year}-{int(month):02d}-{int(day):02d}"
                line = line[: found.start()]
                if not line.strip():
                    continue
            kept.append(line)
        while kept and not kept[-1].strip():
            kept.pop()
        merged = _merged(kept)
        cut = _footnote_block(merged)
        current: Footnote | None = None
        expected: int | None = None
        for _, line in merged[cut:]:
            # Inside the block the vocabulary is not needed: the next number in the count
            # followed by a capital opens a note, whatever its first word.
            match = _FOOTNOTE_START.match(line)
            loose = _LOOSE_START.match(line)
            if not match and loose and _note_number(loose.group(1)) == (expected or 0) + 1:
                match = loose
            number = _note_number(match.group(1)) if match else None
            if match and (current is None or number == (expected or 0) + 1 or number in (1, STAR)):
                current = Footnote(page_number, number, line[match.end() :].strip())
                footnotes.append(current)
                expected = number
            elif line.strip() and current is not None:
                current.text += " " + line.strip()
        body_end = merged[cut][0] if cut < len(merged) else len(kept)
        body_parts.append((page_number, "\n".join(kept[:body_end])))

    numbers = {(f.page, f.number): f.text for f in footnotes}
    text_parts: list[str] = []
    amended: list[Amended] = []
    stack: list[Amended | None] = []
    offset = 0
    for page_number, body in body_parts:
        # A leading "*" on a provision number ("*273.") points at a "*" note on the page.
        body = _STARRED_NUMBER.sub(r"\1", body)
        out: list[str] = []
        written = 0
        position = 0
        # Markers only count where the page really has that footnote; "2[" with no
        # footnote 2 on the page is left alone rather than eaten.
        for match in _MARKER.finditer(body):
            piece = body[position : match.start()]
            out.append(piece)
            written += len(piece)
            position = match.end()
            token = match.group(0)
            here = offset + written
            if match.group(1) is not None or match.group(3) is not None:
                number = STAR if match.group(3) is not None else int(match.group(1))
                opens = match.group(2) == "[" or match.group(3) is not None
                if (page_number, number) not in numbers:
                    out.append(token)
                    written += len(token)
                    if opens:
                        stack.append(None)
                    continue
                span = Amended(here, here, page_number, number, numbers[(page_number, number)])
                if opens:
                    stack.append(span)
                else:  # "4* * *": words omitted in place, no closing bracket
                    amended.append(span)
            elif token == "[":
                stack.append(None)
                out.append(token)
                written += 1
            else:  # "]"
                opened = stack.pop() if stack else None
                if opened is None:
                    out.append(token)
                    written += 1
                else:
                    opened.end = here
                    amended.append(opened)
        out.append(body[position:])
        chunk = "".join(out)
        text_parts.append(chunk)
        offset += len(chunk) + 1
    text = "\n".join(text_parts)
    amended.sort(key=lambda a: a.start)
    return SourceText(text=text, footnotes=footnotes, amended=amended, stamp=stamp)


def strip_stars(text: str) -> str:
    """Remove the "* * *" that stands for omitted words, keeping the layout."""
    return _STARS.sub("", text)


# ---- pakistani.org --------------------------------------------------------------------

_NOTE = re.compile(
    r'<div id="(\d+)"><a name="\d+"><a href="#f\d+"><sup>\d+</sup></a></a>(.*?)</div>', re.S
)
_INLINE_MARK = re.compile(r'<sup>(?:&nbsp;|\s)*<a href="#(\d+)" name="f\d+">\d+</a></sup>')


def _plain(fragment: str) -> str:
    fragment = re.sub(r"(?is)<(script|style).*?</\1>", "", fragment)
    fragment = re.sub(r"(?i)<br\s*/?>|</p>|</h\d>|</div>|</tr>|</blockquote>", "\n", fragment)
    fragment = re.sub(r"(?i)</td>", " ", fragment)
    text = html.unescape(re.sub(r"<[^>]+>", "", fragment))
    text = text.replace("\xa0", " ")
    text = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def pakistani_org(page: str) -> tuple[str, dict[int, str]]:
    """Body text of a pakistani.org statute page, and its notes by number.

    The inline markers ("133[:] 133") are removed from the body; the brackets they wrap
    are kept only where the page did not put them there.
    """
    notes = {int(n): _plain(body) for n, body in _NOTE.findall(page)}
    notes_start = page.find("<h3>Notes</h3>")
    body = page[:notes_start] if notes_start > 0 else page
    body = _INLINE_MARK.sub("\x00", body)
    # "\x00[" opens an amended span and "]\x00" closes one; "\x00[]\x00" is an omission.
    body = body.replace("\x00[", "").replace("]\x00", "").replace("\x00", "")
    return _plain(body), notes
