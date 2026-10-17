"""What the Pakistan Code says *about* an Act, as opposed to its text.

`sources.pakistan_code` turns one Act's PDF into clean text. This reads the three things
around that text which decide how the Act is recorded, and each exists because getting
it wrong produces a corpus that is confidently wrong rather than visibly incomplete:

**The date in the header.** Every Act in the Code opens with its number and the date of
assent in brackets - ``Act No. XXVII of 1997`` / ``[16th August, 1997]``. A date guessed
from the year in the title instead would answer "what was the law on 3 March 1997" with
an Act that had not yet been passed, so an Act whose header cannot be read keeps the
title's year with ``start_known=False``: the corpus's own word for "this is where our
record begins, not when this came into force".

**The repeal note in the title.** 86 of the Code's 1,034 Acts are annotated as repealed
in the title itself - ``Agricultural Census Act, 1958 (Repealed by Act XiV of 2011
s.3(w.e.f 31-05-2011))``. Unread, every provision of all 86 is served as the law in
force today, which is precisely the failure `corpus` was built to prevent. Ten carry an
exact ``w.e.f`` date. The rest name the year at best, and are recorded with
``end_known=False`` against the earliest date they can have ended.

**A short key.** A corpus row's ``statute`` is what a citation has to match, so the key
is what a person would write: ``PPC``, ``CONST``, ``ATA``.
"""

from __future__ import annotations

import datetime as dt
import re

MONTHS = {
    month.lower(): number
    for number, month in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        1,
    )
}

# "[16th August, 1997]", "[ 2nd July 2016 ]", "[16 August, 1997]"
_BRACKET_DATE = re.compile(
    r"\[\s*(\d{1,2})\s*(?:st|nd|rd|th)?\s+([A-Za-z]{3,12})\s*,?\s*(\d{4})\s*\]"
)
# "Act No. XXVII of 1997", "ACT NO. XLV OF 1860", "Ordinance No. LI of 2002".
#
# Not `\b`: the Code prints the header's footnote marker glued to the word, "1Act No.
# XXVII of 1997", and between "1" and "A" there is no word boundary at all - so every
# Act whose number carries a marker was recorded with no number.
_ACT_NUMBER = re.compile(
    r"(?<![A-Za-z])(Act|Ordinance|Order|Regulation)\s+No\.?\s*([IVXLCDM]+|\d+)\s+of\s+(\d{4})",
    re.I,
)
_YEAR = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")

#: How far into the text a bracketed date may be found by position alone. Past this a
#: date in brackets is a commencement notified for one section, or part of an
#: illustration, rather than the Act's own.
HEADER = 12_000

# The Code prints the date of assent right-aligned on a line of its own, between the
# Act's number and the enacting formula. Matching that shape rather than a position is
# what finds it in the Constitution, whose 176-page table of contents puts its
# "[12th April, 1973]" some 30,000 characters in - so the whole Constitution was left
# out of the corpus for having "no date in the header and no year in the title".
_DATE_LINE = re.compile(
    r"^[ \t]*\[\s*\d{1,2}\s*(?:st|nd|rd|th)?\s+[A-Za-z]{3,12}\s*,?\s*\d{4}\s*\][ \t]*$"
)


def _as_date(day: str, month: str, year: str) -> dt.date | None:
    number = MONTHS.get(month.lower())
    if not number:
        return None
    try:
        return dt.date(int(year), number, int(day))
    except ValueError:
        return None  # "[31st June, 1997]" is not a date


def assent_date(text: str) -> dt.date | None:
    """The date of assent printed in the Act's header, or None if it has none."""
    # A line that is nothing but a bracketed date, wherever it falls.
    for line in text.splitlines():
        if _DATE_LINE.match(line):
            found = _BRACKET_DATE.search(line)
            if found and (date := _as_date(*found.groups())):
                return date
    # Otherwise a bracketed date close enough to the top to be the Act's own.
    for day, month, year in _BRACKET_DATE.findall(text[:HEADER]):
        if date := _as_date(day, month, year):
            return date
    return None


def act_number(text: str) -> str:
    """The Act's number as the header prints it, "Act XXVII of 1997", or ""."""
    found = _ACT_NUMBER.search(text[:HEADER])
    if not found:
        return ""
    return f"{found.group(1).title()} {found.group(2).upper()} of {found.group(3)}"


def title_year(title: str) -> int | None:
    years = _YEAR.findall(title)
    return int(years[-1]) if years else None


# ---- the index's annotations ---------------------------------------------------------

# "(Repealed by ...)" / "(Repeal by ...)" / "(Repealed)", to the end: the notes nest
# brackets and often do not close them, so this deliberately runs to the end of the title
# rather than to the matching ")".
#
# The word alone is not enough. "Pakistan International Airlines Corporation (Conversion)
# (Repeal) Act, 2026" is an Act whose PURPOSE is to repeal something - "(Repeal)" is part
# of its name - and reading it as an annotation stripped the name back to the Conversion
# Act, which the repealing Ordinance of the same year also stripped to. Two different
# instruments then shared one statute key, and the corpus held two conflicting "section
# 1"s with a gap in force between them.
# Two shapes, and "(Repeal)" is neither: the bare annotation is the past participle and
# ends the title, while the usual one names the instrument that did it.
_REPEAL_NOTE = re.compile(
    r"\(\s*repealed\s*\)\s*$"
    r"|\(\s*repeal(?:ed)?\s*\b(?:by|vide|through|w\.?e\.?f)\b[^)]*(?:\)[^)]*)*$",
    re.I,
)
_WEF = re.compile(r"w\.?\s*e\.?\s*f\.?\s*:?\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})", re.I)
# "Act XIV of 2015", "Ord. 60 of 2002", "ordinance 20 of 02". A year is two digits or
# four and never three: without the guard, the typo "ord.37 of 200" matched its first two
# digits and filed a 1976 Act as repealed in 2020 - a claim that it was live for 44 years.
_INSTRUMENT = re.compile(
    r"\b(Act|Ord\.?|Ordinance|Order|Regulation)\s*(?:No\.?)?\s*([IVXLCDM]+|\d+)\s*of\s*"
    r"((?:19|20)\d{2}|\d{2})(?!\d)",
    re.I,
)
# Editorial marks printed inside a title: "(Under Review)" on the codes being
# reconsolidated, "(valid 25-03-10)" beside a repeal.
_EDITORIAL = re.compile(r"\(\s*(?:under\s+review|valid\b[^)]*)\)", re.I)


def four_digit(raw: str | int) -> int:
    """Two-digit years, resolved: "01" is 2001 and "97" is 1997.

    A two-digit year left alone reads as year 1 AD.
    """
    year = int(raw)
    if year < 100:
        year += 2000 if year <= 40 else 1900
    return year


def strip_annotations(title: str) -> str:
    """The Act's own name, without the index's notes.

    Brackets are rebalanced rather than trimmed: `.strip(")")` on "Qanun-e-Shahadat
    Order, 1984 (QSO)" leaves "(QSO", which then appears in the provenance line of every
    provision of that Order.
    """
    clean = _EDITORIAL.sub("", _REPEAL_NOTE.sub("", title)).strip(" ,.;")
    while clean.count("(") > clean.count(")"):
        cut = clean.rfind("(")
        if cut <= 0:
            break
        clean = clean[:cut].strip(" ,.;")
    while clean.count(")") > clean.count("("):
        clean = clean.replace(")", "", 1)
    return clean.strip(" ,.;")


def repeal(title: str) -> dict | None:
    """``{"date", "exact", "by"}`` for a title annotated as repealed, else None.

    ``date`` is when it ended, or None when the note gives nothing usable. It is never
    guessed: `ended` turns a missing date into the earliest the Act can have ended.
    """
    found = _REPEAL_NOTE.search(title)
    if not found:
        return None
    note = found.group(0)

    by = ""
    instrument = _INSTRUMENT.search(note)
    if instrument:
        kind = instrument.group(1).rstrip(".").title()
        kind = "Ordinance" if kind.lower().startswith("ord") else kind
        by = f"{kind} {instrument.group(2).upper()} of {four_digit(instrument.group(3))}"

    exact = _WEF.search(note)
    if exact:
        day, month, year = (int(part) for part in exact.groups())
        try:
            return {"date": dt.date(four_digit(year), month, day), "exact": True, "by": by}
        except ValueError:
            pass  # a malformed w.e.f falls through to the year

    years = [int(year) for year in _YEAR.findall(note)]
    if not years and instrument:
        years = [four_digit(instrument.group(3))]
    plausible = [year for year in years if 1800 <= year <= dt.date.today().year + 1]
    if not plausible:
        return {"date": None, "exact": False, "by": by}
    return {"date": dt.date(max(plausible), 1, 1), "exact": False, "by": by}


def ended(start: dt.date, found: dict) -> tuple[dt.date, bool]:
    """The end date to record, and whether it is exact.

    An Act repealed in the year it commenced cannot have ended on 1 January of that
    year, so the date is clamped to the day after commencement - which is the earliest it
    *can* have ended, the same claim `end_known=False` already makes.
    """
    floor = start + dt.timedelta(days=1)
    if found["date"] is None or found["date"] <= start:
        return floor, False
    return found["date"], found["exact"]


# ---- statute keys --------------------------------------------------------------------

_SKIP = {
    "the",
    "of",
    "and",
    "or",
    "a",
    "an",
    "in",
    "for",
    "to",
    "on",
    "by",
    "with",
    "act",
    "ordinance",
    "order",
    "regulation",
    "rules",
    "rule",
    "amendment",
}
#: The forms practitioners actually write. Generated initials would make "302 PPC" miss.
FIXED = {
    "constitution of the islamic": "CONST",
    "pakistan penal code": "PPC",
    "code of criminal procedure": "CrPC",
    "code of civil procedure": "CPC",
    "qanun-e-shahadat": "QSO",
    "contract act": "CONTRACT",
    "specific relief act": "SRA",
    "limitation act": "LIMITATION",
    "companies act": "COMPANIES",
    "income tax ordinance": "ITO",
    "sales tax act": "STA",
    "prevention of electronic crimes act": "PECA",
    "personal data protection act": "PDPA",
}


#: Shorter than this and a key is an ordinary English word before it is a statute name.
#: The Antiquities Act 1975 initialises to "A", which then matched the article "a" as a
#: whole word in any question; `citation.NOT_A_STATUTE_NAME` is the other half of that
#: fix, and this is the half that stops such a key being minted at all.
MIN_KEY = 3


def initials(title: str) -> str:
    """Initials of the title's significant words, lengthened when there are too few.

    A one-word Act gives one letter, so the word itself is abbreviated instead:
    "Antiquities Act, 1975" is ANTIQ rather than A.
    """
    stem = re.sub(r",?\s*\d{4}\s*$", "", title).strip()
    words = [w for w in re.split(r"[^A-Za-z]+", stem) if w and w.lower() not in _SKIP]
    letters = "".join(word[0].upper() for word in words)
    if len(letters) < MIN_KEY and words:
        # Not the initials plus padding: the first word, cut. "ANTIQ" reads as the Act
        # it stands for and "AN" does not.
        letters = "".join(words)[: max(MIN_KEY, 5)].upper()
    return letters[:8] or re.sub(r"[^A-Za-z]", "", stem)[:6].upper() or "ACT"


def assign_keys(titles) -> dict[str, str]:
    """title -> key, with every key used exactly once.

    Collisions are real - two Acts can both initialise to "PA" - and a collision that
    silently merged two statutes under one key would make every lookup of either return
    provisions of both. The year breaks the tie, then a counter.
    """
    keys: dict[str, str] = {}
    taken: set[str] = set()
    for title in sorted(titles):
        low = title.lower()
        fixed = next((code for name, code in FIXED.items() if name in low), None)
        if fixed and fixed not in taken:
            keys[title] = fixed
            taken.add(fixed)
            continue
        base = initials(title)
        candidate = base
        year = re.search(r"\b(1[6-9]\d{2}|20\d{2})\b", title)
        if candidate in taken and year:
            candidate = f"{base}{year.group(1)}"
        counter = 2
        while candidate in taken:
            candidate = f"{base}-{counter}"
            counter += 1
        keys[title] = candidate
        taken.add(candidate)
    return keys
