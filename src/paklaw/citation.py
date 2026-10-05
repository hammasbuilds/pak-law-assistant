"""Pakistani legal citations: parsing, normalisation, formatting.

Legal writing is unusually citation-dense and unusually precise about it. "Section 302"
alone is meaningless — 302 of *what* — and the same provision appears in half a dozen
written forms:

    Section 302 PPC
    s. 302 of the Pakistan Penal Code, 1860
    sec 302, P.P.C.
    §302 PPC
    u/s 302 PPC
    302 PPC

They must all resolve to the same key, or a retrieval system silently treats them as
different provisions and an answer citing one will not match a query naming another.

Three families are handled, because Pakistani legal argument uses all three and they
behave differently:

  **statutory**    a provision of an Act or the Constitution
  **subordinate**  an SRO, notification or rule made under an Act
  **reported**     a judgment, cited by law report — PLD, SCMR, CLC, YLR

Reported citations matter because a statute's *meaning* frequently lives in the case
law rather than the text, and a system that can parse the text but not the case citation
cannot follow the argument.

Two kinds of mistake are worse than missing a citation, and the patterns are built
against both:

  **reading the wrong provision** — "section 302-B" read as 302, which is murder;
  "Article 25 QSO" read as the Constitution's Article 25
  **inventing one** — "Rs. 500" read as "s. 500", "Part 3" as "Article 3"
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, replace

# Common statute abbreviations, normalised to a canonical key.
STATUTES: dict[str, str] = {
    "ppc": "PPC",
    "p.p.c.": "PPC",
    "pakistan penal code": "PPC",
    "penal code": "PPC",
    "tazirat-e-pakistan": "PPC",
    "tazirat e pakistan": "PPC",
    "تعزیرات پاکستان": "PPC",
    "crpc": "CrPC",
    "cr.p.c.": "CrPC",
    "code of criminal procedure": "CrPC",
    "criminal procedure code": "CrPC",
    "ضابطہ فوجداری": "CrPC",
    "cpc": "CPC",
    "c.p.c.": "CPC",
    "code of civil procedure": "CPC",
    "civil procedure code": "CPC",
    "ضابطہ دیوانی": "CPC",
    "qso": "QSO",
    "qanun-e-shahadat": "QSO",
    "qanun e shahadat": "QSO",
    "qanun-e-shahadat order": "QSO",
    "قانون شہادت": "QSO",
    "constitution": "CONST",
    "constitution of pakistan": "CONST",
    "constitution of the islamic republic of pakistan": "CONST",
    "آئین پاکستان": "CONST",
    "آئین": "CONST",
    "companies act": "COMPANIES",
    "income tax ordinance": "ITO",
    "sales tax act": "STA",
    "contract act": "CONTRACT",
    "specific relief act": "SRA",
    "limitation act": "LIMITATION",
    "pdpa": "PDPA",
    "personal data protection act": "PDPA",
    "peca": "PECA",
    "prevention of electronic crimes act": "PECA",
}

# Abbreviations that identify an Act on their own, written as practitioners write them:
# "302 PPC" with no "section" is ordinary usage. Case-sensitive, so a lower-case word
# never reads as an Act.
_BARE_ABBREVIATIONS = ("PPC", "P.P.C.", "CrPC", "Cr.P.C.", "CPC", "C.P.C.", "PECA", "QSO")

# Law reports used in Pakistani practice.
REPORTS = {"PLD", "SCMR", "CLC", "YLR", "MLD", "PTD", "PLC", "CLD", "PCrLJ", "NLR"}

# Foreign series that appear in Pakistani drafting, usually for pre-partition or
# comparative authority. They are recognised so they can be REPORTED as foreign rather
# than dropped: the same argument as an unrecognised Act. A citation nobody parsed is a
# citation nobody checked, and an audit that returns "no citations found" over a draft
# full of AIR authority has told the reader the opposite of the truth.
FOREIGN_REPORTS = {"AIR", "SCC", "AC", "WLR", "QB", "KB", "ER", "Cr.LJ", "CrLJ", "ILR"}

# 302, 302A, 302-B, 20(1), 20(1)(a). The hyphenated form is how 489-F and 354-A are
# written; without it "302-B" is read as 302, a different offence.
# The leading digits must not be all zeros: no Act has a section 0, and "section 0"
# parsing to a real-looking citation meant an audit reported it as an unknown provision
# of the PPC rather than as something that is not a citation at all.
_NUMBER = (
    r"(?!0+(?![1-9]))\d{1,4}(?!\d)(?:-?[A-Z]{1,3}(?![a-z]))?"
    r"(?:\(\d{1,3}[A-Z]?\))*(?:\([a-z]{1,4}\))?"
)
# Later members of a list — "sections 302/34", "302, 34 and 109". At most three digits, so
# "section 20, 2016" does not read the year as a second section.
_LIST_MEMBER = r"\d{1,3}(?!\d)(?:-?[A-Z]{1,3}(?![a-z]))?(?:\(\d{1,3}[A-Z]?\))*(?:\([a-z]{1,4}\))?"
# "ss. 302-304", "sections 10 to 14". Written as an optional tail on a number rather
# than as an alternative to one: an alternation would make the engine attempt a range at
# every number in a list and backtrack out of it, and a charge sheet with ten thousand
# citations is a real input here. The tail must start with a digit, so the letter forms
# ("489-F", "354-A") are consumed by _NUMBER and never read as a range.
_RANGE_TAIL = rf"\s*(?:-|–|—|\bto\b)\s*{_LIST_MEMBER}"
_SPAN = rf"{_NUMBER}(?:{_RANGE_TAIL})?"
_LIST_SPAN = rf"{_LIST_MEMBER}(?:{_RANGE_TAIL})?"
_LIST = rf"{_SPAN}(?:\s*(?:/|,|&|\band\b)\s*{_LIST_SPAN})*"
_MEMBER = re.compile(_NUMBER, re.I)
_RANGE_MEMBER = re.compile(rf"(?P<one>{_NUMBER})(?P<tail>{_RANGE_TAIL})?", re.I)
# A charge sheet cites a handful of consecutive sections; a hundred is a drafting error
# or a mis-parse, and expanding it would bury the real citations.
_RANGE_LIMIT = 50


def members(numbers: str) -> list[str]:
    """Every provision number a citation's number list names, ranges expanded.

    "302-304" is three sections, and reporting one of them means an audit passes a draft
    whose other two citations were never checked. A range wider than `_RANGE_LIMIT` is
    reported as its endpoints rather than expanded, because at that width it is far more
    likely to be a mis-parse than a citation.
    """
    out: list[str] = []
    for match in _RANGE_MEMBER.finditer(numbers or ""):
        first = match.group("one")
        tail = match.group("tail")
        if not tail:
            out.append(first)
            continue
        last = _MEMBER.search(tail).group(0)
        if first.isdigit() and last.isdigit() and 0 <= int(last) - int(first) <= _RANGE_LIMIT:
            out.extend(str(n) for n in range(int(first), int(last) + 1))
        else:
            out.extend([first, last])
    return out


# A letter immediately before the marker means it is the end of a word: "Rs. 500",
# "Part 3", "chart 5". None of those is a citation.
_NOT_AFTER_LETTER = r"(?<![A-Za-z])"

# An Act title this module does not know: up to five capitalised words ending in the
# noun Acts are named with. "Indian Penal Code", "Companies Act", "Qanun-e-Shahadat
# Order". Not a statute key - the point is only to notice that one was named.
_ACT_TITLE = r"(?-i:(?:[A-Z][\w.'\-]*\s+){1,5}(?:Code|Act|Ordinance|Order|Rules|Regulations))"

# Divisions of an Act that are not provisions. The bare pattern would otherwise read
# "Chapter 5 PPC" as section 5 of the PPC, which is a different thing that exists.
_NOT_A_PROVISION = re.compile(
    r"(?:chapter|part|schedule|sched\.?|clause|paragraph|para\.?|proviso|table|form|entry|"
    r"item|preamble|division)\s*$",
    re.I,
)


# Urdu text writes numbers in Extended Arabic-Indic (۳۰۲) or Arabic-Indic (٣٠٢) digits;
# "دفعہ ۳۰۲" and "section 302" are one provision.
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def normalise_number(number: str) -> str:
    """One spelling per provision: '489-f' -> '489F', '20(1)(A)' -> '20(1)(a)'.

    The base is upper-cased and de-hyphenated because both forms appear for the same
    section; a clause letter stays lower-case because that is how it is cited.
    """
    number = str(number).strip().translate(_DIGITS)
    match = re.match(r"(\d+)-?([A-Za-z]*)(.*)$", number)
    if not match:
        return number.upper()
    digits, letters, rest = match.groups()
    return f"{digits}{letters.upper()}{rest.lower()}"


def _names(extra: tuple[str, ...]) -> str:
    return "|".join(
        sorted((re.escape(k) for k in set(STATUTES) | set(extra)), key=len, reverse=True)
    )


@functools.lru_cache(maxsize=16)
def _patterns(extra: tuple[str, ...] = ()) -> tuple[re.Pattern, re.Pattern, re.Pattern]:
    """Section, article and bare-abbreviation patterns for a set of statute names.

    Built per set, because a corpus can hold Acts this module has never heard of:
    without them, "section 5 PRPA" parses as a bare "section 5" of no Act.
    """
    names = _names(extra)
    # A known Act first; failing that, any Act-like title. Capturing the second is what
    # lets a citation say "an Act was named and I do not know it", which is a different
    # thing from naming no Act at all - and the difference decides whether a default
    # statute may be applied. Case-sensitive, because an Act title is capitalised and
    # "under the code" is not one.
    statute = (
        r"(?:\s*,?\s*(?:of\s+the\s+|of\s+)?(?P<statute>" + names + r")(?![A-Za-z])"
        r"|\s*,?\s*(?:of\s+)?(?:the\s+)?(?P<unknown_statute>" + _ACT_TITLE + r")(?![A-Za-z]))?"
    )
    section = re.compile(
        _NOT_AFTER_LETTER
        # The dot is optional on the bare abbreviation too: "s 302 PPC" and
        # "ss 302/34" are written without it constantly, and _NOT_AFTER_LETTER
        # already stops the "s" of a word from starting a citation.
        + r"(?:sections?|secs?\.?|ss?\.?|§§?|u/ss?\.?|daf(?:a|ah|fa)|دفعہ|دفعات)\s*(?P<numbers>"
        + _LIST
        + r")"
        + statute,
        re.I,
    )
    article = re.compile(
        _NOT_AFTER_LETTER + r"(?:articles?|arts?\.|آرٹیکل)\s*(?P<numbers>" + _LIST + r")" + statute,
        re.I,
    )
    bare_names = sorted(
        {re.escape(a) for a in _BARE_ABBREVIATIONS} | {re.escape(k) for k in extra_keys(extra)},
        key=len,
        reverse=True,
    )
    bare = re.compile(
        r"(?<![\w/.\-§])(?P<numbers>\d{1,3}(?!\d)(?:-[A-Z])?)\s+(?P<statute>"
        + "|".join(bare_names)
        + r")(?![A-Za-z])"
    )
    return section, article, bare


def extra_keys(extra: tuple[str, ...]) -> set[str]:
    """Upper-case keys of corpus-only statutes, usable bare like "5 PRPA"."""
    return {e.upper() for e in extra if e.isalpha()}


def statute_aliases(keys) -> dict[str, str]:
    """Lower-cased name -> key for statute keys the built-in table does not know."""
    known = set(STATUTES.values())
    return {k.lower(): k for k in keys if k and k not in known}


# "Order XXXIX Rule 1 CPC" - civil procedure is cited by Order and Rule, not section.
_ORDER_RULE = re.compile(
    _NOT_AFTER_LETTER + r"order\s+(?P<order>[IVXLC]+)\s*,?\s*r(?:ule|\.)\s*(?P<rule>\d+)"
    r"(?:\s*,?\s*(?:of\s+the\s+)?(?P<statute>cpc|c\.p\.c\.|code of civil procedure))?",
    re.I,
)

# "Rule 5 of the Companies Rules", "rule 12 of the Income Tax Rules 2002". Rules made
# under an Act are the commonest form of subordinate legislation in practice, and they
# used to parse to nothing at all - so a draft citing them was audited as though it had
# cited nothing, which is worse than reporting them as unheld: a silent drop looks like a
# clean bill of health.
_RULES = re.compile(
    _NOT_AFTER_LETTER + r"rules?\s*\.?\s*(?P<numbers>" + _LIST + r")"
    r"\s*,?\s*(?:of|under)\s+the\s+(?P<rules>" + _ACT_TITLE + r")"
    r"(?:\s*,?\s*(?P<year>\d{4}))?",
    re.I,
)

# "SRO 1125(I)/2011" - subordinate legislation.
_SRO = re.compile(
    _NOT_AFTER_LETTER
    + r"s\.?r\.?o\.?\s*(?P<number>\d+)\s*(?:\((?P<series>[IVX]+)\))?\s*/\s*(?P<year>\d{4})",
    re.I,
)

# Benches as they are printed in a citation, abbreviated and in full. PLD reports the
# High Courts far more often than the Supreme Court, and a court name the pattern did
# not know used to make the whole citation vanish - so a draft full of High Court
# authority was reported as containing no citations at all.
_COURTS = (
    "SC",
    "LHC",
    "SHC",
    "PHC",
    "BHC",
    "IHC",
    "FSC",
    "Lah",
    "Lahore",
    "Kar",
    "Karachi",
    "Pesh",
    "Peshawar",
    "Quetta",
    "Islamabad",
    "Sindh",
    "Balochistan",
    "Baluchistan",
    "Shariat",
    r"AJ&K",
    "AJK",
    "Gilgit",
    "Trib",
    "Tribunal",
)
# Longest first, so "Lahore" is not matched as "Lah" with "ore" left over.
_COURT = "|".join(sorted((c for c in _COURTS), key=len, reverse=True))

_FOREIGN_UPPER = frozenset(r.upper().replace(".", "") for r in FOREIGN_REPORTS)


def is_foreign_report(report: str) -> bool:
    """Whether this series is a foreign one. `Citation.report` is upper-cased."""
    return report.upper().replace(".", "") in _FOREIGN_UPPER


# Longest first, so PCrLJ is not matched as PLC with a tail left over.
_REPORT_NAMES = "|".join(sorted(REPORTS | FOREIGN_REPORTS, key=len, reverse=True))

# "PLD 2015 SC 401", "2019 SCMR 1234" - both orderings occur in practice.
_REPORT_A = re.compile(
    r"\b(?P<report>" + _REPORT_NAMES + r")\s+(?P<year>\d{4})\s+"
    r"(?P<court>" + _COURT + r")?\s*(?P<page>\d+)\b",
    re.I,
)
_REPORT_B = re.compile(
    r"\b(?P<year>\d{4})\s+(?P<report>" + _REPORT_NAMES + r")\s+"
    r"(?P<court>" + _COURT + r")?\s*(?P<page>\d+)\b",
    re.I,
)


@dataclass(frozen=True)
class Citation:
    kind: str  # "statutory" | "subordinate" | "reported"
    statute: str = ""  # canonical key, e.g. "PPC"
    provision: str = ""  # "302", "302B", "25", "XXXIX/1", "20(1)(a)"
    unit: str = ""  # "section" | "article" | "rule"
    # An Act the text named that this module could not resolve to a key - "the Indian
    # Penal Code". Empty when the text named no Act at all. The two cases look identical
    # in `statute` and must not be treated alike: a default statute may be applied to
    # the second and never to the first.
    named_statute: str = ""
    report: str = ""
    year: str = ""
    court: str = ""
    page: str = ""
    raw: str = ""

    @property
    def key(self) -> str:
        """Canonical identity. Every written form of one provision maps here."""
        if self.kind == "reported":
            return f"{self.report}:{self.year}:{self.court or '-'}:{self.page}"
        if self.kind == "subordinate":
            if self.unit == "rule":
                # Rules made under an Act, not a statutory order.
                return f"RULES:{self.named_statute or self.statute}:{self.provision}"
            return f"SRO:{self.provision}:{self.year}"
        return f"{self.statute or 'UNKNOWN'}:{self.unit}:{self.provision}"

    def pretty(self) -> str:
        if self.kind == "reported":
            return " ".join(x for x in (self.report, self.year, self.court, self.page) if x)
        if self.kind == "subordinate":
            if self.unit == "rule":
                title = self.named_statute or self.statute
                year = f" {self.year}" if self.year else ""
                return f"Rule {self.provision} of the {title}{year}".strip()
            return f"SRO {self.provision}/{self.year}"
        if self.unit == "rule" and "/" in self.provision:
            # Written the way it is cited; "Rule XXXIX/1 CPC" is in no judgment.
            order, rule = self.provision.split("/", 1)
            return f"Order {order} Rule {rule} {self.statute}".strip()
        label = {"section": "Section", "article": "Article", "rule": "Rule"}.get(
            self.unit, self.unit.title()
        )
        return f"{label} {self.provision} {self.statute}".strip()


def _canonical_statute(text: str | None, default: str = "") -> str:
    """Resolve any written form of a statute name to its canonical key.

    Both the dotted and undotted forms are looked up, because abbreviations appear
    either way and the table holds only one of them: stripping the trailing dot turns
    the key "p.p.c." into "p.p.c" and silently loses the match, leaving the provision
    attributed to no Act at all.
    """
    if not text:
        return default
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    return (
        STATUTES.get(cleaned)
        or STATUTES.get(cleaned.rstrip("."))
        or STATUTES.get(cleaned + ".")
        or default
    )


CANONICAL_STATUTES = sorted(set(STATUTES.values()))


def normalise_statute(value: str) -> str | None:
    """A canonical key from the key itself in any case ("peca") or any name the parser
    knows ("Pakistan Penal Code", "P.P.C."). None when it is neither."""
    lowered = value.strip().lower()
    for key in CANONICAL_STATUTES:
        if key.lower() == lowered:
            return key
    return _canonical_statute(lowered) or None


def parse(text: str, *, statutes: dict[str, str] | None = None) -> list[Citation]:
    """Every citation in a passage, in order of appearance."""
    return [citation for _, citation in locate(text, statutes=statutes)]


def locate(text: str, *, statutes: dict[str, str] | None = None) -> list[tuple[int, Citation]]:
    """`parse`, with the character offset where each citation starts.

    Overlaps are resolved by consuming matched spans: "Order XXXIX Rule 1 CPC" must not
    also yield a bare rule, or one citation becomes two and the count of authorities in
    an answer is wrong. Consumed spans are marked in a byte mask, so the cost is linear
    in the text rather than quadratic in the number of citations.

    `statutes` adds names beyond the built-in table (lower-cased name -> key), for Acts a
    corpus holds that this module does not know; see `statute_aliases`.
    """
    extra = statutes or {}
    section_pattern, article_pattern, bare_pattern = _patterns(tuple(sorted(extra)))
    found: list[tuple[int, Citation]] = []
    consumed = bytearray(len(text))

    def claim(match: re.Match) -> bool:
        start, end = match.span()
        if any(consumed[start:end]):
            return False
        consumed[start:end] = b"\x01" * (end - start)
        return True

    def statute_of(named: str | None, default: str = "") -> str:
        if not named:
            return default
        return extra.get(named.lower()) or _canonical_statute(named, default)

    for match in _RULES.finditer(text):
        if claim(match):
            title = " ".join(match.group("rules").split())
            for number in members(match.group("numbers")):
                found.append(
                    (
                        match.start(),
                        Citation(
                            kind="subordinate",
                            unit="rule",
                            provision=normalise_number(number),
                            named_statute=title,
                            year=match.group("year") or "",
                            raw=match.group(0),
                        ),
                    )
                )

    for match in _ORDER_RULE.finditer(text):
        if claim(match):
            found.append(
                (
                    match.start(),
                    Citation(
                        kind="statutory",
                        statute=_canonical_statute(match.group("statute"), "CPC"),
                        provision=f"{match.group('order').upper()}/{match.group('rule')}",
                        unit="rule",
                        raw=match.group(0),
                    ),
                )
            )

    for match in _SRO.finditer(text):
        if claim(match):
            found.append(
                (
                    match.start(),
                    Citation(
                        kind="subordinate",
                        provision=match.group("number"),
                        year=match.group("year"),
                        raw=match.group(0),
                    ),
                )
            )

    for pattern in (_REPORT_A, _REPORT_B):
        for match in pattern.finditer(text):
            if claim(match):
                found.append(
                    (
                        match.start(),
                        Citation(
                            kind="reported",
                            report=match.group("report").upper(),
                            year=match.group("year"),
                            court=(match.group("court") or "").upper(),
                            page=match.group("page"),
                            raw=match.group(0),
                        ),
                    )
                )

    # Articles belong to the Constitution unless another instrument is named: the
    # Qanun-e-Shahadat Order is also numbered in articles, and "Article 25 QSO" is not
    # the equality clause.
    listed = (
        (article_pattern, "article", "CONST"),
        (section_pattern, "section", ""),
        (bare_pattern, "section", ""),
    )
    for pattern, unit, default in listed:
        for match in pattern.finditer(text):
            # "Chapter 5 PPC" is not section 5 of the PPC. The bare pattern cannot see
            # what precedes the number, so the division words are rejected here.
            # Only the words immediately before it: slicing the whole preceding text
            # here is quadratic, and a ten-thousand-citation charge sheet is a real
            # input. The longest division word is well inside this window.
            if pattern is bare_pattern and _NOT_A_PROVISION.search(
                text[max(0, match.start() - 24) : match.start()]
            ):
                continue
            if not claim(match):
                continue
            named = match.groupdict().get("unknown_statute") or ""
            statute = statute_of(match.group("statute"), "" if named else default)
            for number in members(match.group("numbers")):
                found.append(
                    (
                        match.start(),
                        Citation(
                            kind="statutory",
                            statute=statute,
                            provision=normalise_number(number),
                            unit=unit,
                            named_statute=" ".join(named.split()),
                            raw=match.group(0),
                        ),
                    )
                )

    return sorted(found, key=lambda pair: pair[0])


def resolve_bare(citations: list[Citation], *, default_statute: str) -> list[Citation]:
    """Attach a statute to citations that did not name one.

    In a document about the Penal Code, "section 302" means 302 PPC. Resolving that from
    context is normal legal reading — but the default must be supplied explicitly by
    whatever knows the context, never guessed, because attributing a provision to the
    wrong Act is a serious error that looks like a correct answer.

    A citation that named an Act this module could not resolve is left alone. "Section
    302 of the Indian Penal Code" has no `statute`, and filling it from a PPC default
    turns a foreign provision into a Pakistani one with the same number — which is that
    serious error, arrived at by a route that looks like helpfulness.
    """
    return [
        c
        if (c.statute or c.named_statute or c.kind != "statutory")
        else replace(c, statute=default_statute)
        for c in citations
    ]
