"""The statute book, as a thing that changes over time.

The failure this module exists to prevent.

Legal text is **versioned**. Sections are amended, substituted and repealed, and the
text of a provision on one date is not the text on another. A retrieval system that
treats a statute as a flat document will happily answer a question about today's law
using a provision repealed in 2016 — and the answer will be fluent, specific, correctly
cited, and wrong.

That is worse than refusing, because a refusal prompts someone to check and a confident
citation does not. A lawyer acting on a repealed provision, or a citizen told their
conduct is lawful under a section that no longer exists, has been harmed by the system
working exactly as designed.

So every provision carries the dates it was in force, retrieval is always **as of** a
date, and superseded text is retained rather than deleted — because questions about past
conduct are asked against the law as it then stood, and deleting history makes those
unanswerable.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from .citation import Citation, normalise_number

UNITS = {"section", "article", "rule"}

# "(1)", "(1)(a)" after a provision number: subsections and clauses.
_SUBDIVISION = re.compile(r"(?:\([0-9a-z]+\))+$", re.I)


class CorpusError(ValueError):
    pass


def _as_date(value: str | dt.date | None) -> dt.date | None:
    if value is None or isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(value)


@dataclass
class Provision:
    """One section, article or rule, as it stood over one interval."""

    statute: str
    unit: str  # "section" | "article" | "rule"
    number: str
    heading: str
    text: str
    in_force_from: dt.date
    # None means still in force.
    in_force_to: dt.date | None = None
    # What replaced it, where a provision was substituted rather than simply repealed.
    superseded_by: str = ""
    # How it ceased: "repealed", "substituted", "omitted".
    manner: str = ""
    # The instrument that ENDED this version — paired with `manner`, never with the
    # start. Read the other way, a 2016 commencement is attributed to a 2022 ordinance.
    amended_by: str = ""
    language: str = "en"
    chapter: str = ""
    # The instrument that brought this version into force, where it was not the Act itself.
    enacted_by: str = ""
    # False when `in_force_from` is where the corpus's record of this statute begins,
    # not when this text came into force: the text was in force on that date and may
    # have been for years, but nothing earlier is held.
    start_known: bool = True
    # What a reader must know about this version that its dates do not say: a court
    # order striking part of it down, or that its text is not held at all.
    note: str = ""
    # Where this version's text was taken from.
    source: str = ""

    def __post_init__(self) -> None:
        # Checked here, at load, because every one of these otherwise surfaces later as a
        # crash in the middle of answering: an integer number has no .lower(), a null text
        # cannot be tokenised, and "Section" never matches a parsed "section".
        if isinstance(self.number, int) and not isinstance(self.number, bool):
            self.number = str(self.number)
        for name in ("statute", "unit", "number", "heading", "text"):
            if not isinstance(getattr(self, name), str):
                raise CorpusError(f"{name} must be a string, got {getattr(self, name)!r}")
        for name in (
            "superseded_by",
            "manner",
            "amended_by",
            "language",
            "chapter",
            "enacted_by",
            "note",
            "source",
        ):
            if getattr(self, name) is None:
                setattr(self, name, "")
        if not isinstance(self.start_known, bool):
            raise CorpusError(f"start_known must be true or false, got {self.start_known!r}")
        self.unit = self.unit.strip().lower()
        if self.unit not in UNITS:
            raise CorpusError(f"unit must be one of {sorted(UNITS)}, got {self.unit!r}")
        self.number = normalise_number(self.number)
        try:
            self.in_force_from = _as_date(self.in_force_from)
            self.in_force_to = _as_date(self.in_force_to)
        except (TypeError, ValueError) as exc:
            raise CorpusError(f"{self.key}: bad date: {exc}") from exc
        if self.in_force_from is None:
            raise CorpusError(f"{self.key}: in_force_from is required")
        if self.in_force_to and self.in_force_to <= self.in_force_from:
            raise CorpusError(
                f"{self.key}: in force {self.in_force_from} to {self.in_force_to} is never in "
                "force at all"
            )

    @property
    def key(self) -> str:
        return f"{self.statute}:{self.unit}:{self.number}"

    @property
    def version_key(self) -> str:
        return f"{self.key}@{self.in_force_from.isoformat()}"

    def in_force_on(self, date: dt.date) -> bool:
        if date < self.in_force_from:
            return False
        return self.in_force_to is None or date < self.in_force_to

    @property
    def currently_in_force(self) -> bool:
        return self.in_force_to is None

    def citation(self) -> Citation:
        return Citation(
            kind="statutory",
            statute=self.statute,
            unit=self.unit,
            provision=self.number,
            raw=self.key,
        )

    @property
    def text_held(self) -> bool:
        return bool(self.text.strip())

    def status_note(self, as_of: dt.date) -> str:
        """What a reader must be told about this text before relying on it."""
        if self.in_force_on(as_of):
            return ""
        if as_of < self.in_force_from and not self.start_known:
            return (
                f"not recorded before {self.in_force_from.isoformat()}: this corpus holds "
                f"{self.statute} only from that date, so the text in force on "
                f"{as_of.isoformat()} is unknown here"
            )
        if as_of < self.in_force_from:
            return (
                f"not yet in force on {as_of.isoformat()} "
                f"(commenced {self.in_force_from.isoformat()})"
            )
        manner = self.manner or "repealed"
        note = f"{manner} with effect from {self.in_force_to.isoformat()}"
        if self.superseded_by:
            note += f"; see {self.superseded_by}"
        return note


@dataclass
class Corpus:
    provisions: list[Provision] = field(default_factory=list)
    # The date the corpus was last brought up to date: an amendment after it is not in
    # it, so an answer about a later date must say so.
    as_at: dt.date | None = None
    # Free-form provenance (sources, licence), carried through to corpus_info.
    meta: dict = field(default_factory=dict)
    # key -> versions oldest first. Rebuilt when the list changes size, so appending to
    # `provisions` directly still works; without it every lookup scans the whole corpus.
    _by_key: dict[str, list[Provision]] = field(default_factory=dict, repr=False)
    _indexed: int = field(default=-1, repr=False)

    def _index(self) -> dict[str, list[Provision]]:
        if self._indexed != len(self.provisions):
            by_key: dict[str, list[Provision]] = {}
            for p in self.provisions:
                by_key.setdefault(p.key, []).append(p)
            for versions in by_key.values():
                versions.sort(key=lambda p: p.in_force_from)
            self._by_key, self._indexed = by_key, len(self.provisions)
        return self._by_key

    def add(self, provision: Provision) -> Provision:
        self.provisions.append(provision)
        return provision

    def __len__(self) -> int:
        return len(self.provisions)

    def as_of(self, date: str | dt.date) -> list[Provision]:
        """Every provision in force on a date. The only way retrieval should read."""
        date = _as_date(date)
        return [p for p in self.provisions if p.in_force_on(date)]

    def versions(self, key: str) -> list[Provision]:
        """Every version of one provision, oldest first."""
        return list(self._index().get(key, ()))

    def resolve(self, key: str) -> str:
        """The key to look up for a citation, falling back from a subsection to its section.

        Corpora are built per section, and legal writing cites subsections constantly:
        "section 20(1) PECA" must find section 20, not be refused as unknown. The exact
        key wins when a corpus does hold subsections separately.
        """
        if self.versions(key):
            return key
        enclosing = _SUBDIVISION.sub("", key)
        return enclosing if enclosing != key and self.versions(enclosing) else key

    def current(self, key: str) -> Provision | None:
        return next((p for p in self.versions(key) if p.currently_in_force), None)

    def version_on(self, key: str, date: str | dt.date) -> Provision | None:
        date = _as_date(date)
        return next((p for p in self.versions(key) if p.in_force_on(date)), None)

    def history(self, key: str) -> list[dict]:
        """The amendment trail, which is itself frequently the answer."""
        return [
            {
                "from": p.in_force_from.isoformat(),
                "to": p.in_force_to.isoformat() if p.in_force_to else None,
                "manner": p.manner or ("in force" if p.currently_in_force else ""),
                "amended_by": p.amended_by,
                "superseded_by": p.superseded_by,
                "heading": p.heading,
                **({"enacted_by": p.enacted_by} if p.enacted_by else {}),
                **(
                    {"from_note": "the start of this corpus's record, not a commencement"}
                    if not p.start_known
                    else {}
                ),
                **({"note": p.note} if p.note else {}),
            }
            for p in self.versions(key)
        ]

    def recorded_from(self, statute: str) -> dt.date | None:
        """The first date from which the corpus holds `statute`, where it says so."""
        floors = [
            p.in_force_from for p in self.provisions if p.statute == statute and not p.start_known
        ]
        return max(floors) if floors else None

    def coverage_warnings(self, date: dt.date, statutes: Sequence[str] | None = None) -> list[str]:
        """What the corpus cannot vouch for on `date`: before a statute's record begins, or
        after the corpus was last brought up to date."""
        warnings = []
        for statute in sorted(set(statutes) if statutes else {p.statute for p in self}):
            start = self.recorded_from(statute)
            if start and date < start:
                warnings.append(
                    f"{statute} is recorded here only from {start.isoformat()}; provisions of "
                    f"{statute} in force on {date.isoformat()} are not all held"
                )
        if self.as_at and date > self.as_at:
            warnings.append(
                f"the corpus records amendments up to {self.as_at.isoformat()}; a change after "
                f"that date would not be reflected for {date.isoformat()}"
            )
        return warnings

    def validate(self) -> list[str]:
        """Structural problems that would produce wrong answers silently.

        Overlapping versions are the dangerous one: if two versions of a section are
        both in force on a date, retrieval returns whichever it happens to reach first
        and the result is not reproducible.
        """
        problems: list[str] = []
        for key, versions in self._index().items():
            ordered = versions
            for earlier, later in zip(ordered, ordered[1:], strict=False):
                if earlier.in_force_to is None:
                    problems.append(
                        f"{key}: version from {earlier.in_force_from} never ends, but "
                        f"another begins {later.in_force_from}"
                    )
                elif earlier.in_force_to > later.in_force_from:
                    problems.append(
                        f"{key}: versions overlap between {later.in_force_from} and "
                        f"{earlier.in_force_to}"
                    )
                elif earlier.in_force_to < later.in_force_from and not later.enacted_by:
                    # A gap is almost always a data error — a mistyped date. A provision
                    # that really was repealed and later re-enacted says so: the later
                    # version names the instrument that brought it back.
                    problems.append(
                        f"{key}: gap in force between {earlier.in_force_to} and "
                        f"{later.in_force_from}; if it was re-enacted, give the later "
                        "version an enacted_by"
                    )

            if sum(1 for p in versions if p.currently_in_force) > 1:
                problems.append(f"{key}: more than one version is currently in force")
            if any(not p.start_known for p in versions[1:]):
                problems.append(f"{key}: only the first version can begin where the record does")

        floors: dict[str, set] = {}
        for p in self.provisions:
            if not p.start_known:
                floors.setdefault(p.statute, set()).add(p.in_force_from)
        for statute, dates in sorted(floors.items()):
            if len(dates) > 1:
                listed = ", ".join(sorted(d.isoformat() for d in dates))
                problems.append(
                    f"{statute}: the record begins on {len(dates)} different dates ({listed}); "
                    "a statute's record has one beginning"
                )

        return problems

    def __iter__(self) -> Iterator[Provision]:
        return iter(self.provisions)


def build(rows: Sequence[dict], *, as_at: str | dt.date | None = None) -> Corpus:
    corpus = Corpus(as_at=_as_date(as_at))
    for row in rows:
        corpus.add(Provision(**row))
    return corpus
