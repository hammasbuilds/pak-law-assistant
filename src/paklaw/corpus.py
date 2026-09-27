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

    def __post_init__(self) -> None:
        # Checked here, at load, because every one of these otherwise surfaces later as a
        # crash in the middle of answering: an integer number has no .lower(), a null text
        # cannot be tokenised, and "Section" never matches a parsed "section".
        if isinstance(self.number, int) and not isinstance(self.number, bool):
            self.number = str(self.number)
        for name in ("statute", "unit", "number", "heading", "text"):
            if not isinstance(getattr(self, name), str):
                raise CorpusError(f"{name} must be a string, got {getattr(self, name)!r}")
        for name in ("superseded_by", "manner", "amended_by", "language", "chapter", "enacted_by"):
            if getattr(self, name) is None:
                setattr(self, name, "")
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

    def status_note(self, as_of: dt.date) -> str:
        """What a reader must be told about this text before relying on it."""
        if self.in_force_on(as_of):
            return ""
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
            }
            for p in self.versions(key)
        ]

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

        return problems

    def __iter__(self) -> Iterator[Provision]:
        return iter(self.provisions)


def build(rows: Sequence[dict]) -> Corpus:
    corpus = Corpus()
    for row in rows:
        corpus.add(Provision(**row))
    return corpus
