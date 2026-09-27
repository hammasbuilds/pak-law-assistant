"""Grounded answering: cite a provision in force, or refuse.

Four refusal conditions, and each one exists because the alternative is an answer that
is confident and wrong.

  **No provision found**       — a plausible-sounding answer with no citation
  **Provision not in force**   — a correctly cited, authoritative-looking answer about
                                 law that was repealed
  **Weak match**               — the nearest provision returned as though it were the
                                 relevant one
  **Cited provision unknown**  — the user named a section; answering about a different
                                 one because it scored well is worse than saying so

The second is the one specific to law and the one general RAG systems have no concept
of. A repealed section reads exactly like a live one. Nothing in the text says
otherwise. Only the corpus knows, and only if it was built to.

Every answer is assembled from provision text with its citation attached — the system
never writes law, it quotes it and says where from. Narrative phrasing is a separate,
optional step that receives *verified* provisions, never the raw query.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .citation import Citation, parse, statute_aliases
from .corpus import Corpus
from .retrieve import Hit, LawSearch


@dataclass
class Passage:
    """One provision offered in support of an answer."""

    citation: str
    heading: str
    text: str
    statute: str
    in_force_from: str
    in_force_to: str | None
    score: float
    matched_terms: str
    status_note: str = ""

    @property
    def current(self) -> bool:
        return self.in_force_to is None


@dataclass
class Answer:
    question: str
    as_of: str
    passages: list[Passage] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    # Provisions the user cited that were found, but are not in force on the date asked.
    superseded: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    language: str = "en"

    def summary(self) -> dict:
        return {
            "question": self.question,
            "as_of": self.as_of,
            "refused": self.refused,
            "reason": self.refusal_reason,
            "citations": [p.citation for p in self.passages],
            "superseded": self.superseded,
            "warnings": self.warnings,
        }

    def render(self) -> str:
        """Plain text, citation-first. Deliberately dry — a legal answer that reads
        like advice invites reliance the system has not earned."""
        if self.refused:
            return f"No answer: {self.refusal_reason}"

        blocks = []
        for p in self.passages:
            header = f"{p.citation} — {p.heading}" if p.heading else p.citation
            if p.status_note:
                header += f"  [{p.status_note}]"
            blocks.append(f"{header}\n{p.text}")

        out = "\n\n".join(blocks)
        if self.warnings:
            out += "\n\n" + "\n".join(f"Note: {w}" for w in self.warnings)
        return out


REFUSAL_NOTHING_FOUND = "no provision in force on that date matches the question"
REFUSAL_WEAK = "no provision matched strongly enough to answer from"
REFUSAL_NOT_IN_FORCE = "the cited provision was not in force on that date"
REFUSAL_UNKNOWN_CITATION = "the cited provision is not in this corpus"
REFUSAL_NO_ACT = (
    "the cited provision names no Act, and guessing one could answer about the wrong law"
)


@dataclass
class LawAssistant:
    corpus: Corpus
    search: LawSearch = None  # type: ignore[assignment]
    min_score: float = 1.0
    # A match must contain more than this share of the question's content words. A raw
    # score threshold alone let "Whoever commits theft" through to the murder section on
    # "whoever" and "commits" — words every penal provision contains.
    min_coverage: float = 0.5
    max_passages: int = 3

    def __post_init__(self) -> None:
        if self.search is None:
            self.search = LawSearch(corpus=self.corpus)

    @property
    def aliases(self) -> dict[str, str]:
        """Statutes the corpus holds that the citation parser has no name for."""
        return statute_aliases({p.statute for p in self.corpus})

    # ---- helpers ---------------------------------------------------------------

    @staticmethod
    def _to_passage(hit: Hit, as_of: dt.date) -> Passage:
        p = hit.provision
        return Passage(
            citation=p.citation().pretty(),
            heading=p.heading,
            text=p.text,
            statute=p.statute,
            in_force_from=p.in_force_from.isoformat(),
            in_force_to=p.in_force_to.isoformat() if p.in_force_to else None,
            score=hit.score,
            matched_terms=hit.why(),
            status_note=p.status_note(as_of),
        )

    def _cited_provisions(
        self, question: str, as_of: dt.date, default_statute: str | None
    ) -> tuple[list[Citation], list[dict]]:
        """Citations the user named, and any that are not in force on the date."""
        citations = [c for c in parse(question, statutes=self.aliases) if c.kind == "statutory"]
        if default_statute:
            citations = [
                c if c.statute else Citation(**{**c.__dict__, "statute": default_statute})
                for c in citations
            ]

        superseded: list[dict] = []
        for citation in citations:
            key = self.corpus.resolve(f"{citation.statute}:{citation.unit}:{citation.provision}")
            if not self.corpus.versions(key):
                continue
            if self.corpus.version_on(key, as_of) is None:
                # Describe the version nearest the date. The latest one says a provision
                # asked about before its first enactment "commenced" at its last amendment.
                versions = self.corpus.versions(key)
                nearest = next(
                    (v for v in reversed(versions) if v.in_force_from <= as_of), versions[0]
                )
                superseded.append(
                    {
                        "citation": citation.pretty(),
                        "status": nearest.status_note(as_of) or "not in force on this date",
                        "history": self.corpus.history(key),
                    }
                )
        return citations, superseded

    # ---- the answer ------------------------------------------------------------

    def answer(
        self,
        question: str,
        *,
        as_of: str | dt.date | None = None,
        statute: str | None = None,
    ) -> Answer:
        as_of_date = (
            dt.date.today()
            if as_of is None
            else dt.date.fromisoformat(as_of)
            if isinstance(as_of, str)
            else as_of
        )
        result = Answer(question=question, as_of=as_of_date.isoformat())

        citations, superseded = self._cited_provisions(question, as_of_date, statute)
        result.superseded = superseded

        # A question that names a provision is a lookup, not a search. Returning the
        # nearest match to a citation the user spelled out answers about the wrong law.
        if citations:
            resolved: list[Passage] = []
            unknown: list[str] = []
            no_act: list[str] = []

            for citation in citations:
                if not citation.statute:
                    no_act.append(citation.pretty())
                    continue
                cited_key = f"{citation.statute}:{citation.unit}:{citation.provision}"
                key = self.corpus.resolve(cited_key)
                if not self.corpus.versions(key):
                    unknown.append(citation.pretty())
                    continue
                provision = self.corpus.version_on(key, as_of_date)
                if provision is None:
                    continue  # already recorded in `superseded`
                if key != cited_key:
                    result.warnings.append(
                        f"{citation.pretty()} is a subdivision; the corpus holds whole "
                        f"provisions, so all of {provision.citation().pretty()} is returned"
                    )
                resolved.append(
                    self._to_passage(
                        Hit(provision=provision, score=float("inf"), matched_terms={"cited": 1.0}),
                        as_of_date,
                    )
                )

            if resolved:
                result.passages = resolved[: self.max_passages]
                # Every citation not answered is named. Answering two of three and saying
                # nothing about the third reads as though the third had been covered.
                if len(resolved) > self.max_passages:
                    result.warnings.append(
                        f"{len(resolved)} cited provisions; only the first {self.max_passages} "
                        "are returned — ask about the rest separately"
                    )
                if unknown:
                    result.warnings.append(f"{REFUSAL_UNKNOWN_CITATION}: {', '.join(unknown)}")
                if no_act:
                    result.warnings.append(f"{REFUSAL_NO_ACT}: {', '.join(no_act)}")
                if superseded:
                    result.warnings.append(
                        f"{len(superseded)} cited provision(s) were not in force on "
                        f"{result.as_of}; their history is included"
                    )
                return result

            if superseded:
                result.refused = True
                result.refusal_reason = f"{REFUSAL_NOT_IN_FORCE}: " + "; ".join(
                    f"{s['citation']} — {s['status']}" for s in superseded
                )
                return result

            if unknown:
                result.refused = True
                result.refusal_reason = f"{REFUSAL_UNKNOWN_CITATION}: {', '.join(unknown)}"
                return result

            if no_act:
                result.refused = True
                result.refusal_reason = f"{REFUSAL_NO_ACT}: {', '.join(no_act)}"
                return result

        hits = self.search.search(
            question, as_of=as_of_date, limit=self.max_passages, statute=statute
        )

        if not hits:
            result.refused = True
            result.refusal_reason = REFUSAL_NOTHING_FOUND
            return result

        if hits[0].score < self.min_score:
            result.refused = True
            result.refusal_reason = (
                f"{REFUSAL_WEAK} (best score {hits[0].score:.2f} below {self.min_score:.2f})"
            )
            return result

        if hits[0].coverage <= self.min_coverage:
            top = hits[0]
            result.refused = True
            result.refusal_reason = (
                f"{REFUSAL_WEAK}: the nearest provision contains {len(top.matched_terms)} of "
                f"the question's {len(top.matched_terms) + len(top.missing_terms)} content "
                f"words (missing: {', '.join(top.missing_terms)})"
            )
            return result

        # Passages after the first are held to the same bar, or a strong first answer
        # carries two unrelated ones in with it.
        kept = [h for h in hits if h.coverage > self.min_coverage and h.score >= self.min_score]
        result.passages = [self._to_passage(h, as_of_date) for h in kept]
        return result

    # ---- history ---------------------------------------------------------------

    def history(self, citation_text: str, *, statute: str | None = None) -> dict:
        """The amendment trail of a provision, which is frequently the question itself.

        "When did this change?" cannot be answered from a flat corpus at all, and it is
        asked constantly — about conduct that occurred before an amendment, about which
        version applies to a pending case.
        """
        citations = [
            c for c in parse(citation_text, statutes=self.aliases) if c.kind == "statutory"
        ]
        if not citations:
            return {"error": "no statutory citation found in the request"}

        citation = citations[0]
        resolved_statute = citation.statute or statute or ""
        key = self.corpus.resolve(f"{resolved_statute}:{citation.unit}:{citation.provision}")
        history = self.corpus.history(key)

        if not history:
            return {"citation": citation.pretty(), "error": REFUSAL_UNKNOWN_CITATION}

        return {
            "citation": citation.pretty(),
            "versions": len(history),
            "history": history,
            "currently_in_force": self.corpus.current(key) is not None,
        }
