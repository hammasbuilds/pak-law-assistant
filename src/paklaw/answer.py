"""Grounded answering: cite a provision in force, or refuse.

Nine refusal conditions, and each one exists because the alternative is an answer that
is confident and wrong. Every refusal also carries a `refusal_status` token, so a caller
branches on the kind rather than on prose that will be reworded.

  **No provision found**       — a plausible-sounding answer with no citation
  **Provision not in force**   — a correctly cited, authoritative-looking answer about
                                 law that was repealed
  **Weak match**               — the nearest provision returned as though it were the
                                 relevant one
  **Cited provision unknown**  — the user named a section; answering about a different
                                 one because it scored well is worse than saying so
  **No Act named**             — "section 9" of what? A default could answer about the
                                 wrong Act entirely
  **Act not recognised**       — an Act WAS named and is not one this corpus holds.
                                 "Section 302 of the Indian Penal Code" must not become
                                 Section 302 PPC, which exists and says something else
  **Different offence**        — the question carries a word that selects a neighbouring
                                 provision. "Attempt to murder" is s.324, and answering
                                 it with s.302 returns the death penalty for the wrong
                                 offence
  **Subject not in corpus**    — a word of the question appears nowhere in the corpus in
                                 any form. "Dacoity" is absent from thirteen sections of
                                 the Penal Code, and the other two words of "the sentence
                                 for dacoity with murder" were enough to open the gate
  **Citation unreliable**      — the provision that answers the question is sitting
                                 inside another provision's text, because the source's
                                 contents list stopped before its body did. The answer
                                 is there; the citation would name the wrong section,
                                 which is worse than no answer

The second is the one specific to law and the one general RAG systems have no concept
of. A repealed section reads exactly like a live one. Nothing in the text says
otherwise. Only the corpus knows, and only if it was built to.

Every answer is assembled from provision text with its citation attached — the system
never writes law, it quotes it and says where from. Narrative phrasing is a separate,
optional step that receives *verified* provisions, never the raw query.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from .citation import Citation, find_statute, names_unloaded_act, parse, statute_aliases
from .corpus import Corpus
from .retrieve import Hit, LawSearch, tokenise
from .split import buried_offset


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
    #: What the order was actually decided by. `score` is BM25 alone and is NOT
    #: monotonic down this list - a passage below can score higher - so a client that
    #: sorted on it reordered the answer. The sort is (coverage, ranking); both are
    #: here, so the order a client is given can be checked against numbers it has.
    ranking: float
    coverage: float
    #: The same share weighted by how much each question term tells you, which is what
    #: decides whether this answer is given at all. A question can lose the only word
    #: it is about and still clear a count-based bar: see `Hit.information_coverage`.
    #: Both are reported because they answer different questions, and the gate is on
    #: this one.
    information_coverage: float
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
    # WHY it refused, as a token a client can branch on. The reason is prose and will be
    # reworded; a caller that wants to handle "not in force" differently from "nothing
    # matched" had to match on that prose, which is a contract nobody agreed to.
    # Empty when the question was answered. The names match check_citations' statuses
    # where the two tools mean the same thing.
    refusal_status: str = ""
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
            "refusal_status": self.refusal_status,
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
# An Act WAS named and is not one this corpus knows. Distinct from the above, because
# "names no Act" is false of the input and the audit already draws the distinction:
# answering about a Pakistani provision of the same number is the error.
REFUSAL_FOREIGN_ACT = "the cited provision names an Act this corpus does not hold"
# The question carries a word that selects a neighbouring offence, and the best
# provision does not contain it. "Attempt to murder" is s.324, not s.302.
REFUSAL_QUALIFIER = "the nearest provision is about a different offence"
REFUSAL_UNRELIABLE_CITATION = (
    "the provision that answers this is inside another provision's text in this corpus, "
    "so citing it would name the wrong one"
)
REFUSAL_NOT_IN_CORPUS = (
    "no provision in this corpus contains a word the question turns on, in any form"
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

    # Ranking parameters, forwarded to the index. Here so a sweep can reach them
    # without editing the source, which is how one sweep came to measure nothing.
    tuning: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.search is None:
            self.search = LawSearch(corpus=self.corpus, tuning=self.tuning)

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
            ranking=hit.ranking,
            coverage=round(hit.coverage, 4),
            information_coverage=round(hit.information_coverage, 4),
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
        """An answer, or a refusal, with what the corpus cannot know attached.

        A thin wrapper so the coverage caveat is added at ONE place. `_answer` below
        has eleven return statements - a citation spelled out, a citation not in
        force, an unknown statute, a weak match, an unreliable citation - and
        `Corpus.coverage_warnings` was called from none of them. Its only caller was
        the MCP server's `answer_question`, so `demo.py` and every library consumer
        got the confident answer with no caveat while the README's "record coverage"
        row reads as a property of the library.

        Adding it to the successful return was the first fix and it was wrong: the
        citation branch returns four lines earlier, so a question that spells a
        section out still had no caveat. Eleven returns is eleven chances to miss one,
        which is the same reason it was in the server to begin with.
        """
        result = self._answer(question, as_of=as_of, statute=statute)
        as_of_date = dt.date.fromisoformat(result.as_of)
        # What the corpus does not claim to know about this date. Scoped to the
        # statutes actually cited where there are any, and to the whole corpus
        # otherwise - a refusal is still an answer about a date.
        cited = sorted({p.statute for p in result.passages})
        for note in self.corpus.coverage_warnings(as_of_date, cited or None):
            if note not in result.warnings:
                result.warnings.append(note)
        return result

    def _answer(
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
            # An Act named and not recognised, kept apart from no Act at all. The two
            # look identical in `statute` and mean opposite things to a reader.
            foreign_act: list[str] = []

            for citation in citations:
                if not citation.statute:
                    if citation.named_statute:
                        foreign_act.append(f"{citation.pretty()} ({citation.named_statute})")
                    else:
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
                result.refusal_status = "not_in_force"
                result.refusal_reason = f"{REFUSAL_NOT_IN_FORCE}: " + "; ".join(
                    f"{s['citation']} — {s['status']}" for s in superseded
                )
                return result

            if unknown:
                result.refused = True
                result.refusal_status = "unknown_provision"
                result.refusal_reason = f"{REFUSAL_UNKNOWN_CITATION}: {', '.join(unknown)}"
                return result

            if foreign_act:
                # An Act WAS named and is not one this corpus knows. Saying "names no
                # Act" here was simply false of the input, and check_citations already
                # drew the distinction - two tools reaching different conclusions about
                # the same string is worse than either conclusion.
                result.refused = True
                result.refusal_status = "act_not_recognised"
                result.refusal_reason = (
                    f"{REFUSAL_FOREIGN_ACT}: {', '.join(foreign_act)}. A Pakistani "
                    "provision of the same number is not the same provision."
                )
                return result

            if no_act:
                result.refused = True
                result.refusal_status = "no_act_named"
                result.refusal_reason = f"{REFUSAL_NO_ACT}: {', '.join(no_act)}"
                return result

        # A question that NAMES an Act is scoped to it: "what does the word animal mean
        # in the Penal Code?" says which statute to read, and the words that say so are
        # not content to match against a provision. Leaving them in refused the question
        # for missing "penal" and "code" — words that appear in no provision's text,
        # because they are the name of the book the provisions are in.
        # An Act or a jurisdiction this corpus does not answer for, named in prose
        # rather than in a citation. The citation route already refused "section 302 of
        # the Indian Penal Code"; the prose route scoped on the substring "penal code"
        # and answered "the punishment for murder under the Indian Penal Code" with
        # s.302 PPC. Same words, same question, and the wrong country's criminal law.
        if statute is None:
            foreign = names_unloaded_act(question, extra=self.aliases)
            if foreign is not None:
                result.refused = True
                result.refusal_status = "act_not_recognised"
                result.refusal_reason = (
                    f"{REFUSAL_FOREIGN_ACT}: {foreign!r}. This corpus holds Pakistani "
                    "statutes, and a provision of the same number in another country's "
                    "Act says something else."
                )
                return result

        asked = question
        if statute is None:
            named = find_statute(question, extra=self.aliases)
            if named is not None:
                statute, phrase = named
                asked = re.sub(re.escape(phrase), " ", question, flags=re.I)

        hits = self.search.search(asked, as_of=as_of_date, limit=self.max_passages, statute=statute)

        if not hits:
            # Name the absent word when there is one. "what does the law say about
            # dacoity?" produced no hits at all - every other word of it is a stopword
            # or a verb of asking - and came back "no provision in force on that date
            # matches the question", which is true and says nothing. The same question
            # phrased "what is the punishment for dacoity?" DID name it, because
            # `punishment` matched something and the code reached the branch below
            # that reports unknown terms. Two phrasings of one question, two qualities
            # of answer, for no reason the reader can see.
            missing = self.search.absent_terms(asked, as_of=as_of_date)
            result.refused = True
            if missing:
                result.refusal_status = "subject_not_in_corpus"
                result.refusal_reason = (
                    f"{REFUSAL_NOT_IN_CORPUS}: "
                    f"{', '.join(repr(t) for t in missing)} "
                    f"{'appear' if len(missing) > 1 else 'appears'} in no provision "
                    "here, and nothing else in the question matched either."
                )
            else:
                result.refusal_status = "nothing_matched"
                result.refusal_reason = REFUSAL_NOTHING_FOUND
            return result

        if hits[0].score < self.min_score:
            result.refused = True
            result.refusal_status = "weak_match"
            result.refusal_reason = (
                f"{REFUSAL_WEAK} (best score {hits[0].score:.2f} below {self.min_score:.2f})"
            )
            return result

        # A qualifier the provision does not contain means the question is about a
        # neighbouring offence. Asked "punishment for attempt to murder", the corpus
        # offered s.302 - "punished with death as qisas" - because "attempt" counted as
        # one interchangeable word out of three. It is not interchangeable.
        if hits[0].missing_qualifiers:
            top = hits[0]
            result.refused = True
            result.refusal_status = "different_offence"
            result.refusal_reason = (
                f"{REFUSAL_QUALIFIER}: the question says "
                f"{', '.join(repr(q) for q in top.missing_qualifiers)}, which "
                f"{top.provision.citation().pretty()} does not. That word usually names a "
                "different provision, and this corpus does not hold one matching it."
            )
            return result

        # A word the corpus has never seen, in any form, anywhere. That is a stronger
        # signal than a word this provision happens to lack: asked for the sentence for
        # "dacoity with murder", a corpus of thirteen sections answered s.302, because
        # two of its three words were covered and the one that named the offence was
        # not. The honest answer is that the corpus does not contain dacoity.
        if hits[0].unknown_terms:
            top = hits[0]
            result.refused = True
            result.refusal_status = "subject_not_in_corpus"
            # Says what is true. The old wording - "the question names something this
            # corpus has no provision about" - is true of "dacoity" and false of
            # "acting", and a blocklist of ordinary words cannot tell them apart. An
            # independent review got that sentence out of a question Section 52 PPC
            # answers verbatim, which is a wrong statement of fact delivered with a
            # refusal's authority: the same defect class as a wrong citation.
            result.refusal_reason = (
                f"{REFUSAL_NOT_IN_CORPUS}: "
                f"{', '.join(repr(t) for t in top.unknown_terms)} "
                f"{'appear' if len(top.unknown_terms) > 1 else 'appears'} in no provision "
                f"here. The nearest is {top.provision.citation().pretty()}. If one of "
                "those words is what the question is about, this corpus does not hold it; "
                "if it is incidental, rephrase in the statute's own words."
            )
            return result

        # The weighted share, not the count. See `Hit.information_coverage`: a
        # question can lose the only word it is about and still clear a count-based
        # bar on two or three words every provision contains.
        if hits[0].information_coverage <= self.min_coverage:
            top = hits[0]
            result.refused = True
            result.refusal_status = "weak_match"
            result.refusal_reason = (
                f"{REFUSAL_WEAK}: the nearest provision contains {len(top.matched_terms)} of "
                f"the question's {len(top.matched_terms) + len(top.missing_terms)} content "
                f"words (missing: {', '.join(top.missing_terms)})"
            )
            return result

        # Passages after the first are held to the same bar, or a strong first answer
        # carries two unrelated ones in with it.
        kept = [
            h
            for h in hits
            # The COUNT here, not the weighted share. This filter decides which
            # hits are shown beside the first one, and widening it flipped a
            # deliberate refusal into an answer: "can a court impose simple
            # imprisonment?" is held inside s.57's blob, and letting s.53 through
            # instead of refusing is a different decision from the one being fixed.
            # The weighted share gates the TOP hit, which is where the wrong citation
            # came from.
            if h.coverage > self.min_coverage
            and h.score >= self.min_score
            and not h.missing_qualifiers
        ]
        # A provision whose text runs on into later provisions is not one section, and
        # the reader is the only one who can tell. It is also the one most likely to be
        # returned: nine headings' worth of words match almost any question about the
        # subject, and coverage is the first sort key. Said here rather than only at
        # import, because the person reading the answer is not the person who built the
        # corpus.
        swallowed = self.corpus.swallowed_headings()
        if swallowed:
            index = self.search._index_for(as_of_date)
            for hit in list(kept):
                buried = swallowed.get(hit.provision.key)
                if not buried:
                    continue
                text = hit.provision.text
                cut = buried_offset(text, hit.provision.number)
                own = set(tokenise(text[:cut])) | set(tokenise(hit.provision.heading))
                supported = [t for t in hit.matched_terms if index._forms(t) & own]
                if len(supported) / len(hit.matched_terms) > self.min_coverage:
                    # The provision's own words answer the question; the buried text is
                    # extra, and a caveat is the right size of response.
                    result.warnings.append(
                        f"{hit.provision.citation().pretty()} runs on into "
                        f"{hit.provision.unit}s {', '.join(buried)}; the text served under "
                        "this citation is longer than one provision"
                    )
                    continue
                # The match came from the buried part, so the citation would name the
                # wrong provision - the harm this whole system exists to prevent, and
                # one a warning does not undo. A reader filing "Section 57 PPC" for the
                # fine-default rule has filed s.65.
                kept.remove(hit)
                result.warnings.append(
                    f"a provision answering this appears inside "
                    f"{hit.provision.citation().pretty()}, which runs on into "
                    f"{hit.provision.unit}s {', '.join(buried)}; it is not cited here "
                    "because the citation would name the wrong one"
                )
        if not kept and hits:
            result.refused = True
            result.refusal_status = "citation_unreliable"
            result.refusal_reason = REFUSAL_UNRELIABLE_CITATION
            return result
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
            return {
                "error": "no statutory citation found in the request",
                "refusal_status": "no_citation_found",
            }

        citation = citations[0]
        resolved_statute = citation.statute or statute or ""
        key = self.corpus.resolve(f"{resolved_statute}:{citation.unit}:{citation.provision}")
        history = self.corpus.history(key)

        if not history:
            return {
                "citation": citation.pretty(),
                "error": REFUSAL_UNKNOWN_CITATION,
                "refusal_status": "unknown_provision",
            }

        return {
            "citation": citation.pretty(),
            "versions": len(history),
            "history": history,
            "currently_in_force": self.corpus.current(key) is not None,
        }
