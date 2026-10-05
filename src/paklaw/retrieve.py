"""BM25 retrieval over a statute book, always as of a date.

BM25 rather than embeddings, and that is a considered choice for this corpus rather
than a shortcut.

Legal text is **terminology-bound**. "Reasonable apprehension of death" is a term of
art; a semantically similar paraphrase is not the same provision and retrieving it
instead is a wrong answer. Embeddings are good at paraphrase, which is exactly the
wrong strength here. Statutes also use a small, stable vocabulary that classic term
weighting handles well.

The more important property is that BM25 is **inspectable**. When a lawyer asks why a
provision was returned, "these terms matched with these weights" is an answer. "It was
close in a 384-dimensional space" is not, and in a domain where the reasoning has to be
auditable that difference decides the design.

Implemented from the formula. It is a sum over query terms.
"""

from __future__ import annotations

import bisect
import datetime as dt
import math
import re
from collections import Counter, OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .corpus import Corpus, Provision

_TOKEN = re.compile(r"[\w؀-ۿ']+", re.UNICODE)

# Words that carry no discriminating power in a statute book, where nearly every
# provision contains "section", "act", "shall" and "person".
LEGAL_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "to",
    "for",
    "by",
    "or",
    "and",
    "any",
    "such",
    "shall",
    "be",
    "is",
    "are",
    "as",
    "on",
    "with",
    "under",
    "this",
    "that",
    "it",
    "act",
    "section",
    "sub",
    "clause",
    "provided",
    "may",
    "which",
    "who",
    "where",
    "has",
    "have",
    "had",
    "been",
    "was",
    "were",
    "not",
    "no",
    "if",
    "than",
    "then",
    # Penal boilerplate: nearly every offence begins "Whoever commits ... shall be
    # punished", so these words match a murder section to a question about theft.
    "whoever",
    "person",
    "persons",
    "commit",
    "commits",
    "committed",
    "punished",
    "punishable",
    "liable",
    # Question words. A question is phrased around them and no provision is about them.
    "what",
    "when",
    "how",
    "why",
    "can",
    "could",
    "does",
    "do",
    "did",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "they",
    "their",
    "his",
    "her",
    "him",
    "about",
    "there",
    "law",
    "legal",
}


# Pakistani statutes name offences in Urdu and Arabic terms of art, so the word a person
# asks with is frequently not the word the statute uses: a question about *murder* has to
# reach "qatl-i-amd", and one about *theft* has to reach "chori". Without this bridge the
# weak-match refusal fires on perfectly answerable questions - the nearest provision is
# right there, and the only thing missing is the vocabulary.
#
# This maps vocabulary, never meaning. Each entry is a term the statute book itself uses
# for the concept the English word names; nothing here broadens a provision's scope, and
# no entry is a near-synonym or a paraphrase. A wrong entry would cite the wrong offence,
# so the table stays small and every line has to be defensible from the statute's own
# wording.
STATUTE_VOCABULARY: dict[str, tuple[str, ...]] = {
    "murder": ("qatl", "amd"),
    "homicide": ("qatl",),
    "manslaughter": ("qatl", "khata"),
    # The statute says "causing death"; a person asks about killing.
    "killing": ("qatl", "death"),
    # Ikrah is compulsion. s.303 PPC is headed "Qatl committed under ikrah-i-tam or
    # ikrah-i-naqis", and a question about duress reached none of it.
    "duress": ("ikrah",),
    "compulsion": ("ikrah",),
    "theft": ("chori",),
    "robbery": ("haraabah",),
    "adultery": ("zina",),
    "retaliation": ("qisas",),
    "bloodmoney": ("diyat",),
    "blood": ("diyat",),
    "discretionary": ("tazir",),
    "hurt": ("jurh",),
    "defamation": ("qazf",),
    "intoxication": ("hadd",),
}
# Read the other way too, so a question in the statute's own terms still finds the
# English wording of a heading.
_VOCABULARY_REVERSE: dict[str, tuple[str, ...]] = {}
for _english, _terms in STATUTE_VOCABULARY.items():
    for _term in _terms:
        _VOCABULARY_REVERSE.setdefault(_term, ())
        _VOCABULARY_REVERSE[_term] += (_english,)

# Words that choose BETWEEN neighbouring offences rather than describing one. Attempt
# to murder is s.324, abetment is s.109, conspiracy is s.120B - none of them s.302. A
# question carrying one of these is about a different provision from the same question
# without it, so a provision that does not contain the word is not an answer to it.
#
# Found by an independent review, which asked "what is the punishment for attempt to
# murder?" and was told s.302: "punished with death as qisas". Coverage counted
# "attempt" as one interchangeable content word among three, so two of three cleared
# the gate. The qualifier is not interchangeable; it is the whole question.
QUALIFIERS = frozenset(
    {
        "attempt",
        "attempted",
        "abetment",
        "abet",
        "abetting",
        "conspiracy",
        "conspiring",
        "omission",
        "preparation",
        "threat",
        "threatening",
        "negligence",
        "negligent",
        "accidental",
        "unintentional",
        "involuntary",
        "mitigated",
        "aggravated",
    }
)

# Conservative English suffixes. "punishment" must reach "punished", which is the single
# most common mismatch in a penal code: the question nominalises what the statute
# conjugates. Only applied to ASCII words long enough that the stem stays a word.
_SUFFIXES = ("ments", "ment", "ingly", "ing", "edly", "ed", "es", "s")


def _stems(term: str) -> set[str]:
    """`term` and the stems it could share with a differently inflected form."""
    out = {term}
    if not term.isascii() or not term.isalpha():
        return out
    for suffix in _SUFFIXES:
        if term.endswith(suffix) and len(term) - len(suffix) >= 4:
            out.add(term[: -len(suffix)])
            break
    return out


def expand(term: str) -> set[str]:
    """Every form of `term` that counts as the same term when matching a provision.

    Query-side only: the index keeps the statute's own words, so `Hit.matched_terms` is
    still keyed by what the person actually asked and `why()` stays readable.
    """
    forms = _stems(term)
    for related in STATUTE_VOCABULARY.get(term, ()) + _VOCABULARY_REVERSE.get(term, ()):
        forms |= _stems(related)
    # A stem of the asked word may itself be a vocabulary key ("murders" -> "murder").
    for stem in list(forms):
        for related in STATUTE_VOCABULARY.get(stem, ()) + _VOCABULARY_REVERSE.get(stem, ()):
            forms |= _stems(related)
    return forms


def tokenise(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Words, lowercased. Urdu script is preserved as its own tokens.

    Legal numbers are kept: "302" is one of the most discriminating tokens in a penal
    code, and a tokeniser that drops digits loses the ability to find a section by its
    number.
    """
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in LEGAL_STOPWORDS]


@dataclass
class Hit:
    provision: Provision
    score: float
    matched_terms: dict[str, float] = field(default_factory=dict)
    # Distinct content terms of the question this provision does not contain.
    missing_terms: list[str] = field(default_factory=list)
    # What the ranking actually sorted on: `score` weighted by coverage. Kept as a
    # field rather than recomputed, so the order a client sees can be checked
    # against a number it was given.
    ranking: float = 0.0
    # Share of the question's content terms that appear in this provision's HEADING.
    heading_coverage: float = 0.0
    # Terms that select a different provision and are absent from this one. A hit with
    # any of these is about a neighbouring offence, not this question.
    missing_qualifiers: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Share of the question's content terms the provision contains."""
        total = len(self.matched_terms) + len(self.missing_terms)
        return len(self.matched_terms) / total if total else 0.0

    def why(self) -> str:
        """Why this provision was returned, in terms a lawyer can check."""
        ranked = sorted(self.matched_terms.items(), key=lambda kv: -kv[1])[:5]
        terms = ", ".join(f"{term} ({weight:.2f})" for term, weight in ranked)
        matched = len(self.matched_terms)
        total = matched + len(self.missing_terms)
        # The share matched is what the ranking weights by, so it belongs in the
        # explanation - but only for a search. A provision the question cited by
        # number is a lookup with one synthetic term, and "1 of 1 terms" there
        # describes nothing.
        return f"{matched} of {total} terms — {terms}" if total > 1 else terms


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    # Matching a section *number* should outrank matching its prose.
    number_boost: float = 3.0
    heading_boost: float = 2.0
    # How hard to prefer a provision that contains ALL of the question's terms.
    #
    # BM25 alone gets this wrong here, and the failure is not subtle. Asked "how is
    # imprisonment for life reckoned in fractions of punishment?", it ranked s.53
    # "Punishments" (385 characters, 3 of 5 terms) above s.57 "Fractions of terms of
    # punishment" (2,602 characters, 5 of 5) — because length normalisation taxes the
    # longer provision harder than the two extra terms reward it. The answer was in
    # the corpus, scored second, and a confident citation of the wrong section came
    # out instead.
    #
    # Statute text is terminology-bound: a question's words are terms of art, and a
    # provision containing every one of them is the answer. So the score is weighted
    # by the fraction of query terms matched — Lucene's old `coord`, dropped there on
    # scoring-theory grounds that do not apply to short keyword queries over a small,
    # controlled vocabulary.
    #
    # Measured on tests/test_retrieval_quality.py: any weight above zero fixes that
    # question and nothing regresses; `b` makes no difference at all across 0.0-0.75,
    # which is why it is still the standard 0.75 rather than tuned. 1.0 is the linear
    # form — multiply by the share matched — and is the easiest to explain to anyone
    # auditing a ranking.
    coverage_weight: float = 1.5
    # The heading is a TIEBREAK, never a multiplier, and the distinction was forced by
    # two questions that pull opposite ways.
    #
    # "What is the State?" returned Article 2 CONST ("Islam shall be the State
    # religion") over Article 7 CONST, headed "Definition of the State", because both
    # contain the word and Article 2 is shorter. One content word also makes coverage
    # 1.00 for everything that matches at all, so coverage cannot separate them.
    #
    # "Imprisonment for life is reckoned as how many years" wants s.57 "Fractions of
    # terms of punishment", whose heading shares NO word with the question, over s.55
    # "Commutation of sentence of imprisonment for life", whose heading matches two. As
    # a multiplier the heading made that worse, overturning the coverage difference
    # (0.80 against 0.60) that had just been fixed.
    #
    # So coverage decides first and the heading only separates hits that cover the
    # question equally well. `heading_focus` is the share of the HEADING'S OWN words the
    # question matched - how much that provision is ABOUT those words - not the share of
    # the question found in the heading, which both of the Article 2/7 headings score at
    # 1.00 and which therefore separates nothing.
    heading_weight: float = 1.0

    documents: list[Provision] = field(default_factory=list)
    _tokens: list[list[str]] = field(default_factory=list)
    _frequencies: list[Counter] = field(default_factory=list)
    _document_frequency: Counter = field(default_factory=Counter)
    _average_length: float = 0.0
    # stem -> the statute's own words that reduce to it, so a query term can find the
    # inflection the statute actually used without the index losing that word.
    _by_stem: dict[str, set[str]] = field(default_factory=dict)
    # The heading's own tokens, kept apart from the body. A provision whose HEADING is
    # about the question is the provision about the question - "Definition of the State"
    # answers "what is the State?" and Article 2, which merely mentions the State while
    # being about Islam, does not.
    _heading_tokens: list[set[str]] = field(default_factory=list)

    def fit(self, provisions: Sequence[Provision]) -> BM25Index:
        self.documents = list(provisions)
        self._tokens = []
        self._frequencies = []
        self._document_frequency = Counter()
        self._by_stem = {}
        self._heading_tokens = []

        for provision in self.documents:
            tokens = (
                tokenise(provision.text)
                + tokenise(provision.heading) * int(self.heading_boost)
                # The number is repeated so term frequency carries the boost, rather
                # than bolting a separate score on afterwards.
                + [provision.number.lower()] * int(self.number_boost)
            )
            self._tokens.append(tokens)
            self._heading_tokens.append(set(tokenise(provision.heading)))
            frequencies = Counter(tokens)
            self._frequencies.append(frequencies)
            self._document_frequency.update(frequencies.keys())
            for token in frequencies:
                for stem in _stems(token):
                    self._by_stem.setdefault(stem, set()).add(token)

        lengths = [len(t) for t in self._tokens]
        self._average_length = sum(lengths) / len(lengths) if lengths else 0.0
        return self

    def _forms(self, term: str) -> set[str]:
        """Index terms that count as `term`: itself, its inflections, its statute word."""
        forms = {term}
        for stem in expand(term):
            forms |= self._by_stem.get(stem, set())
        return forms

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self._document_frequency.get(term, 0)
        # Lucene's variant: always positive, so a term in most documents contributes
        # little rather than subtracting score from documents that contain it.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(
        self, query: str, *, limit: int = 5, where: Callable[[Provision], bool] | None = None
    ) -> list[Hit]:
        """Top `limit` hits among the provisions `where` accepts.

        The filter is applied before the cut, not after: filtering the top nine for one
        Act finds nothing when the other Act's provisions happen to score higher.
        """
        if not self.documents:
            return []

        terms = tokenise(query)
        if not terms:
            return []

        hits: list[Hit] = []
        for index, provision in enumerate(self.documents):
            if where is not None and not where(provision):
                continue
            frequencies = self._frequencies[index]
            length = len(self._tokens[index])
            score = 0.0
            matched: dict[str, float] = {}

            for term in set(terms):
                # The term as asked, plus the statute's own wording for it. Scored on
                # the best single form rather than the sum, so a word that happens to
                # have several inflections in one provision does not outweigh a word
                # that appears once.
                best = 0.0
                for form in self._forms(term):
                    frequency = frequencies.get(form, 0)
                    if not frequency:
                        continue
                    idf = self._idf(form)
                    denominator = frequency + self.k1 * (
                        1 - self.b + self.b * length / (self._average_length or 1)
                    )
                    best = max(best, idf * frequency * (self.k1 + 1) / denominator)
                if not best:
                    continue
                score += best
                matched[term] = round(best, 4)

            if score > 0:
                missing = sorted(set(terms) - set(matched))
                coverage = len(matched) / (len(matched) + len(missing))
                heading = self._heading_tokens[index]
                hit_in_heading = {t for t in heading if any(t in self._forms(q) for q in terms)}
                heading_coverage = len(hit_in_heading) / len(heading) if heading else 0.0
                hits.append(
                    Hit(
                        provision=provision,
                        score=round(score, 6),
                        matched_terms=matched,
                        missing_terms=missing,
                        heading_coverage=round(heading_coverage, 4),
                        missing_qualifiers=sorted(set(missing) & QUALIFIERS),
                        ranking=round(score * coverage**self.coverage_weight, 6),
                    )
                )

        # Coverage first, then the ranking, then how much the heading is about the
        # question. A provision containing every word of the question is a better answer
        # than one containing most of them, whatever the term weights say - and among
        # provisions that cover it equally, the one the draftsman headed with those
        # words is the one about them.
        hits.sort(
            key=lambda h: (-h.coverage, -h.ranking * (1 + self.heading_weight * h.heading_coverage))
        )
        return hits[:limit]


@dataclass
class LawSearch:
    """Retrieval that is always anchored to a date.

    There is no way to search the corpus without supplying one. That is deliberate: an
    optional date parameter defaults to something, and the default eventually gets used
    for a question where it is wrong.
    """

    corpus: Corpus
    # One index per date asked about. A long-running server is asked about arbitrarily
    # many dates, so the cache is bounded; the least recently used index is dropped.
    max_indexes: int = 32
    _indexes: OrderedDict[int, BM25Index] = field(default_factory=OrderedDict)
    _boundaries: list[dt.date] = field(default_factory=list)
    _boundaries_for: int = -1

    def _epoch(self, as_of: dt.date) -> int:
        """Which interval between consecutive commencements/repeals a date falls in.

        Every date in one interval sees exactly the same provisions, so they share an
        index: 2020-06-01 and 2021-03-15 are one index, not two.
        """
        size = len(self.corpus)
        if self._boundaries_for != size:
            dates = {p.in_force_from for p in self.corpus}
            dates |= {p.in_force_to for p in self.corpus if p.in_force_to}
            self._boundaries = sorted(dates)
            self._boundaries_for = size
            self._indexes.clear()
        return bisect.bisect_right(self._boundaries, as_of)

    def _index_for(self, as_of: dt.date) -> BM25Index:
        key = self._epoch(as_of)
        if key in self._indexes:
            self._indexes.move_to_end(key)
        else:
            self._indexes[key] = BM25Index().fit(self.corpus.as_of(as_of))
            while len(self._indexes) > self.max_indexes:
                self._indexes.popitem(last=False)
        return self._indexes[key]

    def search(
        self,
        query: str,
        *,
        as_of: str | dt.date,
        limit: int = 5,
        statute: str | None = None,
    ) -> list[Hit]:
        as_of = dt.date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
        where = (lambda p: p.statute == statute) if statute else None
        return self._index_for(as_of).search(query, limit=limit, where=where)

    def by_citation(self, key: str, *, as_of: str | dt.date) -> Provision | None:
        """Look up a provision the user named directly.

        A question that cites a section is not a search problem — returning the
        *nearest* provision to a citation the user spelled out is how a system answers
        about the wrong law.
        """
        return self.corpus.version_on(key, as_of)
