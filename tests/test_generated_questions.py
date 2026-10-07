"""Questions generated FROM the corpus, as a population nobody chose.

The three answerable sets in `test_retrieval_quality.py` were written by hand - one
with the retriever open, one against the provisions, one from the provisions alone -
and an independent review's point about the must-refuse sets applies to these too: a
benchmark of hand-written sentences is a claim about those sentences. Each was written
knowing what it should return, which is exactly the knowledge a reader does not have.

So this generates them instead: a heading question per provision, and three windows of
the provision's own words at seeded offsets. The expected citation is the provision the
question came from, so nothing was chosen.

What it measures, over 101 questions on the 26-provision fixture corpus:

    right 87    wrong 5    refused 9

Three of those five are genuinely ambiguous rather than wrong. Every word of the
question appears in BOTH the expected provision and the one cited - s.54 and s.55 are
both about commutation by the provincial government, and a four-word window drawn from
one is a true description of the other. The expectation is arbitrary there, and
counting it as an error would make this test a measurement of the generator. The two
that remain are in the s.53-s.57 cluster, where the provisions share most of their
vocabulary, and they are recorded as the figure rather than explained away.

The refusals were the interesting half and are the reason this file exists. It first
measured 27 of 101 - a quarter of questions drawn from a provision's own words turned
away - and 26 of those were the heading questions, EVERY ONE of them. "what does the
law say about punishment of qatl-i-amd?" was refused with "'say' appears in no
provision here", of a corpus whose s.302 is headed "Punishment of qatl-i-amd".

The verbs of asking were not in `NOT_A_SUBJECT`, so `say`, `tell`, `show` and
`explain` were each read as the subject of the question. Adding them took this set
from 69 right to 87 with no new wrong answers and left the hand-written benchmark
unchanged - which is the whole argument for a population nobody chose: every
hand-written answerable question was phrased to be answered, so the commonest phrasing
a reader actually uses was the one nothing tested.

Nine refusals remain, and a refusal costs a reader a lookup where a wrong citation
costs them the argument, so that is the right side to err on.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

# The question sets live in the test module beside this one, which is not importable
# by name unless its directory is on the path - the same two lines `bench.py` uses.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_retrieval_quality import _corpus_rows  # noqa: E402

from paklaw.answer import LawAssistant  # noqa: E402
from paklaw.ingest import build_checked  # noqa: E402

#: Words carried by nearly every provision, which make a window about nothing.
_COMMON = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "or",
    "and",
    "shall",
    "be",
    "is",
    "with",
    "for",
    "by",
    "which",
    "any",
    "as",
    "that",
    "this",
    "it",
    "not",
    "he",
    "his",
}

#: The measurement, pinned. `right + wrong + refused` is asserted against the count of
#: questions, so none of the three can move without one of the others.
#:
#: 121, not 101: the population is four questions per provision and the corpus went
#: from 26 provisions to 31 when the importer stopped folding ss.58-66 into s.57. It
#: was 87 right, 5 wrong, 9 refused over 101 - 86% right; it is 110 / 3 / 8 over 121,
#: which is 91%, on a strictly larger set of questions.
EXPECTED = {"asked": 121, "right": 110, "wrong": 3, "refused": 8}

#: Of the three, the ones where the question's every word is in the cited provision as
#: well as in the expected one. All of them, now: s.54 and s.55 commute death and
#: imprisonment for life and share the Provincial Government proviso almost verbatim,
#: and s.64 and s.65 are both about imprisonment for non-payment of a fine. Named,
#: because "all three are ambiguous" is a claim and
#: `test_the_ambiguous_mistakes_really_are_ambiguous` is what makes it checkable.
AMBIGUOUS = 3


@pytest.fixture(scope="module")
def corpus():
    return build_checked(_corpus_rows(), source="fixtures")


@pytest.fixture(scope="module")
def assistant(corpus):
    return LawAssistant(corpus=corpus)


def _questions_for(provision) -> list[str]:
    out = [f"what does the law say about {provision.heading.lower()}?"]
    words = [w for w in provision.text.lower().replace(",", " ").split() if w not in _COMMON]
    for size in (4, 6, 8):
        if len(words) > size:
            start = random.Random(f"{provision.key}{size}").randrange(0, len(words) - size)
            out.append(" ".join(words[start : start + size]))
    return out


def _measure(corpus, assistant):
    random.seed(11)
    right = wrong = refused = 0
    mistakes = []
    seen = set()
    asked = 0
    for provision in corpus.provisions:
        if provision.key in seen:
            continue
        seen.add(provision.key)
        expected = provision.citation().pretty()
        for question in _questions_for(provision):
            asked += 1
            answer = assistant.answer(question, as_of="2026-01-01")
            if answer.refused:
                refused += 1
            elif answer.passages[0].citation == expected:
                right += 1
            else:
                wrong += 1
                mistakes.append((question, expected, answer.passages[0].citation))
    return {"asked": asked, "right": right, "wrong": wrong, "refused": refused}, mistakes


def test_the_generated_population_is_the_size_it_says(corpus, assistant):
    measured, _ = _measure(corpus, assistant)
    assert measured["asked"] == EXPECTED["asked"], measured
    assert measured["right"] + measured["wrong"] + measured["refused"] == measured["asked"]


def test_the_figures_are_the_ones_the_docstring_states(corpus, assistant):
    """Pinned as counts, not as a band. A band here would absorb the thing this is
    for: a change that answers two more questions wrongly and refuses two fewer."""
    measured, _ = _measure(corpus, assistant)
    assert measured == EXPECTED, measured


def test_the_ambiguous_mistakes_really_are_ambiguous(corpus, assistant):
    """All three counted as mistakes are not; it was three of five.

    A window of four words drawn from s.55 can be a true description of s.54 - both
    are about commutation by the provincial government - so the generator's
    expectation is arbitrary. Asserted rather than asserted-in-prose: a question whose
    every word appears in the cited provision too is one this test cannot judge.

    `AMBIGUOUS == len(mistakes)` is asserted as well as `AMBIGUOUS == both`, because
    "all three" is a stronger claim than "three" and it would otherwise survive a
    fourth, genuine mistake appearing beside them.
    """
    _, mistakes = _measure(corpus, assistant)
    texts = {p.citation().pretty(): p.text.lower() for p in corpus.provisions}

    both = 0
    for question, expected, got in mistakes:
        words = [w for w in question.replace(".", "").split() if w not in _COMMON]
        if all(w in texts.get(got, "") for w in words) and all(
            w in texts.get(expected, "") for w in words
        ):
            both += 1
    assert both == AMBIGUOUS, [m for m in mistakes]
    assert len(mistakes) == AMBIGUOUS, (
        f"a mistake that is not one of the ambiguous pairs has appeared: {[m for m in mistakes]}"
    )


def test_the_refusal_share_is_what_it_says(corpus, assistant):
    """The number this repository could not see.

    It was a quarter, and 26 of those 27 were the heading questions - refused because
    `say`, `tell` and `show` were read as the subject of the question. It is 8 of 121
    now. Bounded rather than pinned on its own: the exact counts are pinned by
    `test_the_figures_are_the_ones_the_docstring_states`, and this says the share is
    small, which is the claim.
    """
    measured, _ = _measure(corpus, assistant)
    share = measured["refused"] / measured["asked"]
    assert 0.03 < share < 0.15, share
