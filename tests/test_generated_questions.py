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

    right 69    wrong 5    refused 27

Three of those five are genuinely ambiguous rather than wrong. Every word of the
question appears in BOTH the expected provision and the one cited - s.54 and s.55 are
both about commutation by the provincial government, and a four-word window drawn from
one is a true description of the other. The expectation is arbitrary there, and
counting it as an error would make this test a measurement of the generator. The two
that remain are in the s.53-s.57 cluster, where the provisions share most of their
vocabulary, and they are recorded as the figure rather than explained away.

The refusals are the interesting half: 27 of 101, a quarter, where a window of a
provision's own words did not reach the gate. A refusal costs a reader a lookup, so
that is the right side to err on - but it is a number this repository had no way to
see before, because every hand-written answerable question was written to be answered.
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
EXPECTED = {"asked": 101, "right": 69, "wrong": 5, "refused": 27}

#: Of the five, the ones where the question's every word is in the cited provision as
#: well as in the expected one. Named, because "three are ambiguous" is a claim and
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
    """Three of the five are counted as mistakes and are not.

    A window of four words drawn from s.55 can be a true description of s.54 - both
    are about commutation by the provincial government - so the generator's
    expectation is arbitrary. Asserted rather than asserted-in-prose: a question whose
    every word appears in the cited provision too is one this test cannot judge.
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


def test_a_quarter_of_the_generated_questions_are_refused(corpus, assistant):
    """The number this repository could not see.

    Every hand-written answerable question was written to be answered, so the share of
    a provision's own words that does NOT reach the gate was invisible. A refusal costs
    a reader a lookup where a wrong citation costs them the argument, so this is the
    right side to err on - and it is a cost, stated.
    """
    measured, _ = _measure(corpus, assistant)
    share = measured["refused"] / measured["asked"]
    assert 0.20 < share < 0.35, share
