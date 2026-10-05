"""Retrieval measured on real statute text, with questions in a person's words.

Everything else in this suite checks behaviour: that a repealed provision is refused,
that a citation parses, that a range expands. None of it says whether retrieval
*finds the right provision* — and a legal assistant that refuses correctly and
retrieves badly is still useless.

So this builds a corpus out of the real-source fixtures (26 provisions, ~11,000
characters of actual Penal Code and Constitution text, no network) and asks it
questions the way someone would type them. Two numbers come out:

**Answered correctly** — the right provision cited first. This is the number a
retrieval change has to improve.

**Wrongly answered** — a confident answer about the wrong provision. This is the
number that must stay at zero, and it is not the complement of the first: a
refusal is a third outcome and an acceptable one. The whole argument of this
repository is that refusing beats guessing, which means a change that lifts
accuracy by turning refusals into wrong answers has made the product worse.

The thresholds below are floors, not targets. They are set under the measured
values so that a regression fails and an improvement does not have to edit them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paklaw.answer import LawAssistant
from paklaw.ingest import build_checked
from paklaw.sources import pakistan_code, pakistani_org, strip_stars
from paklaw.split import split_act

FIXTURES = Path(__file__).parent / "fixtures"

#: Questions in the words someone would actually use, and the provision that answers
#: each. Where the question deliberately avoids the statute's own vocabulary
#: ("murder" for qatl-i-amd, "death sentence" for "sentence of death") that is the
#: point: a person asking about Pakistani law does not know the Arabic terms of art.
ANSWERABLE: list[tuple[str, str]] = [
    ("what is the punishment for murder?", "Section 302 PPC"),
    ("murder committed under duress", "Section 303 PPC"),
    ("killing a person other than the one intended", "Section 301 PPC"),
    ("can a sentence of death be commuted?", "Section 54 PPC"),
    ("commutation of a sentence of imprisonment for life", "Section 55 PPC"),
    ("what does the word animal mean in the Penal Code?", "Section 47 PPC"),
    ("what does the word vessel denote?", "Section 48 PPC"),
    ("definition of good faith", "Section 52 PPC"),
    ("does an oath include a solemn affirmation?", "Section 51 PPC"),
    ("what punishments are offenders liable to?", "Section 53 PPC"),
    ("how is imprisonment for life reckoned in fractions of punishment?", "Section 57 PPC"),
    ("is Islam the state religion?", "Article 2 CONST"),
    ("what is the official name of the country?", "Article 1 CONST"),
    ("high treason for subverting the Constitution", "Article 6 CONST"),
    ("a law inconsistent with fundamental rights", "Article 8 CONST"),
    ("elimination of exploitation by the State", "Article 3 CONST"),
    ("loyalty to the State is the duty of every citizen", "Article 5 CONST"),
]

#: Nothing in this corpus supports an answer. Each must be refused rather than
#: answered from the nearest-scoring provision, which is the failure the whole
#: design exists to prevent.
UNANSWERABLE: list[str] = [
    "what are the rules on cryptocurrency exchange licensing?",
    "how do I register a private limited company?",
    "what is the limitation period for a civil suit?",
    "what are the visa requirements for Pakistan?",
    "what is the rate of sales tax on services?",
    "what notice must a landlord give before eviction?",
]

#: Measured, with the coordination factor in retrieve.py: 13 right, 0 wrong, 4
#: refused of 17, and 6 of 6 refused when the corpus cannot answer. Before it, 12
#: right and 1 wrong - "how is imprisonment for life reckoned in fractions of
#: punishment?" cited s.53 Punishments instead of s.57 Fractions of terms of
#: punishment, which contains every word of the question.
#:
#: The four refusals are honest misses, not bugs: each asks in words the statute
#: never uses ("the official name of the country" against "shall be known as the
#: Islamic Republic of Pakistan"), and lexical retrieval cannot bridge them. They
#: are left in rather than deleted, because a benchmark trimmed to what already
#: passes measures nothing - and because refusing is the correct behaviour when the
#: match is weak.
MIN_ANSWERED = 13  # of 17, measured
MAX_WRONG = 0  # a confident answer about the wrong provision
MIN_REFUSED_WHEN_UNANSWERABLE = 6  # of 6


def _corpus_rows() -> list[dict]:
    """The three real-source fixtures, parsed into provisions."""
    read = lambda name: (FIXTURES / name).read_text(encoding="utf-8")  # noqa: E731

    rows: list[dict] = []
    text, _ = pakistani_org(read("ppc_pakistani_org_ss300-303.html"))
    parsed, _ = split_act(text, statute="PPC", in_force_from="2016-01-01")
    rows += parsed

    source = pakistan_code(read("ppc_pakistan_code_pp39-41.txt"))
    parsed, _ = split_act(strip_stars(source.text), statute="PPC", in_force_from="1860-10-06")
    rows += parsed

    source = pakistan_code(read("constitution_pakistan_code_pp18-20.txt"))
    parsed, _ = split_act(
        strip_stars(source.text), statute="CONST", in_force_from="1973-08-14", unit="article"
    )
    rows += parsed
    return rows


@pytest.fixture(scope="module")
def assistant() -> LawAssistant:
    return LawAssistant(corpus=build_checked(_corpus_rows(), source="fixtures"))


@pytest.fixture(scope="module")
def outcomes(assistant: LawAssistant) -> dict[str, list]:
    """Every answerable question, sorted into right, wrong and refused."""
    right, wrong, refused = [], [], []
    for question, expected in ANSWERABLE:
        answer = assistant.answer(question, as_of="2026-01-01")
        if answer.refused:
            refused.append((question, expected))
        elif answer.passages[0].citation == expected:
            right.append((question, expected))
        else:
            wrong.append((question, expected, answer.passages[0].citation))
    return {"right": right, "wrong": wrong, "refused": refused}


def test_the_corpus_is_the_real_statute_text():
    rows = _corpus_rows()
    assert len(rows) == 26
    assert sum(len(r["text"]) for r in rows) > 10_000
    # Both Acts, and the provisions a question below depends on.
    numbers = {(r["statute"], r["number"]) for r in rows}
    assert ("PPC", "302") in numbers and ("CONST", "6") in numbers


def test_the_right_provision_is_cited_first(outcomes):
    answered = len(outcomes["right"])
    assert answered >= MIN_ANSWERED, (
        f"{answered}/{len(ANSWERABLE)} answered correctly; floor is {MIN_ANSWERED}. "
        f"Refused: {[q for q, _ in outcomes['refused']]}"
    )


def test_no_question_is_answered_from_the_wrong_provision(outcomes):
    """The number that must stay at zero.

    A wrong citation is worse than a refusal here, and a retrieval change that
    raises accuracy by converting refusals into wrong answers has made the
    product worse while making its headline number look better.
    """
    assert len(outcomes["wrong"]) <= MAX_WRONG, "answered about the wrong provision: " + "; ".join(
        f"{q!r} -> {got} (wanted {want})" for q, want, got in outcomes["wrong"]
    )


def test_questions_this_corpus_cannot_answer_are_refused(assistant):
    refused = [q for q in UNANSWERABLE if assistant.answer(q, as_of="2026-01-01").refused]
    answered = [q for q in UNANSWERABLE if q not in refused]
    assert len(refused) >= MIN_REFUSED_WHEN_UNANSWERABLE, f"answered anyway: {answered}"


def test_the_statute_vocabulary_bridge_earns_its_place(assistant):
    """The English word reaches the provision the statute names in Urdu.

    Without the bridge this question is refused while "punishment for
    qatl-i-amd" is answered - the corpus holding the answer the whole time.
    """
    answer = assistant.answer("what is the punishment for murder?", as_of="2026-01-01")
    assert not answer.refused
    assert answer.passages[0].citation == "Section 302 PPC"
    assert "qatl" in answer.passages[0].text.lower()


def test_a_cited_provision_is_a_lookup_not_a_search(assistant):
    """A question naming a provision gets exactly that provision."""
    for question, expected in [
        ("what does section 47 PPC say?", "Section 47 PPC"),
        ("Article 6 of the Constitution", "Article 6 CONST"),
    ]:
        answer = assistant.answer(question, as_of="2026-01-01")
        assert not answer.refused, question
        assert answer.passages[0].citation == expected
