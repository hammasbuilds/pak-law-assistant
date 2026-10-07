"""A refusal states a fact, and the fact has to be true of this corpus.

`answer.py` says it of itself, about a different refusal: *"a wrong statement of fact
delivered with a refusal's authority: the same defect class as a wrong citation."* Three
refusals were failing that test.

**`citation_unreliable` as a catch-all.** It means "the provision answering this is
inside another provision's text, so citing it would name the wrong one" - and it was
returned whenever the display filter emptied, for any reason. `Corpus.swallowed_headings()`
is `{}` for this corpus, so nothing is inside anything. All eight refusals in the
121-question generated population carried it, and a sweep of `what counts as <heading>?`
over all 31 provisions refused eight that way: every single-word-heading definition
section, which are the provisions most likely to be looked up.

**A jurisdiction word the corpus itself uses.** PPC s.49 reads "reckoned according to the
British calendar", and `FOREIGN_JURISDICTIONS` holds "british", so a question in the
statute's own words was refused for "naming an Act this corpus does not hold: 'British'".

**A capitalised phrase read as an instrument.** `Order`, `Rules` and `Regulations` are
ordinary English, so "can the Provincial Government pass a Commutation Order?" was
refused for naming an Act called "Commutation Order" - two words, the first of which is
s.54's own heading.

And one that was not false but inconsistent: the guard ran after the citation route, so
"is an Indian citizen protected by Article 4?" was answered while the same sentence
without the citation was refused.
"""

from __future__ import annotations

import pytest

from paklaw.answer import LawAssistant
from paklaw.ingest import build_checked
from tests.test_retrieval_quality import _corpus_rows


@pytest.fixture(scope="module")
def corpus():
    return build_checked(_corpus_rows(), source="fixtures")


@pytest.fixture(scope="module")
def assistant(corpus):
    return LawAssistant(corpus=corpus)


def test_nothing_in_this_corpus_is_buried_in_anything(corpus):
    """The premise. If a provision ever does run on into another, the sweep below is
    measuring a different thing and should be read differently."""
    assert corpus.swallowed_headings() == {}


def test_no_question_about_a_provision_is_refused_for_an_unreliable_citation(corpus, assistant):
    """The sweep that found it: one question per provision, from its own heading.

    Eight of thirty-one came back `citation_unreliable` - every definition section,
    whose whole job is to be looked up - in a corpus where that status cannot be true.
    """
    seen: set[str] = set()
    wrongly_refused = []
    for provision in corpus.provisions:
        if provision.key in seen:
            continue
        seen.add(provision.key)
        question = f"what counts as {provision.heading.strip(chr(34))}?"
        answer = assistant.answer(question, as_of="2026-01-01")
        if answer.refusal_status == "citation_unreliable":
            wrongly_refused.append(question)
    assert wrongly_refused == [], (
        f"{len(wrongly_refused)} question(s) refused with `citation_unreliable` in a "
        f"corpus where nothing runs on into anything: {wrongly_refused}"
    )


def test_the_status_is_only_reachable_when_it_is_true(assistant):
    """Read as code: the branch returning it is the one that removes a buried hit.

    Asserted on the source because the state it needs - a corpus whose provisions run on
    into each other - is the one the importer now parses correctly, so there is no
    fixture here that reaches it. `tests/test_split_real.py` covers the detector.
    """
    import inspect

    from paklaw import answer as module

    source = inspect.getsource(module)
    assert source.count('refusal_status = "citation_unreliable"') == 1
    where = source.index('refusal_status = "citation_unreliable"')
    preceding = source[:where]
    # The only path to it runs through the swallowed-provision removal.
    assert "kept.remove(hit)" in preceding
    assert preceding.rindex("kept.remove(hit)") < where


DEFINITION_QUESTIONS = {
    "what counts as an oath?": "Section 51 PPC",
    'what does the law say about "animal"?': "Section 47 PPC",
    "what counts as a vessel?": "Section 48 PPC",
}


@pytest.mark.parametrize("question", sorted(DEFINITION_QUESTIONS))
def test_a_definition_section_answers_the_question_it_defines(assistant, question: str):
    answer = assistant.answer(question, as_of="2026-01-01")
    assert not answer.refused, (answer.refusal_status, answer.refusal_reason)
    assert answer.passages[0].citation == DEFINITION_QUESTIONS[question], [
        p.citation for p in answer.passages
    ]


# -- a name made of this corpus's own words is not another country's statute ----------

CORPUS_LANGUAGE = {
    # PPC s.49: "reckoned according to the British calendar".
    "is a year reckoned by the British calendar?": "Section 49 PPC",
    # "Commutation" is s.54's own heading; "Order" is ordinary English.
    "What is a High Treason Act?": "Article 6 CONST",
}

STILL_FOREIGN = (
    "what is the punishment for murder under the Indian Penal Code?",
    "what is the punishment for murder in India?",
    "what does section 302 of the Indian Penal Code say?",
    "is an Indian citizen protected by Article 4?",
)


@pytest.mark.parametrize("question", sorted(CORPUS_LANGUAGE))
def test_the_corpus_own_words_do_not_read_as_a_foreign_act(assistant, question: str):
    answer = assistant.answer(question, as_of="2026-01-01")
    assert answer.refusal_status != "act_not_recognised", answer.refusal_reason
    assert not answer.refused, (answer.refusal_status, answer.refusal_reason)
    assert answer.passages[0].citation == CORPUS_LANGUAGE[question]


@pytest.mark.parametrize("question", STILL_FOREIGN)
def test_another_country_statute_is_still_refused(assistant, question: str):
    """The guard has to keep working, or this traded a false refusal for a wrong
    citation about the wrong country's criminal law."""
    answer = assistant.answer(question, as_of="2026-01-01")
    assert answer.refused, [p.citation for p in answer.passages]
    assert answer.refusal_status == "act_not_recognised", answer.refusal_status


def test_naming_a_jurisdiction_is_refused_whether_or_not_a_section_is_cited(assistant):
    """The guard used to run after the citation route, so a section number in the
    sentence decided whether it applied at all."""
    with_citation = assistant.answer(
        "is an Indian citizen protected by Article 4?", as_of="2026-01-01"
    )
    without = assistant.answer("are Indian citizens protected?", as_of="2026-01-01")
    assert with_citation.refusal_status == without.refusal_status == "act_not_recognised"


def test_a_commutation_order_is_not_reported_as_an_unheld_act(assistant):
    """`Order` is ordinary English. Whatever this question gets, the reason must not be
    that the corpus does not hold an Act called "Commutation Order"."""
    answer = assistant.answer(
        "Can the Provincial Government pass a Commutation Order?", as_of="2026-01-01"
    )
    assert answer.refusal_status != "act_not_recognised", answer.refusal_reason
    assert "Commutation Order" not in answer.refusal_reason
