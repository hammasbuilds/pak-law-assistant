"""The safeguard against citing a provision that is really nine provisions.

`answer.py` holds about fifty lines for one case: a provision whose text runs on into
later provisions, because it matches almost any question about its subject - nine
headings' worth of words, and coverage is the first sort key - and the citation it would
be served under names the wrong section. It either attaches a caveat or withholds the
hit, and withholding the only hit is the one thing `citation_unreliable` is true of.

**None of it was executed by the suite.** The test asserting the status is only
reachable when true read `inspect.getsource(module)` and matched strings in it, and said
why: *"the state it needs - a corpus whose provisions run on into each other - is the one
the importer now parses correctly, so there is no fixture here that reaches it."*

The importer is not the only way to build a `Corpus`. `Corpus(provisions=[...])` takes
the provisions it is given, which is what a corpus built by any other tool looks like -
and that corpus is exactly the one this code exists for, since a corpus imported by this
package's own splitter is the one case where the problem has already been solved.

So the fifty lines are driven here, both branches, and the strings they put in front of
a reader are asserted as strings a reader gets rather than as source code.
"""

from __future__ import annotations

import datetime as dt

import pytest

from paklaw.answer import LawAssistant
from paklaw.corpus import Corpus, Provision

#: s.57's real shape in a badly split corpus: its own rule about fractions of terms,
#: and then the next sections' headings and text, with nothing separating them.
BURIED_TEXT = (
    "In calculating fractions of terms of punishment, imprisonment for life shall be "
    "reckoned as equivalent to imprisonment for twenty-five years. "
    "58. Offenders sentenced to transportation how dealt with. In every case in which "
    "a sentence of transportation is passed, the offender shall be dealt with in the "
    "same manner as if sentenced to rigorous imprisonment. "
    "59. Transportation instead of imprisonment. In every case in which an offender is "
    "punishable with imprisonment for a term of seven years or upwards, it shall be "
    "competent to the Court to sentence the offender to transportation for a term not "
    "less than seven years. "
    "60. Sentence may be wholly or partly rigorous or simple. In every case in which an "
    "offender is punishable with imprisonment, the Court may direct that such "
    "imprisonment shall be wholly rigorous, or that such imprisonment shall be wholly "
    "simple."
)


def _provision(number: str, heading: str, text: str) -> Provision:
    return Provision(
        statute="PPC",
        unit="section",
        number=number,
        heading=heading,
        text=text,
        in_force_from=dt.date(1860, 1, 1),
    )


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    """A corpus with one provision holding three others, which is what a corpus
    imported by some other tool looks like."""
    built = Corpus(
        provisions=[
            _provision(
                "57",
                "Fractions of terms of punishment",
                BURIED_TEXT,
            ),
            _provision(
                "302",
                "Punishment of qatl-i-amd",
                "Whoever commits qatl-i-amd shall be punished with death or "
                "imprisonment for life as ta'zir.",
            ),
        ],
        as_at=dt.date(2026, 1, 1),
    )
    return built


@pytest.fixture(scope="module")
def assistant(corpus: Corpus) -> LawAssistant:
    return LawAssistant(corpus=corpus)


def test_the_premise_this_file_rests_on(corpus: Corpus):
    """If this is empty the whole path below is skipped and the tests pass vacuously.

    Which is how the path came to ship untested: the only corpus in the suite had
    `swallowed_headings() == {}`, so every assertion about this code was an assertion
    about nothing.
    """
    assert corpus.swallowed_headings() == {"PPC:section:57": ["58", "59", "60"]}


def test_a_question_the_provisions_own_words_answer_gets_a_caveat_not_a_refusal(
    assistant: LawAssistant,
):
    """s.57's own first sentence is about fractions of terms, so the citation is right
    and the only problem is that the text served is too long. A caveat is the right
    size of response, and withholding the answer would be the wrong one."""
    answer = assistant.answer(
        "how is imprisonment for life reckoned in calculating fractions of terms?",
        as_of="2026-01-01",
    )
    assert not answer.refused, (answer.refusal_status, answer.refusal_reason)
    assert answer.passages[0].citation == "Section 57 PPC"
    assert any("runs on into" in w for w in answer.warnings), answer.warnings
    assert any("longer than one provision" in w for w in answer.warnings), answer.warnings


def test_a_question_only_the_buried_text_answers_is_refused_rather_than_miscited(
    assistant: LawAssistant,
):
    """The harm this system exists to prevent, in its purest form.

    "Wholly rigorous or simple" is s.60. In this corpus those words are inside s.57's
    text, so the hit would be served as "Section 57 PPC" - and a reader filing that has
    filed the wrong section. The hit is withheld, and because it was the only one, the
    refusal is `citation_unreliable`, which here is a true statement.
    """
    answer = assistant.answer(
        "may a sentence be wholly rigorous or wholly simple?", as_of="2026-01-01"
    )
    assert answer.refused, [p.citation for p in answer.passages]
    assert answer.refusal_status == "citation_unreliable", answer.refusal_status
    assert "inside another provision's text" in answer.refusal_reason
    assert any("would name the wrong one" in w for w in answer.warnings), answer.warnings
    assert any("Section 57 PPC" in w for w in answer.warnings), answer.warnings


def test_the_warning_names_the_provisions_that_are_buried(assistant: LawAssistant):
    """A reader has to be able to go and read the right one, so the numbers are in the
    message rather than a count of them."""
    answer = assistant.answer(
        "may a sentence be wholly rigorous or wholly simple?", as_of="2026-01-01"
    )
    text = " ".join(answer.warnings)
    assert "sections 58, 59, 60" in text, text


def test_an_unaffected_provision_in_the_same_corpus_answers_normally(
    assistant: LawAssistant,
):
    """The removal is per hit. A corpus with one bad provision still answers about its
    good ones, with no caveat and no refusal."""
    answer = assistant.answer("what is the punishment for qatl-i-amd?", as_of="2026-01-01")
    assert not answer.refused, (answer.refusal_status, answer.refusal_reason)
    assert answer.passages[0].citation == "Section 302 PPC"
    assert not any("runs on into" in w for w in answer.warnings), answer.warnings


def test_the_status_is_not_returned_by_a_corpus_with_nothing_buried():
    """The defect that put this status on eight answers out of thirty-one: it was the
    catch-all for an empty `kept`, however it emptied.

    Driven over a corpus that has nothing buried in it, which is the condition under
    which the sentence `citation_unreliable` states is false of every provision.
    """
    clean = Corpus(
        provisions=[
            _provision(
                "302",
                "Punishment of qatl-i-amd",
                "Whoever commits qatl-i-amd shall be punished with death.",
            )
        ],
        as_at=dt.date(2026, 1, 1),
    )
    assert clean.swallowed_headings() == {}
    assistant = LawAssistant(corpus=clean)
    statuses = set()
    for question in (
        "what is the punishment for qatl-i-amd?",
        "what is the rate of sales tax on imported machinery?",
        "how are fractions of terms of punishment calculated?",
        "what counts as a vessel?",
        "may a sentence be wholly rigorous?",
    ):
        answer = assistant.answer(question, as_of="2026-01-01")
        if answer.refused:
            statuses.add(answer.refusal_status)
    assert "citation_unreliable" not in statuses, statuses
