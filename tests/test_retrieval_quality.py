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

import datetime as dt
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

#: Measured: 16 right, 0 wrong, 1 refused of 17, and 6 of 6 refused when the corpus
#: cannot answer. Three changes got it there from 12 right / 1 wrong, and each was
#: made because this file reported the failure:
#:
#:   coordination factor   s.53 Punishments (385 chars, 3 of 5 terms) outranked s.57
#:                         Fractions of terms of punishment (2,602 chars, 5 of 5),
#:                         because BM25 length normalisation taxed the longer
#:                         provision harder than two extra terms rewarded it.
#:   statute scoping       "what does the word animal mean in the Penal Code?" was
#:                         refused for missing "penal" and "code" - words in no
#:                         provision's text, because they name the book rather than
#:                         anything in it.
#:   statute vocabulary    "duress" never reached ikrah; "killing" never reached
#:                         "causing death".
#:
#: The one refusal left is an honest miss: "the official name of the country" against
#: "shall be known as the Islamic Republic of Pakistan" shares no word with it, and
#: lexical retrieval cannot bridge that. It stays in rather than being deleted,
#: because a benchmark trimmed to what already passes measures nothing - and because
#: refusing is the correct behaviour when the match is weak.
MIN_ANSWERED = 16  # of 17, measured
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


def test_naming_an_act_scopes_the_question_instead_of_failing_it(assistant):
    """The words that name a statute are not words to match inside it.

    "what does the word animal mean in the Penal Code?" was refused for missing
    "penal" and "code" — which appear in no provision's text, because they are the
    name of the book the provisions are in.
    """
    answer = assistant.answer(
        "what does the word animal mean in the Penal Code?", as_of="2026-01-01"
    )
    assert not answer.refused, answer.refusal_reason
    assert answer.passages[0].citation == "Section 47 PPC"


def test_scoping_restricts_the_search_to_the_named_act(assistant):
    """Naming an Act must not just drop words - it must narrow where to look.

    Unscoped, this is answered from the Constitution. Scoped to the Penal Code it
    must refuse rather than reach for the next-best thing, because a question about
    the Penal Code answered from the Constitution is an answer about the wrong law.
    """
    everywhere = assistant.answer("is Islam the state religion?", as_of="2026-01-01")
    assert not everywhere.refused
    assert everywhere.passages[0].statute == "CONST"

    scoped = assistant.answer(
        "is Islam the state religion under the Penal Code?", as_of="2026-01-01"
    )
    assert scoped.refused or scoped.passages[0].statute == "PPC"


def test_an_explicit_statute_argument_still_wins(assistant):
    """The caller knows the context; a name in the text must not override it."""
    answer = assistant.answer(
        "what does the word animal mean in the Penal Code?",
        as_of="2026-01-01",
        statute="CONST",
    )
    # Scoped to the Constitution by the caller, this cannot be answered at all, and
    # answering it from the PPC because the text said "Penal Code" would be the
    # argument being silently ignored.
    assert answer.refused or answer.passages[0].statute == "CONST"


#: Questions an independent review used to get a confident wrong answer, kept as part of
#: the benchmark so each stays fixed. The first group must refuse: the qualifier names a
#: provision this corpus does not hold, and answering from the neighbouring offence is
#: how "attempt to murder" returned the death penalty.
MUST_REFUSE_QUALIFIED = [
    "what is the punishment for attempt to murder?",
    "what is the punishment for abetment of murder?",
    "punishment for conspiracy to murder",
]


def test_a_neighbouring_offence_is_refused_not_offered(assistant):
    for question in MUST_REFUSE_QUALIFIED:
        answer = assistant.answer(question, as_of="2026-01-01")
        assert answer.refused, f"{question!r} answered with {answer.passages[0].citation}"
        assert answer.refusal_status == "different_offence"


def test_a_rephrasing_of_the_same_question_still_finds_the_same_provision(assistant):
    """The coordination factor was built for one phrasing; this is another.

    "how is imprisonment for life reckoned in fractions of punishment?" borrows s.57's
    heading words, so it was a weak test of the fix. This one shares none of them and
    went to s.55 "Commutation of sentence of imprisonment for life" until coverage was
    made to dominate the ranking.
    """
    answer = assistant.answer(
        "imprisonment for life is reckoned as how many years", as_of="2026-01-01"
    )
    assert not answer.refused, answer.refusal_reason
    assert answer.passages[0].citation == "Section 57 PPC"


def test_the_provision_headed_with_the_question_wins_a_tie(assistant):
    """Asked "what is the State?", it answered from Article 2 CONST.

    Article 2 is "Islam to be State religion"; Article 7 is headed "Definition of the
    State". Both contain the word, both cover the question's one content word
    completely, and Article 2 is shorter — so BM25 preferred it and coverage could not
    separate them. Among provisions that cover a question equally, the one the
    draftsman headed with those words is the one about them.
    """
    answer = assistant.answer("what is the State?", as_of="2026-01-01")
    assert not answer.refused, answer.refusal_reason
    assert answer.passages[0].citation == "Article 7 CONST"


def test_the_heading_tiebreak_cannot_overturn_coverage():
    """Which is why it is a tiebreak. As a multiplier it broke the fractions question.

    s.55 "Commutation of sentence of imprisonment for life" matches two of that
    question's heading words and s.57 "Fractions of terms of punishment" matches none —
    so a heading multiplier large enough to fix "what is the State?" also reversed the
    coverage difference, 0.80 against 0.60, that was the whole point of the earlier fix.
    Coverage is the primary key, so no heading weight can do that.
    """
    corpus = build_checked(_corpus_rows(), source="fixtures")
    question = "imprisonment for life is reckoned as how many years"
    for heading_weight in (0.0, 1.5, 5.0, 50.0):
        assistant = LawAssistant(corpus=corpus, tuning={"heading_weight": heading_weight})
        answer = assistant.answer(question, as_of="2026-01-01")
        assert not answer.refused, heading_weight
        assert answer.passages[0].citation == "Section 57 PPC", heading_weight


def test_the_ranking_parameters_are_reachable_without_editing_the_source():
    """A sweep that sets them on the class measures nothing, and one did.

    `BM25Index` is a dataclass, so its defaults are captured in `__init__` and
    `BM25Index.b = 0.1` does not change a new instance. A sweep written that way
    reported that `b` made no difference across its whole range — true of the
    experiment, and nothing to do with `b`.
    """
    corpus = build_checked(_corpus_rows(), source="fixtures")
    default = LawAssistant(corpus=corpus)
    tuned = LawAssistant(corpus=corpus, tuning={"heading_weight": 0.0})
    question = "what is the State?"
    # The parameter reaches the index: at weight 0 the heading cannot break the tie and
    # the answer changes.
    assert default.answer(question, as_of="2026-01-01").passages[0].citation == "Article 7 CONST"
    assert tuned.answer(question, as_of="2026-01-01").passages[0].citation != "Article 7 CONST"


# --- the same questions, asked differently ---------------------------------------------
#
# An independent review's finding, and it was right: 17 sentences with MAX_WRONG = 0 is
# a claim about 17 sentences. It re-asked the same corpus in its own words and got 8
# confident wrong answers, including s.302 - "punished with death as qisas" - for
# qatl-i-khata, which is s.322 and punishable by diyat.
#
# Two phrasings per provision, written against the provisions rather than against the
# retriever, plus the offences a reader would most plausibly confuse with the ones that
# are loaded. The must-refuse list is the half that matters: this corpus holds thirteen
# sections of the Penal Code and seven Articles, so every offence below has a plausible
# neighbour here, and answering from the neighbour is the failure the repository exists
# to prevent.

PARAPHRASES: list[tuple[str, str]] = [
    ("what sentence does a murderer get?", "Section 302 PPC"),
    ("punishment for qatl-i-amd", "Section 302 PPC"),
    ("what happens to someone who kills under ikrah?", "Section 303 PPC"),
    ("qatl committed under compulsion", "Section 303 PPC"),
    ("he meant to kill one man and killed another", "Section 301 PPC"),
    ("causing the death of a person whose death was not intended", "Section 301 PPC"),
    ("can the government commute a death sentence?", "Section 54 PPC"),
    ("may a sentence of death be changed to something else?", "Section 54 PPC"),
    ("can life imprisonment be commuted?", "Section 55 PPC"),
    ("commuting a life sentence to a term of years", "Section 55 PPC"),
    ("does animal include every living creature?", "Section 47 PPC"),
    ("the meaning of animal in this Code", "Section 47 PPC"),
    ("what counts as a vessel?", "Section 48 PPC"),
    ("the meaning of the word vessel", "Section 48 PPC"),
    ("what is good faith?", "Section 52 PPC"),
    ("when is something done in good faith?", "Section 52 PPC"),
    ("is a solemn affirmation an oath?", "Section 51 PPC"),
    ("the meaning of oath in this Code", "Section 51 PPC"),
    ("what penalties does the Code provide?", "Section 53 PPC"),
    ("list of punishments under the Penal Code", "Section 53 PPC"),
    ("imprisonment for life is equivalent to how long?", "Section 57 PPC"),
    ("fractions of terms of punishment", "Section 57 PPC"),
    ("what is the state religion of Pakistan?", "Article 2 CONST"),
    ("is Pakistan an Islamic state by its Constitution?", "Article 2 CONST"),
    ("what shall Pakistan be known as?", "Article 1 CONST"),
    ("the Republic and its territories", "Article 1 CONST"),
    ("what is high treason?", "Article 6 CONST"),
    ("abrogating the Constitution by force", "Article 6 CONST"),
    ("a law repugnant to fundamental rights is void", "Article 8 CONST"),
    ("laws inconsistent with the rights conferred by this Chapter", "Article 8 CONST"),
    ("the State shall eliminate exploitation", "Article 3 CONST"),
    ("elimination of all forms of exploitation", "Article 3 CONST"),
    ("the basic duty of every citizen", "Article 5 CONST"),
    ("obedience to the Constitution and law", "Article 5 CONST"),
]

#: Real offences with no provision in this corpus, each a near neighbour of one that is
#: here. Several were answered confidently before: qatl-i-khata and attempt to murder
#: both returned s.302.
MUST_REFUSE: list[str] = [
    "what is the punishment for culpable homicide not amounting to murder?",
    "what is the punishment for qatl-i-khata?",
    "what is the punishment for attempt to murder?",
    "what is the sentence for dacoity with murder?",
    "what is the punishment for theft?",
    "what is the punishment for robbery?",
    "what is the punishment for rape?",
    "what is the punishment for kidnapping?",
    "what is the punishment for criminal breach of trust?",
    "what is the punishment for qatl shibh-i-amd?",
]

#: Measured, not chosen: 26 right, 0 wrong, 8 refused of 34, and 10 of 10 refused.
#:
#: It was 25/1/8. The one wrong answer was "is Pakistan an Islamic state by its
#: Constitution?" returning Article 1, and it was two gaps rather than a ranking
#: failure: "its" was a content word while "it" was a stopword, and "Islamic" did not
#: stem to "Islam", so Article 2 - "Islam shall be the State religion" - was missing the
#: word the question was about while Article 1, which carries "Islamic" inside a name,
#: was not. The budget is zero again rather than one, because a budget nobody is using
#: is the only kind worth keeping.
MIN_PARAPHRASE_RIGHT = 26
MAX_PARAPHRASE_WRONG = 0

#: Wrong answers that are known and accepted. Empty, and a count on its own would let a
#: new wrong answer in as soon as an old one was fixed - which is how a budget becomes a
#: ratchet - so the test names them as well as counting them.
KNOWN_WRONG: dict[str, str] = {}


def test_the_same_questions_asked_differently(assistant):
    """Recall is allowed to fall on a paraphrase. Precision is not."""
    right, wrong, refused = 0, [], []
    for question, citation in PARAPHRASES:
        answer = assistant.answer(question, as_of="2026-01-01")
        if answer.refused:
            refused.append(question)
        elif answer.passages[0].citation == citation:
            right += 1
        else:
            wrong.append((question, answer.passages[0].citation))

    assert right >= MIN_PARAPHRASE_RIGHT, f"right {right}; refused {refused}; wrong {wrong}"
    assert len(wrong) <= MAX_PARAPHRASE_WRONG, wrong
    for question, got in wrong:
        assert KNOWN_WRONG.get(question) == got, f"a NEW wrong answer: {question} -> {got}"


def test_an_offence_this_corpus_does_not_hold_is_refused(assistant):
    """The half that matters. Each of these has a plausible neighbour loaded, and
    answering from the neighbour is a citation a lawyer would act on."""
    answered = {}
    for question in MUST_REFUSE:
        answer = assistant.answer(question, as_of="2026-01-01")
        if not answer.refused:
            answered[question] = answer.passages[0].citation
    assert answered == {}, answered


def test_the_species_of_qatl_are_not_synonyms():
    """The vocabulary bridge, crossed twice, made them one word.

    `khata` reached `manslaughter` and `manslaughter` reached `qatl`, so the single
    term separating s.322 (diyat) from s.302 (death as qisas) matched s.302.
    """
    from paklaw.retrieve import expand

    assert "qatl" not in expand("khata")
    assert "khata" not in expand("amd")
    assert "amd" not in expand("khata")
    # The bridge itself still works in the direction it was built for.
    assert "qatl" in expand("murder")
    assert "ikrah" in expand("duress")


@pytest.mark.parametrize(
    "asked,in_the_statute",
    [
        ("denote", "denotes"),  # the question nominalises, the statute conjugates
        ("transmitting", "transmits"),  # English doubles the consonant, the statute does not
        ("elimination", "eliminate"),
        ("penalties", "penalty"),
        ("murderer", "murder"),
        ("sections", "section"),
        ("punishments", "punished"),
        ("kills", "killing"),
    ],
)
def test_the_two_spellings_of_one_word_meet(asked, in_the_statute):
    """Each pair was a question this corpus could answer and did not."""
    from paklaw.retrieve import _stems

    assert _stems(asked) & _stems(in_the_statute), (asked, in_the_statute)


def test_a_word_the_corpus_has_never_seen_is_not_a_near_miss(assistant):
    """ "dacoity" is absent from thirteen sections of the Penal Code. Two of the three
    words in "the sentence for dacoity with murder" were covered, so the gate opened
    and s.302 came back - a confident citation for an offence the corpus does not hold.
    """
    answer = assistant.answer("what is the sentence for dacoity with murder?", as_of="2026-01-01")
    assert answer.refused
    assert answer.refusal_status == "subject_not_in_corpus"
    assert "dacoity" in answer.refusal_reason


def test_an_ordinary_english_word_is_not_treated_as_the_subject(assistant):
    """The same rule, not firing. A statute book contains no "many" and no "count",
    and s.57 answers this exactly."""
    answer = assistant.answer(
        "how many years does imprisonment for life count as?", as_of="2026-01-01"
    )
    assert not answer.refused, answer.refusal_reason


def test_a_query_term_is_expanded_once_not_once_per_provision(monkeypatch):
    """`_forms` sat inside the per-document loop and its value depends only on the term.

    One five-word question over 10,000 provisions called it 99,174 times and spent four
    fifths of the query there - 0.78s per answer, on the headline tool. Counting calls
    rather than timing them, because a clock on CI measures the CI box.
    """
    import paklaw.retrieve as retrieve

    corpus = build_checked(_corpus_rows(), source="fixtures")
    law = LawAssistant(corpus=corpus)

    calls: list[str] = []
    real = retrieve.expand
    monkeypatch.setattr(retrieve, "expand", lambda term: calls.append(term) or real(term))

    question = "what is the punishment for murder committed under duress"
    law.answer(question, as_of="2026-01-01")

    provisions = len(corpus.as_of(dt.date(2026, 1, 1)))
    assert provisions > 10, provisions
    # Once per distinct term, not once per (term x provision). The margin is for the
    # heading pass, which asks about the same terms again.
    assert len(calls) <= 3 * len(set(calls)), (len(calls), len(set(calls)))
    assert len(calls) < provisions, (len(calls), provisions)


def test_a_provision_that_runs_on_into_others_says_so_in_the_answer(assistant):
    """s.57 holds s.58 to s.66, because the contents stop before the text does.

    The importer reports that when a corpus is built. Nothing reported it when one was
    served - and query time is where it bites: nine headings' worth of words match
    almost any question about punishment, and coverage is the first sort key, so the
    malformed provision is the one most likely to be returned.
    """
    answer = assistant.answer(
        "how is imprisonment for life reckoned in fractions of punishment?",
        as_of="2026-01-01",
    )
    assert [p.citation for p in answer.passages][0] == "Section 57 PPC"
    warned = [w for w in answer.warnings if "Section 57 PPC" in w]
    assert len(warned) == 1, answer.warnings
    for number in ("58", "66"):
        assert number in warned[0]


def test_a_well_formed_answer_carries_no_such_warning(assistant):
    """It has to be quiet on the provisions that are fine, or it is noise on every
    answer."""
    answer = assistant.answer("what is the punishment for murder?", as_of="2026-01-01")
    assert [p.citation for p in answer.passages][0] == "Section 302 PPC"
    assert [w for w in answer.warnings if "runs on into" in w] == []


def test_the_corpus_names_what_it_is_known_to_get_wrong(assistant):
    corpus = assistant.corpus
    assert corpus.swallowed_headings() == {
        "PPC:section:57": ["58", "59", "60", "61", "62", "63", "64", "65", "66"]
    }
    # Cached like the index, and invalidated the same way.
    before = corpus.swallowed_headings()
    assert corpus.swallowed_headings() is before


# --- a question about another country's law -------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "what is the punishment for murder under the Indian Penal Code?",
        "what is the punishment for murder in India?",
        "under Indian law, what is the punishment for murder?",
        "is Islam the state religion under the Indian Constitution?",
        "what is high treason under the Indian Constitution?",
        "what does section 302 of the Bangladesh Penal Code say?",
    ],
)
def test_an_act_this_corpus_does_not_hold_is_refused_in_prose_too(assistant, question):
    """The citation route always refused these. The prose route answered them.

    `find_statute` substring-matched "penal code" inside "Indian Penal Code", scoped
    the search to the PPC and stripped the words that said so - discarding "Indian",
    the only word that mattered. The result was a confident, correctly formatted,
    warning-free citation of Pakistani law in answer to a question about another
    country's, produced by the exact string the README uses as its example of what the
    `act_not_recognised` refusal prevents. India matters most: its Penal Code shares
    this one's numbering, so s.302 exists there and says something else.
    """
    answer = assistant.answer(question, as_of="2026-01-01")
    assert answer.refused, [p.citation for p in answer.passages]
    assert answer.refusal_status == "act_not_recognised"


@pytest.mark.parametrize(
    "question,citation",
    [
        ("what does the word animal mean in the Penal Code?", "Section 47 PPC"),
        ("what is the punishment for murder under the Pakistan Penal Code?", "Section 302 PPC"),
        ("what is the punishment for murder?", "Section 302 PPC"),
    ],
)
def test_scoping_to_a_pakistani_act_still_works(assistant, question, citation):
    """The guard must not refuse the questions the scoping exists for."""
    answer = assistant.answer(question, as_of="2026-01-01")
    assert not answer.refused, answer.refusal_reason
    assert answer.passages[0].citation == citation


# --- a citation that would name the wrong section --------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "what is the limit to imprisonment for non-payment of fine?",  # s.65
        "what is the amount of a fine where no sum is expressed?",  # s.63
        "can a court impose simple imprisonment?",  # s.60
    ],
)
def test_an_answer_from_inside_another_provision_is_not_cited(assistant, question):
    """s.57 holds s.58 to s.66, so these were all answered "Section 57 PPC".

    A warning was not enough. The reader acts on the citation, and a lawyer filing
    "Section 57 PPC" for the fine-default rule has filed s.65. The provision is dropped
    when its own text does not support the match, and kept - with a caveat - when it
    does, so the question s.57 really answers is not lost with it.
    """
    answer = assistant.answer(question, as_of="2026-01-01")
    assert answer.refused, [p.citation for p in answer.passages]
    assert answer.refusal_status == "citation_unreliable"
    assert any("wrong one" in w for w in answer.warnings), answer.warnings


def test_the_provision_that_blob_really_is_still_answers(assistant):
    """s.57's own first sentence is the answer to this, and dropping every malformed
    provision outright would have lost it."""
    answer = assistant.answer(
        "how is imprisonment for life reckoned in fractions of punishment?", as_of="2026-01-01"
    )
    assert not answer.refused, answer.refusal_reason
    assert answer.passages[0].citation == "Section 57 PPC"
    assert any("runs on into" in w for w in answer.warnings)


# --- what a refusal says is true -------------------------------------------------------


def test_the_subject_refusal_does_not_assert_something_false(assistant):
    """ "acting" is absent from the corpus; what the question is about is not.

    s.52 PPC reads "…done or believed without due care and attention", so telling a
    reader the corpus has no provision about this question was a wrong statement of
    fact delivered with a refusal's authority. The refusal still fires - it earns its
    keep, by ablation - but it now says what is true: the word appears in no provision.
    """
    answer = assistant.answer("acting without due care and attention", as_of="2026-01-01")
    assert answer.refused
    assert answer.refusal_status == "subject_not_in_corpus"
    assert "'acting' appears in no provision here" in answer.refusal_reason
    assert "has no provision about" not in answer.refusal_reason


def test_an_adverb_does_not_veto_an_answer(assistant):
    answer = assistant.answer("can a sentence of death never be commuted?", as_of="2026-01-01")
    assert not answer.refused, answer.refusal_reason
    assert answer.passages[0].citation == "Section 54 PPC"


# --- the statute's spelling and the reader's -------------------------------------------


@pytest.mark.parametrize("question", ["murder as tazir", "what is the tazir punishment?"])
def test_an_apostrophe_in_a_term_of_art_is_not_a_different_word(assistant, question):
    """The statute writes "ta'zir"; a person types "tazir". The tokeniser kept the
    apostrophe, so they were different words - and because "tazir" is a QUALIFIER, the
    mismatch escalated from a lower score to a hard `different_offence` refusal on a
    question s.302(b) answers literally."""
    answer = assistant.answer(question, as_of="2026-01-01")
    assert not answer.refused, answer.refusal_reason
    assert answer.passages[0].citation == "Section 302 PPC"


# --- a third set, written from the provisions rather than from the retriever ------------
#
# The criticism a benchmark cannot answer about itself is that it was written after the
# fact. ANSWERABLE was written with the retriever; PARAPHRASES was written against the
# provisions after an independent review showed ANSWERABLE was too narrow. This set was
# written a third time, later again, by reading each loaded provision and asking what a
# person would ask about it - in none of the earlier wording, and several as the bare
# phrases people actually type.
#
# Measured when written: 14 right, 1 wrong, 5 refused of 20, and 10 of 10 refused for
# subjects the corpus does not hold. The one wrong is "consent of the heirs of the
# victim", which comes back as s.55A rather than s.54: that phrase appears verbatim in
# the provisos of s.54 and s.55 and in the body of s.55A, so the bare fragment does not
# determine which, and the expectation was more specific than the question.

FRESH: list[tuple[str, str]] = [
    ("if a man is forced to kill, what does the law give him?", "Section 303 PPC"),
    ("twenty-five years for killing under threat", "Section 303 PPC"),
    ("death as qisas", "Section 302 PPC"),
    ("fasad-fil-arz and clause (c)", "Section 302 PPC"),
    ("when is an act said to be qatl-e-amd?", "Section 300 PPC"),
    ("bodily injury likely to cause death in the ordinary course of nature", "Section 300 PPC"),
    ("the victim was not the person he meant to harm", "Section 301 PPC"),
    ("may the Federal Government reduce a death sentence?", "Section 54 PPC"),
    ("consent of the heirs of the victim", "Section 54 PPC"),
    ("fourteen years instead of life", "Section 55 PPC"),
    ("arsh and daman", "Section 53 PPC"),
    ("rigorous and simple imprisonment", "Section 53 PPC"),
    ("twenty-five years equivalent", "Section 57 PPC"),
    ("does the word animal cover a bird?", "Section 47 PPC"),
    ("anything made for conveyance on water", "Section 48 PPC"),
    ("due care and attention", "Section 52 PPC"),
    ("a solemn affirmation substituted by law", "Section 51 PPC"),
    ("the Islamic Republic of Pakistan", "Article 1 CONST"),
    ("exploitation and the gradual fulfilment of the principle", "Article 3 CONST"),
    ("abrogate or subvert the Constitution by use of force", "Article 6 CONST"),
]

#: Subjects this corpus does not hold, written at the same time. Six are offences in the
#: Penal Code that are simply not loaded; four are other bodies of law entirely.
FRESH_MUST_REFUSE: list[str] = [
    "what is the punishment for forgery?",
    "what is the sentence for cheating?",
    "what is the punishment for extortion?",
    "what is the punishment for mischief?",
    "what is the punishment for defamation?",
    "what is the punishment for sedition?",
    "how long do I have to file an appeal?",
    "what are the grounds for divorce under Muslim law?",
    "what is the stamp duty on a sale deed?",
    "what is the punishment for qatl committed by a minor?",
]

MIN_FRESH_RIGHT = 14
MAX_FRESH_WRONG = 1

#: Named, so fixing this one cannot quietly make room for another.
FRESH_KNOWN_WRONG = {"consent of the heirs of the victim": "Section 55A PPC"}


def test_questions_written_after_the_retriever_was_finished(assistant):
    """Precision on wording the retriever was never tuned against."""
    right, wrong, refused = 0, [], []
    for question, citation in FRESH:
        answer = assistant.answer(question, as_of="2026-01-01")
        if answer.refused:
            refused.append(question)
        elif answer.passages[0].citation == citation:
            right += 1
        else:
            wrong.append((question, answer.passages[0].citation))

    assert right >= MIN_FRESH_RIGHT, f"right {right}; refused {refused}; wrong {wrong}"
    assert len(wrong) <= MAX_FRESH_WRONG, wrong
    for question, got in wrong:
        assert FRESH_KNOWN_WRONG.get(question) == got, f"a NEW wrong answer: {question} -> {got}"


def test_subjects_this_corpus_does_not_hold_are_still_refused(assistant):
    """The safety property, on subjects chosen after the rule that enforces it existed.

    Six are offences in the Penal Code that are simply not loaded, so the nearest
    provision is always a neighbour; four are other bodies of law entirely.
    """
    answered = {}
    for question in FRESH_MUST_REFUSE:
        answer = assistant.answer(question, as_of="2026-01-01")
        if not answer.refused:
            answered[question] = answer.passages[0].citation
    assert answered == {}, answered


def test_the_readme_quotes_the_totals_these_sets_actually_produce(assistant):
    """The Limits section states the trade with a number, and the number has to be the
    one the sets produce. It was written as 13 and is 14, which is exactly the kind of
    count that moves when a question is added and nobody re-reads the prose.
    """
    import re

    right = wrong = refused = 0
    for question, citation in ANSWERABLE + PARAPHRASES + FRESH:
        answer = assistant.answer(question, as_of="2026-01-01")
        if answer.refused:
            refused += 1
        elif answer.passages[0].citation == citation:
            right += 1
        else:
            wrong += 1
    total = len(ANSWERABLE) + len(PARAPHRASES) + len(FRESH)
    assert right + wrong + refused == total

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    stated = re.search(
        r"that is (\d+)\s+refusals of (\d+) answerable questions against (\d+) wrong answer",
        readme.replace("\n  ", " "),
    )
    assert stated, "the README no longer states the trade"
    assert (int(stated.group(1)), int(stated.group(2)), int(stated.group(3))) == (
        refused,
        total,
        wrong,
    )

    held = sum(
        1
        for question in MUST_REFUSE + FRESH_MUST_REFUSE
        if assistant.answer(question, as_of="2026-01-01").refused
    )
    assert held == len(MUST_REFUSE) + len(FRESH_MUST_REFUSE)


def test_the_three_question_sets_are_the_sizes_the_readme_describes():
    """So the denominator cannot shrink quietly.

    `test_the_readme_quotes_the_totals_these_sets_actually_produce` recomputes
    right/wrong/refused and checks they sum to the whole population, which stops a
    question being dropped from the count. It does not stop a question being dropped
    from the *set*: delete one and `total` falls with it, the README is updated to
    match, and nothing says the benchmark got smaller.

    The README describes the sets by size and by when they were written. Those sizes
    are the claim, so they are asserted here, and growing a set is a deliberate edit
    to this test rather than a silent one.
    """
    import re
    from pathlib import Path

    assert (len(ANSWERABLE), len(PARAPHRASES), len(FRESH)) == (17, 34, 20)
    assert (len(MUST_REFUSE), len(FRESH_MUST_REFUSE)) == (10, 10)
    assert len(ANSWERABLE) + len(PARAPHRASES) + len(FRESH) == 71

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    described = re.search(
        r"(\d+) in a person's words \(\d+ right\), (\d+) paraphrases of them.*?"
        r"and (\d+) written later again",
        readme.replace("\n", " "),
    )
    assert described, "the README no longer describes the three sets"
    assert tuple(int(g) for g in described.groups()) == (
        len(ANSWERABLE),
        len(PARAPHRASES),
        len(FRESH),
    )

    unheld = re.search(r"\*\*(\d+) subjects the corpus does not hold, all \d+ refused\*\*", readme)
    assert unheld, "the README no longer states the unanswerable count"
    assert int(unheld.group(1)) == len(MUST_REFUSE) + len(FRESH_MUST_REFUSE)


def test_no_question_appears_in_two_sets():
    """Three sets written at three different times, so an overlap is an accident -
    and a duplicated question is counted twice in a total that reads as 71 distinct
    ones."""
    asked = [q for q, _ in ANSWERABLE + PARAPHRASES + FRESH]
    duplicates = {q for q in asked if asked.count(q) > 1}
    assert duplicates == set(), duplicates

    refusable = MUST_REFUSE + FRESH_MUST_REFUSE
    assert len(set(refusable)) == len(refusable)
    assert set(asked).isdisjoint(refusable), "a question cannot be both answerable and not"
