"""Pakistan law assistant tests.

Temporal correctness is exact — a provision either was in force on a date or was not —
so the leakage-equivalent here is asserted rather than sampled for.
"""

from __future__ import annotations

import datetime as dt

import pytest

from paklaw.answer import (
    REFUSAL_NOT_IN_FORCE,
    REFUSAL_UNKNOWN_CITATION,
    LawAssistant,
)
from paklaw.citation import STATUTES, parse, resolve_bare
from paklaw.corpus import Corpus, CorpusError, Provision
from paklaw.retrieve import BM25Index, LawSearch, tokenise

PECA_OLD = (
    "Whoever intentionally and publicly exhibits any information through an information "
    "system which he knows to be false and intimidates or harms the reputation of a "
    "natural person shall be punished with imprisonment which may extend to three years."
)
PECA_NEW = (
    "Whoever intentionally and publicly exhibits any false information through an "
    "information system which harms the reputation of a natural person shall be "
    "punished with imprisonment which may extend to five years."
)


def corpus() -> Corpus:
    c = Corpus()
    c.add(
        Provision(
            statute="PECA",
            unit="section",
            number="20",
            heading="Offences against dignity of a natural person",
            text=PECA_OLD,
            in_force_from="2016-08-19",
            in_force_to="2022-02-20",
            manner="substituted",
            amended_by="Ordinance II of 2022",
            superseded_by="PECA s.20 as amended 2022",
        )
    )
    c.add(
        Provision(
            statute="PECA",
            unit="section",
            number="20",
            heading="Offences against dignity of a natural person",
            text=PECA_NEW,
            in_force_from="2022-02-20",
        )
    )
    c.add(
        Provision(
            statute="CONST",
            unit="article",
            number="25",
            heading="Equality of citizens",
            text="All citizens are equal before law and are entitled to equal protection of "
            "law. There shall be no discrimination on the basis of sex.",
            in_force_from="1973-08-14",
        )
    )
    c.add(
        Provision(
            statute="PPC",
            unit="section",
            number="302",
            heading="Punishment of qatl-i-amd",
            text="Whoever commits qatl-i-amd shall be punished with death as qisas, or with "
            "death or imprisonment for life as tazir.",
            in_force_from="1997-04-11",
        )
    )
    return c


def assistant() -> LawAssistant:
    return LawAssistant(corpus=corpus())


class TestCitationParsing:
    @pytest.mark.parametrize(
        "text",
        ["Section 302 PPC", "s. 302 of the Pakistan Penal Code", "sec 302, P.P.C.", "§302 PPC"],
    )
    def test_every_written_form_resolves_to_one_key(self, text):
        """Otherwise retrieval treats them as different provisions and an answer
        citing one will not match a query naming another."""
        assert parse(text)[0].key == "PPC:section:302"

    def test_constitutional_articles(self):
        citation = parse("Article 25 of the Constitution guarantees equality")[0]
        assert citation.key == "CONST:article:25"

    def test_order_and_rule_is_one_citation_not_two(self):
        """Civil procedure is cited by Order and Rule; splitting it would make the
        count of authorities in an answer wrong."""
        citations = parse("Order XXXIX Rule 1 CPC governs injunctions")
        assert len(citations) == 1
        assert citations[0].provision == "XXXIX/1"
        assert citations[0].pretty() == "Order XXXIX Rule 1 CPC"

    def test_subordinate_legislation(self):
        """The series marker is part of the number.

        This asserted `SRO:1125:2011` for `SRO 1125(I)/2011`: the pattern captured the
        roman numeral in a group named `series` and nothing read it, so
        `SRO 1125(I)/2011` and `SRO 1125(II)/2011` - different instruments, from
        different ministry sequences - resolved to one key. One instrument written
        several ways resolving to one key is the point of these keys; two instruments
        doing it is the same defect pointed the other way.
        """
        assert parse("Notified via SRO 1125(I)/2011")[0].key == "SRO:1125(I):2011"
        assert parse("SRO 1125(II)/2011")[0].key == "SRO:1125(II):2011"
        assert parse("SRO 1125/2011")[0].key == "SRO:1125:2011"
        # And the written forms of one SRO still meet.
        assert parse("S.R.O. 1125(I)/2011")[0].key == parse("SRO 1125(I)/2011")[0].key
        assert parse("SRO 1125(I)/2011")[0].pretty() == "SRO 1125(I)/2011"

    @pytest.mark.parametrize(
        ("text", "key"),
        [("PLD 2015 SC 401", "PLD:2015:SC:401"), ("2019 SCMR 1234", "SCMR:2019:-:1234")],
    )
    def test_law_reports_parse_in_both_orderings(self, text, key):
        """A statute's meaning frequently lives in the case law, and both orderings
        occur in practice."""
        assert parse(text)[0].key == key

    def test_several_citations_in_one_sentence(self):
        citations = parse("Under section 9 read with Article 199, per PLD 2020 LHC 55.")
        assert len(citations) == 3
        assert {c.kind for c in citations} == {"statutory", "reported"}

    def test_a_bare_section_has_no_statute(self):
        assert parse("section 9 applies")[0].statute == ""

    def test_context_can_resolve_a_bare_section(self):
        """Normal legal reading — but the default must be supplied, never guessed:
        attributing a provision to the wrong Act looks like a correct answer."""
        resolved = resolve_bare(parse("section 9 applies"), default_statute="PECA")
        assert resolved[0].key == "PECA:section:9"

    def test_subsections_are_preserved(self):
        assert parse("section 20(1)(a) PECA")[0].provision.startswith("20")

    def test_prose_without_citations_yields_nothing(self):
        assert parse("The parties reached an agreement.") == []

    def test_statute_table_is_lowercase_keyed(self):
        assert all(k == k.lower() for k in STATUTES)


class TestCorpus:
    def test_a_provision_knows_when_it_was_in_force(self):
        c = corpus()
        assert c.version_on("PECA:section:20", "2018-01-01").text == PECA_OLD
        assert c.version_on("PECA:section:20", "2026-01-01").text == PECA_NEW

    def test_a_date_before_commencement_returns_nothing(self):
        assert corpus().version_on("PECA:section:20", "2015-01-01") is None

    def test_the_boundary_date_belongs_to_the_new_version(self):
        """Half-open intervals: a provision substituted with effect from a date is the
        new text on that date, not the old one."""
        assert corpus().version_on("PECA:section:20", "2022-02-20").text == PECA_NEW

    def test_current_returns_the_live_version(self):
        assert corpus().current("PECA:section:20").text == PECA_NEW

    def test_history_is_the_amendment_trail(self):
        history = corpus().history("PECA:section:20")
        assert len(history) == 2
        assert history[0]["manner"] == "substituted"
        assert history[1]["to"] is None

    def test_superseded_text_is_retained_not_deleted(self):
        """Questions about past conduct are asked against the law as it then stood."""
        assert len(corpus().versions("PECA:section:20")) == 2

    def test_an_impossible_interval_is_refused(self):
        with pytest.raises(CorpusError):
            Provision(
                statute="X",
                unit="section",
                number="1",
                heading="",
                text="",
                in_force_from="2020-01-01",
                in_force_to="2019-01-01",
            )

    def test_a_clean_corpus_validates(self):
        assert corpus().validate() == []

    def test_overlapping_versions_are_caught(self):
        """If two versions are in force on one date, retrieval returns whichever it
        reaches first and the result is not reproducible."""
        c = corpus()
        c.add(
            Provision(
                statute="PECA",
                unit="section",
                number="20",
                heading="",
                text="third",
                in_force_from="2020-01-01",
            )
        )
        assert any("overlap" in p or "never ends" in p for p in c.validate())

    def test_more_than_one_live_version_is_caught(self):
        c = Corpus()
        for _ in range(2):
            c.add(
                Provision(
                    statute="X",
                    unit="section",
                    number="1",
                    heading="",
                    text="t",
                    in_force_from="2020-01-01",
                )
            )
        assert any("currently in force" in p for p in c.validate())

    def test_status_note_explains_a_repeal(self):
        old = corpus().versions("PECA:section:20")[0]
        note = old.status_note(dt.date(2026, 1, 1))
        assert "substituted" in note and "2022-02-20" in note


class TestRetrieval:
    def test_section_numbers_are_kept_as_tokens(self):
        """302 is among the most discriminating tokens in a penal code."""
        assert "302" in tokenise("section 302 of the code")

    def test_legal_stopwords_are_dropped(self):
        """Nearly every provision contains 'shall', 'section' and 'act'."""
        assert "shall" not in tokenise("the person shall be liable under this act")

    def test_urdu_script_survives_tokenisation(self):
        assert tokenise("قانون shahadat")[0].isalpha()

    def test_bm25_ranks_the_relevant_provision_first(self):
        search = LawSearch(corpus=corpus())
        hits = search.search("equality discrimination citizens", as_of="2026-01-01")
        assert hits[0].provision.number == "25"

    def test_a_section_number_query_finds_that_section(self):
        search = LawSearch(corpus=corpus())
        hits = search.search("302", as_of="2026-01-01")
        assert hits[0].provision.number == "302"

    def test_retrieval_never_returns_a_repealed_provision(self):
        """The failure the whole corpus design exists to prevent."""
        search = LawSearch(corpus=corpus())
        hits = search.search("reputation false information", as_of="2026-01-01")
        assert all(h.provision.currently_in_force for h in hits)

    def test_retrieval_on_a_past_date_returns_the_then_current_text(self):
        search = LawSearch(corpus=corpus())
        hits = search.search("reputation false information", as_of="2018-01-01")
        assert hits and hits[0].provision.text == PECA_OLD

    def test_results_are_explainable(self):
        """'These terms matched with these weights' is an answer a lawyer can check;
        'close in a 384-dimensional space' is not."""
        search = LawSearch(corpus=corpus())
        assert search.search("qatl-i-amd", as_of="2026-01-01")[0].why()

    def test_an_empty_query_returns_nothing(self):
        assert LawSearch(corpus=corpus()).search("the of and", as_of="2026-01-01") == []

    def test_idf_is_never_negative(self):
        """A term in most documents should contribute little, not subtract score from
        documents that contain it."""
        index = BM25Index().fit(list(corpus()))
        assert all(index._idf(t) >= 0 for t in ("person", "shall", "qatl"))


class TestAnswering:
    def test_the_same_question_gets_different_answers_on_different_dates(self):
        """The demonstration. A flat corpus answers both identically and one is
        wrong."""
        a = assistant()
        now = a.answer("punishment under section 20 PECA", as_of="2026-09-11")
        then = a.answer("punishment under section 20 PECA", as_of="2018-01-01")
        assert now.passages[0].text == PECA_NEW
        assert then.passages[0].text == PECA_OLD

    def test_a_cited_provision_is_looked_up_not_searched(self):
        """Returning the nearest match to a citation the user spelled out answers
        about the wrong law."""
        result = assistant().answer("What does Article 25 say?", as_of="2026-01-01")
        assert result.passages[0].citation.startswith("Article 25")

    def test_a_citation_not_in_force_is_refused_with_its_history(self):
        """A repealed section reads exactly like a live one. Only the corpus knows."""
        c = Corpus()
        c.add(
            Provision(
                statute="PECA",
                unit="section",
                number="66",
                heading="Repealed",
                text="old text",
                in_force_from="2016-01-01",
                in_force_to="2020-01-01",
                manner="repealed",
            )
        )
        result = LawAssistant(corpus=c).answer("section 66 PECA", as_of="2026-01-01")
        assert result.refused
        assert REFUSAL_NOT_IN_FORCE in result.refusal_reason
        assert result.superseded

    def test_a_date_before_enactment_names_the_first_commencement(self):
        """PECA s.20 was enacted in 2016 and substituted in 2022. Asked about 2015, the
        refusal once described the 2022 version and said it "commenced 2022-02-20"."""
        result = assistant().answer("section 20 PECA", as_of="2015-01-01")
        assert result.refused
        assert "commenced 2016-08-19" in result.refusal_reason
        assert "2022" not in result.superseded[0]["status"]

    def test_an_unknown_citation_is_refused_not_approximated(self):
        result = assistant().answer("What does section 999 PECA say?", as_of="2026-01-01")
        assert result.refused
        assert REFUSAL_UNKNOWN_CITATION in result.refusal_reason

    def test_a_weak_match_is_refused(self):
        """Returning the nearest provision as though it were relevant is how a
        confident wrong answer is produced."""
        a = LawAssistant(corpus=corpus(), min_score=1000.0)
        assert a.answer("unrelated question about shipping", as_of="2026-01-01").refused

    def test_nothing_found_is_refused(self):
        assert assistant().answer("zzzz qqqq", as_of="2026-01-01").refused

    def test_a_date_before_the_corpus_begins_finds_nothing(self):
        assert assistant().answer("qatl-i-amd", as_of="1900-01-01").refused

    def test_every_passage_carries_its_citation_and_dates(self):
        """The system never writes law, it quotes it and says where from."""
        passage = assistant().answer("qatl-i-amd punishment", as_of="2026-01-01").passages[0]
        assert passage.citation and passage.in_force_from

    def test_the_rendered_answer_is_citation_first(self):
        rendered = assistant().answer("qatl-i-amd punishment", as_of="2026-01-01").render()
        assert rendered.startswith("Section 302 PPC")

    def test_a_refusal_renders_as_a_refusal(self):
        rendered = assistant().answer("zzzz", as_of="2026-01-01").render()
        assert rendered.startswith("No answer:")

    def test_history_answers_when_did_this_change(self):
        """Asked constantly — about conduct before an amendment, about which version
        applies to a pending case — and unanswerable from a flat corpus."""
        history = assistant().history("section 20 PECA")
        assert history["versions"] == 2
        assert history["currently_in_force"]

    def test_history_of_an_unknown_provision_says_so(self):
        assert "error" in assistant().history("section 999 PECA")

    def test_history_requires_a_citation(self):
        assert "error" in assistant().history("what changed recently")


class TestCitationEdges:
    def test_a_statute_name_does_not_bind_to_the_front_of_a_longer_word(self):
        assert parse("section 5 PPCX")[0].statute == ""

    def test_extra_statutes(self):
        c = parse("section 5 of the PRPA", statutes={"prpa": "PRPA"})[0]
        assert c.key == "PRPA:section:5"

    def test_subsection_resolves_to_its_section(self):
        c = corpus()
        assert c.resolve("PECA:section:20(1)(a)") == "PECA:section:20"
        assert c.resolve("PECA:section:21(1)") == "PECA:section:21(1)"  # nothing to fall to
