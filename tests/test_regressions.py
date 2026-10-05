"""Defects found by an independent review that used the server rather than reading it.

Each test reproduces what was observed, so none can come back quietly. Several were the
failure this repo exists to prevent — an answer about the wrong provision — reached by a
route the original tests did not walk.
"""

from __future__ import annotations

import io
import json
import time

import pytest

from paklaw.answer import REFUSAL_NO_ACT, REFUSAL_UNKNOWN_CITATION, LawAssistant
from paklaw.citation import is_foreign_report, parse
from paklaw.corpus import CorpusError, Provision, build
from paklaw.ingest import build_checked, insert, repeal, split_act
from paklaw.mcp_server import LawServer, load_corpus, serve

ROW = {"statute": "PPC", "unit": "section", "heading": "h", "in_force_from": "2000-01-01"}


def sample() -> LawServer:
    return LawServer(*load_corpus(None))


# ---- the parser read the wrong provision, or invented one --------------------------------


@pytest.mark.parametrize(
    ("text", "keys"),
    [
        ("section 302-B PPC", ["PPC:section:302B"]),  # not 302, which is murder
        ("section 489-F PPC", ["PPC:section:489F"]),
        ("sec. 26-A PECA", ["PECA:section:26A"]),
        ("u/s 302 PPC", ["PPC:section:302"]),
        ("sections 302/34 PPC", ["PPC:section:302", "PPC:section:34"]),
        ("sections 302, 34 and 109 PPC", ["PPC:section:302", "PPC:section:34", "PPC:section:109"]),
        ("302 PPC and 497 CrPC", ["PPC:section:302", "CrPC:section:497"]),
        ("Article 25 QSO", ["QSO:article:25"]),  # not the Constitution's equality clause
        ("Article 17 of the Qanun-e-Shahadat Order", ["QSO:article:17"]),
        ("Art. 199", ["CONST:article:199"]),
        ("fined Rs. 500 under section 20 PECA", ["PECA:section:20"]),  # not "s. 500"
        ("a fine of Rs. 10 million", []),
        ("Part 3 and chart 5", []),  # not Articles 3 and 5
        ("section 20, 2016", ["UNKNOWN:section:20"]),  # the year is not section 201
        ("1860 PPC", []),
        ("section 20(1)(a) PECA", ["PECA:section:20(1)(a)"]),  # clause stays lower-case
    ],
)
def test_citation_forms(text, keys):
    assert [c.key for c in parse(text)] == keys


# ---- answering ----------------------------------------------------------------------------


def test_a_hyphenated_section_is_not_answered_with_its_base_section():
    a = sample().answer_question(
        {"question": "punishment under section 302-B?", "as_of": "2026-01-01", "statute": "PPC"}
    )
    assert a["refused"] and "302B" in a["refusal_reason"]


@pytest.mark.parametrize(
    "question",
    [
        "Whoever commits theft",  # matched murder on "whoever" and "commits"
        "What is the fine for a traffic violation in rupees?",  # matched PECA on fine, rupees
        "Can my landlord evict me? person rupees",
    ],
)
def test_off_topic_questions_are_refused(question):
    assert sample().answer_question({"question": question, "as_of": "2026-01-01"})["refused"]


@pytest.mark.parametrize(
    ("question", "citation"),
    [
        ("What is the penalty for publicly transmitting false information?", "Section 20 PECA"),
        ("punishment of qatl-i-amd", "Section 302 PPC"),
        ("equality of citizens before law", "Article 25 CONST"),
    ],
)
def test_on_topic_questions_still_answer(question, citation):
    a = sample().answer_question({"question": question, "as_of": "2026-01-01"})
    assert [p["citation"] for p in a["passages"]] == [citation]


def test_every_unanswered_citation_is_named():
    a = sample().answer_question(
        {"question": "section 20 PECA, section 21 PECA and section 302", "as_of": "2026-01-01"}
    )
    assert [p["citation"] for p in a["passages"]] == ["Section 20 PECA"]
    assert any("Section 21 PECA" in w for w in a["warnings"])
    assert any(REFUSAL_NO_ACT in w and "Section 302" in w for w in a["warnings"])


def test_a_bare_section_is_refused_for_naming_no_act():
    a = sample().answer_question({"question": "What does section 302 say?", "as_of": "2026-01-01"})
    assert a["refused"] and a["refusal_reason"].startswith(REFUSAL_NO_ACT)
    assert REFUSAL_UNKNOWN_CITATION not in a["refusal_reason"]


def test_the_statute_filter_applies_before_the_cut():
    """Twenty PPC provisions about bail outscored the one CrPC provision, and filtering
    the top nine for CrPC found nothing."""
    rows = [
        ROW | {"number": str(n), "text": f"bail bail bail bail offence {n}"} for n in range(1, 21)
    ]
    # unrelated provisions, as in any real Act, so "bail" is a discriminating word
    rows += [ROW | {"number": str(n), "text": f"property {n}"} for n in range(100, 130)]
    rows.append(
        ROW
        | {
            "statute": "CrPC",
            "number": "497",
            "heading": "When bail may be taken",
            "text": "bail may be granted",
        }
    )
    answer = LawAssistant(corpus=build(rows)).answer("bail", as_of="2026-01-01", statute="CrPC")
    assert [p.citation for p in answer.passages] == ["Section 497 CrPC"]


def test_a_draft_quoting_later_words_is_not_called_clean():
    report = sample().check_citations(
        {"text": "punishable under section 20 PECA with up to five years", "as_of": "2019-03-01"}
    )
    assert report["citations"][0]["status"] == "amended_since"
    assert report["verdict"] == "review"


# ---- corpus rows are checked where they are loaded, not where they crash ------------------


def test_an_integer_number_is_accepted_as_text():
    assert Provision(**ROW, number=302, text="t").number == "302"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"text": None}, "text must be a string"),
        ({"unit": "clause"}, "unit must be one of"),
        ({"in_force_to": "2000-01-01"}, "never in force"),
        ({"in_force_from": "2000-13-01"}, "bad date"),
    ],
)
def test_bad_rows_are_refused_at_load_naming_the_row(change, message):
    rows = [ROW | {"number": "1", "text": "t"}, ROW | {"number": "2", "text": "t"} | change]
    with pytest.raises(CorpusError, match=message) as caught:
        build_checked(rows)
    assert "provision 2" in str(caught.value)


def test_unit_case_is_normalised():
    assert Provision(**ROW | {"unit": "Section"}, number="1", text="t").unit == "section"


def test_a_provision_can_be_re_enacted_after_a_gap():
    rows = [ROW | {"statute": "PECA", "number": "3", "text": "t"}]
    repeal(rows, "section 3 PECA", on="2024-01-01", by="Act X of 2024")
    insert(rows, "section 3 PECA", on="2025-06-01", by="Act II of 2025", text="t2", heading="h")
    assert build_checked(rows).validate() == []


def test_an_unexplained_gap_is_still_an_error():
    rows = [
        ROW | {"number": "3", "text": "t", "in_force_to": "2010-01-01"},
        ROW | {"number": "3", "text": "t", "in_force_from": "2012-01-01"},
    ]
    assert "gap in force" in build(rows).validate()[0]


# ---- the importer -------------------------------------------------------------------------


def test_hyphenated_headings_and_a_schedule():
    text = (
        "1. Short title.— This Act.\n"
        "489-F. Dishonestly issuing a cheque.— Whoever dishonestly issues a cheque.\n"
        "THE SCHEDULE\n1. Item one.— x\n2. Item two: y\n"
    )
    rows, report = split_act(text, statute="PPC", in_force_from="2000-01-01")
    assert [r["number"] for r in rows] == ["1", "489F"]
    assert "Item" not in rows[-1]["text"]
    assert report["not_imported"]


# ---- protocol -----------------------------------------------------------------------------


def run(*messages) -> list[dict]:
    stdout = io.BytesIO()
    lines = b"".join(json.dumps(m).encode() + b"\n" for m in messages)
    serve(sample(), io.BytesIO(lines), stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def call(name, arguments):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def test_a_null_id_is_an_invalid_request():
    reply = run({"jsonrpc": "2.0", "id": None, "method": "ping"})[0]
    assert reply["error"]["code"] == -32600


def test_a_misspelt_argument_is_a_tool_error_not_ignored():
    result = run(call("answer_question", {"question": "q", "asof": "2026-01-01"}))[0]["result"]
    assert result["isError"] and "asof" in result["content"][0]["text"]


def test_an_internal_failure_is_a_tool_error(monkeypatch):
    def broken(self, arguments):
        raise KeyError("boom")

    monkeypatch.setattr(LawServer, "corpus_info", broken)
    result = run(call("corpus_info", {}))[0]["result"]
    assert result["isError"] and "internal error in corpus_info" in result["content"][0]["text"]


def test_oversized_text_is_refused():
    result = run(call("parse_citations", {"text": "x" * 200_001}))[0]["result"]
    assert result["isError"] and "limit" in result["content"][0]["text"]


def test_not_found_has_the_refusal_shape_everywhere():
    s = sample()
    history = s.provision_history({"citation": "section 99 PECA"})
    compare = s.compare_versions(
        {"citation": "section 99 PECA", "before": "2020-01-01", "after": "2026-01-01"}
    )
    for result in (history, compare):
        assert result["refused"] is True and "not in this corpus" in result["refusal_reason"]


def test_history_of_a_subsection_says_so_and_can_carry_text():
    h = sample().provision_history({"citation": "section 20(1)(a) PECA", "with_text": True})
    assert "subdivision" in h["warnings"][0]
    assert "three years" in h["history"][0]["text"] and "five years" in h["history"][1]["text"]


# ---- scale --------------------------------------------------------------------------------


def big_corpus_server(versions_per_section=2, sections=5000) -> LawServer:
    rows = []
    for n in range(1, sections + 1):
        rows.append(
            ROW
            | {
                "number": str(n),
                "text": f"text {n}",
                "in_force_to": "2010-01-01",
                "manner": "substituted",
                "amended_by": "x",
            }
        )
        rows.append(ROW | {"number": str(n), "text": f"new {n}", "in_force_from": "2010-01-01"})
    corpus = build(rows)
    return LawServer(corpus, "test")


def test_whole_corpus_queries_are_fast_and_paged():
    """10,000 versions: changes_between took 25 s and returned 1.2 MB unpaged."""
    server = big_corpus_server()
    started = time.perf_counter()
    events = server.changes_between({"start": "1990-01-01", "end": "2026-01-01"})
    listing = server.list_provisions({"statute": "PPC", "as_of": "2026-01-01"})
    assert time.perf_counter() - started < 5
    assert events["total"] == 10_000 and len(events["events"]) == 500
    assert events["next_offset"] == 500 and len(listing["provisions"]) == 200


def test_many_citations_are_linear():
    """Overlap checking was quadratic: 20,000 citations took 25 s."""
    server = big_corpus_server()
    text = " ".join(f"section {n} PPC" for n in range(1, 5001) for _ in range(2))
    started = time.perf_counter()
    report = server.check_citations({"text": text, "as_of": "2026-01-01"})
    assert time.perf_counter() - started < 5
    assert report["total"] == 10_000 and len(report["citations"]) == 500
    assert report["counts"] == {"in_force": 10_000}


# --- second independent review -------------------------------------------------------
#
# Four of these were answers that looked right and were not: a question the corpus could
# answer being refused, and a foreign Act being certified as Pakistani law in force.


@pytest.mark.parametrize(
    "question,citation",
    [
        # The vocabulary gap: the question nominalises ("punishment") what the statute
        # conjugates ("punished"), and names the offence in English where the statute
        # names it in Urdu. Both provisions are present and both questions were refused.
        ("What is the punishment for murder?", "Section 302 PPC"),
        ("punishment for qatl-i-amd", "Section 302 PPC"),
        ("What is the punishment for homicide?", "Section 302 PPC"),
    ],
)
def test_statute_vocabulary_is_bridged(question, citation):
    server = LawServer(*load_corpus(None))
    answer = server.answer_question({"question": question, "as_of": "2026-01-01"})
    assert not answer["refused"], answer["refusal_reason"]
    assert answer["passages"][0]["citation"] == citation


@pytest.mark.parametrize(
    "question",
    [
        # ...and the bridge must not become a licence to answer anything. Each of these
        # shares exactly one content word with s.302 ("punishment"), and answering any
        # of them with the murder provision is the failure this repo exists to prevent.
        "What is the punishment for cryptocurrency?",
        "punishment for trespass",
        "punishment for blasphemy",
        "punishment for insider trading",
    ],
)
def test_one_shared_word_is_still_too_weak(question):
    server = LawServer(*load_corpus(None))
    answer = server.answer_question({"question": question, "as_of": "2026-01-01"})
    assert answer["refused"]
    assert answer["passages"] == []


def test_a_foreign_act_is_not_certified_as_pakistani_law():
    """A foreign code was reported as Section 302 PPC, in force.

    The parser saw no Act it recognised, which looked identical to no Act being named,
    so `default_statute` was applied and the audit returned `all_in_force` with zero
    problems. The `raw` field said "Section 302", so the trail did not show it either.
    """
    server = LawServer(*load_corpus(None))
    report = server.check_citations(
        {
            "text": "Section 302 of the Indian Penal Code governs.",
            "as_of": "2026-01-01",
            "default_statute": "PPC",
        }
    )
    assert report["verdict"] == "problems"
    (entry,) = report["citations"]
    assert entry["status"] == "act_not_recognised"
    assert "Indian Penal Code" in entry["note"]


def test_a_named_act_blocks_the_default_but_a_bare_section_does_not():
    """The default statute is a real feature; it must survive the fix above."""
    server = LawServer(*load_corpus(None))
    report = server.check_citations(
        {
            "text": "Under section 302, the accused is liable.",
            "as_of": "2026-01-01",
            "default_statute": "PPC",
        }
    )
    assert report["verdict"] == "all_in_force"
    assert report["citations"][0]["citation"] == "Section 302 PPC"


def test_an_unknown_act_is_parsed_but_left_unresolved():
    (citation,) = parse("Section 302 of the Indian Penal Code")
    assert citation.statute == ""
    assert citation.named_statute == "Indian Penal Code"


@pytest.mark.parametrize(
    "text,provisions",
    [
        # A range was one citation, and the Act after it was thrown away: a charge sheet
        # under ss.302-304 PPC was audited as a single section.
        ("ss. 302-304 PPC", ["302", "303", "304"]),
        ("sections 10 to 14 PPC", ["10", "11", "12", "13", "14"]),
        ("sections 302, 324-326 PPC", ["302", "324", "325", "326"]),
        # The letter forms are not ranges.
        ("section 489-F PPC", ["489F"]),
        ("section 354-A PPC", ["354A"]),
        # A span this wide is a mis-parse far more often than a citation, so the
        # endpoints are reported rather than five hundred sections.
        ("sections 1-500 PPC", ["1", "500"]),
    ],
)
def test_section_ranges_are_expanded(text, provisions):
    citations = parse(text)
    assert [c.provision for c in citations] == provisions
    assert {c.statute for c in citations} == {"PPC"}


@pytest.mark.parametrize(
    "text", ["Chapter 5 PPC", "Part 3 PPC", "Schedule 2 PECA", "In chapter 5 PPC the offence"]
)
def test_a_division_of_an_act_is_not_read_as_a_section(text):
    """A chapter is not a section: "Chapter 5 PPC" was reported as "Section 5 PPC"."""
    assert parse(text) == []


@pytest.mark.parametrize(
    "text,court",
    [
        # PLD reports the High Courts constantly, and a bench name in full made the
        # whole citation vanish — so a draft of High Court authority audited as
        # containing no citations at all.
        ("PLD 2015 Lahore 401", "LAHORE"),
        ("PLD 2015 Karachi 88", "KARACHI"),
        ("2021 YLR Peshawar 55", "PESHAWAR"),
        ("PLD 2018 Islamabad 7", "ISLAMABAD"),
        ("PLD 2019 SC 1", "SC"),
    ],
)
def test_reported_citations_outside_the_supreme_court_are_found(text, court):
    (citation,) = parse(text)
    assert citation.kind == "reported"
    assert citation.court == court


@pytest.mark.parametrize(
    "text,expected",
    [
        # No Act has a section 0, so this is not a citation at all. It parsed, and
        # an audit then reported it as an unknown provision OF the PPC - a fake
        # provision of a real Act, which reads like a finding about the draft.
        ("section 0 PPC", []),
        ("section 00 PPC", []),
        # Written without the dot constantly, and silently missed.
        ("s 302 ppc", ["302"]),
        ("ss 302/34 PPC", ["302", "34"]),
        # ...without turning the "s" of a word into a citation marker.
        ("Rs 500 was paid", []),
        ("its 302 sections", []),
        ("as 302 goes", []),
    ],
)
def test_the_bare_section_marker(text, expected):
    assert [c.provision for c in parse(text)] == expected


def test_one_provision_cited_five_times_is_one_problem():
    """`problems` counted occurrences, so a draft with one bad section cited
    repeatedly reported five problems and five things to fix."""
    server = LawServer(*load_corpus(None))
    report = server.check_citations(
        {
            "text": "Under section 999 PPC; see s.999 PPC again; and section 999 of the "
            "Pakistan Penal Code.",
            "as_of": "2026-01-01",
        }
    )
    # Every occurrence is still reported, with its offset, because an editor has
    # to find each one.
    assert len(report["citations"]) == 3
    assert report["problems"] == 3
    assert report["distinct_problems"] == 1
    assert report["distinct_citations"] == 1


@pytest.mark.parametrize(
    "text,pretty,key",
    [
        # Rules made under an Act are the commonest subordinate form in practice and
        # parsed to nothing at all, so a draft citing them audited as though it cited
        # nothing. A silent drop reads as a clean bill of health.
        (
            "Rule 5 of the Companies Rules",
            "Rule 5 of the Companies Rules",
            "RULES:Companies Rules:5",
        ),
        (
            "rule 12 of the Income Tax Rules 2002",
            "Rule 12 of the Income Tax Rules 2002",
            "RULES:Income Tax Rules:12",
        ),
    ],
)
def test_rules_made_under_an_act_are_found(text, pretty, key):
    (citation,) = parse(text)
    assert citation.kind == "subordinate"
    assert citation.pretty() == pretty
    assert citation.key == key


def test_a_rules_range_is_every_rule_in_it():
    assert [c.provision for c in parse("rules 3-5 of the Customs Rules")] == ["3", "4", "5"]


@pytest.mark.parametrize("text", ["the rules of the game", "under those rules 5 people left"])
def test_prose_about_rules_is_not_a_citation(text):
    assert [c for c in parse(text) if c.kind == "subordinate"] == []


def test_order_and_rule_of_the_cpc_still_parses_as_one_statutory_citation():
    """The new rules pattern must not claim the number first."""
    (citation,) = parse("Order XXXIX Rule 1 CPC")
    assert (citation.kind, citation.key) == ("statutory", "CPC:rule:XXXIX/1")


@pytest.mark.parametrize("text,report", [("AIR 1950 SC 27", "AIR"), ("AIR 1950 Lahore 12", "AIR")])
def test_a_foreign_report_is_reported_as_foreign_not_dropped(text, report):
    """It vanished entirely, so a draft of Indian authority audited as citation-free.

    Same argument as an unrecognised Act: a citation nobody parsed is a citation
    nobody checked, and silence told the reader the opposite of the truth.
    """
    (citation,) = parse(text)
    assert citation.report == report
    assert is_foreign_report(citation.report)

    server = LawServer(*load_corpus(None))
    report_out = server.check_citations({"text": text, "as_of": "2026-01-01"})
    (entry,) = report_out["citations"]
    assert entry["status"] == "not_checkable"
    assert "foreign law report" in entry["note"]


def test_a_pakistani_report_is_not_called_foreign():
    assert not is_foreign_report("PLD")
    assert not is_foreign_report("SCMR")
    # Written either way; both are the Pakistani series.
    assert not is_foreign_report("PCrLJ")


# --- third independent review ---------------------------------------------------------
#
# Every one of these was a confident wrong answer, which is the only kind of defect this
# repository treats as serious.


@pytest.mark.parametrize(
    "question",
    [
        # s.324 PPC. Answered with s.302 - "punished with death as qisas" - because
        # "attempt" counted as one interchangeable content word out of three.
        "what is the punishment for attempt to murder?",
        "what is the punishment for attempted murder?",
        # s.109. Same shape.
        "what is the punishment for abetment of murder?",
        # s.120B.
        "punishment for conspiracy to murder",
    ],
)
def test_a_qualifier_that_selects_another_offence_forces_a_refusal(question):
    server = LawServer(*load_corpus(None))
    answer = server.answer_question({"question": question, "as_of": "2026-01-01"})
    assert answer["refused"], answer["passages"][:1]
    assert answer["refusal_status"] == "different_offence"
    assert answer["passages"] == []


def test_the_same_question_without_the_qualifier_is_still_answered():
    """The qualifier gate must not swallow the question it was built beside."""
    server = LawServer(*load_corpus(None))
    answer = server.answer_question(
        {"question": "what is the punishment for murder?", "as_of": "2026-01-01"}
    )
    assert not answer["refused"]
    assert answer["passages"][0]["citation"] == "Section 302 PPC"


@pytest.mark.parametrize(
    "question,status",
    [
        ("what is section 302 of the Indian Penal Code?", "act_not_recognised"),
        ("what does section 302 say?", "no_act_named"),
        ("section 999 PPC", "unknown_provision"),
        ("what are the rules on cryptocurrency exchange licensing?", "nothing_matched"),
        ("what is the punishment for attempt to murder?", "different_offence"),
    ],
)
def test_every_refusal_carries_a_status_a_client_can_branch_on(question, status):
    """`refusal_reason` is prose and will be reworded; this is the contract.

    A client that wanted to treat "not in force" differently from "nothing matched" had
    to match on that prose, which is a contract nobody agreed to.
    """
    server = LawServer(*load_corpus(None))
    answer = server.answer_question({"question": question, "as_of": "2026-01-01"})
    assert answer["refused"]
    assert answer["refusal_status"] == status


def test_an_answered_question_has_no_refusal_status():
    server = LawServer(*load_corpus(None))
    answer = server.answer_question({"question": "punishment for murder", "as_of": "2026-01-01"})
    assert not answer["refused"]
    assert answer["refusal_status"] is None


def test_answer_question_no_longer_claims_a_named_act_was_unnamed():
    """It said "the cited provision names no Act" for text that plainly named one.

    check_citations got this right at the same time, so the two tools reached
    different conclusions about the same string - worse than either conclusion.
    """
    server = LawServer(*load_corpus(None))
    answer = server.answer_question(
        {"question": "what is section 302 of the Indian Penal Code?", "as_of": "2026-01-01"}
    )
    assert "names no Act" not in answer["refusal_reason"]
    assert "Indian Penal Code" in answer["refusal_reason"]

    audit = server.check_citations(
        {"text": "Section 302 of the Indian Penal Code.", "as_of": "2026-01-01"}
    )
    # The same conclusion, by the same name, from both tools.
    assert audit["citations"][0]["status"] == "act_not_recognised"
    assert answer["refusal_status"] == "act_not_recognised"


@pytest.mark.parametrize(
    "text,kind,pretty",
    [
        # "410 U.S. 113" came back as "Section 113" of no Act: the "S." of "U.S." was
        # read as the section marker, because the character before it is a full stop
        # rather than a letter.
        ("410 U.S. 113", "reported", "US 410 113"),
        ("see 410 U.S. 113 (1973)", "reported", "US 410 113"),
        # A parenthesised year is how English series are cited, and it parsed to nothing.
        ("(1932) AC 562", "reported", "AC 1932 562"),
    ],
)
def test_a_foreign_report_is_not_read_as_a_pakistani_section(text, kind, pretty):
    (citation,) = parse(text)
    assert citation.kind == kind
    assert citation.pretty() == pretty
    assert is_foreign_report(citation.report)


@pytest.mark.parametrize("text", ["U.S. policy says", "Rs 500 was paid", "its 302 sections"])
def test_the_abbreviation_fix_does_not_lose_real_citations(text):
    assert [c for c in parse(text) if c.kind == "statutory"] == []


def test_a_parenthesised_rules_title_parses():
    """The standard Pakistani naming style, and it produced nothing at all.

    `_ACT_TITLE` allowed only capitalised words, so the title pattern stopped at the
    bracket and the whole citation vanished - a draft citing it audited as citation-free.
    """
    (citation,) = parse("Rule 5 of the Companies (General Provisions) Rules 2018")
    assert citation.kind == "subordinate"
    assert citation.pretty() == "Rule 5 of the Companies (General Provisions) Rules 2018"
    # The unparenthesised and comma forms must still work.
    assert len(parse("Rule 5 of the Companies General Provisions Rules 2018")) == 1
    assert len(parse("rule 3 of the Income Tax Rules, 2002")) == 1
