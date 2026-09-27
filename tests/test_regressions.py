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
from paklaw.citation import parse
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
