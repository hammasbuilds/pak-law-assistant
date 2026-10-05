"""The MCP server, driven the way a client drives it: JSON-RPC lines in, lines out.

Most tests run the real read loop over byte streams; the subprocess tests start the
server exactly as an MCP client configuration would, and check that stdout carries
nothing but protocol.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from paklaw.corpus import CorpusError
from paklaw.mcp_server import (
    CITATION_LIMIT,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PROTOCOL_VERSIONS,
    SAMPLE_WARNING,
    TOOLS,
    LawServer,
    ToolError,
    load_corpus,
    serve,
)

SRC = Path(__file__).resolve().parents[1] / "src"
QUESTION = "What is the penalty for publicly transmitting false information about a person?"


def run(lines: list, *, corpus_path: str | None = None) -> list[dict]:
    """Feed messages through the real read loop; return every reply, in order."""
    corpus, source = load_corpus(corpus_path)
    stdin = io.BytesIO(
        b"".join(
            (line if isinstance(line, bytes) else json.dumps(line).encode()) + b"\n"
            for line in lines
        )
    )
    stdout = io.BytesIO()
    serve(LawServer(corpus, source), stdin, stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def init(version: str = "2025-06-18") -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {"protocolVersion": version, "capabilities": {}, "clientInfo": {"name": "t"}},
    }


def call(name: str, arguments: dict, request_id: int = 1) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def tool(name: str, arguments: dict, **kwargs) -> dict:
    """The tool result for one call, after a normal handshake."""
    reply = run([init(), call(name, arguments)], **kwargs)[-1]
    return reply["result"]


def body(result: dict) -> dict:
    return json.loads(result["content"][0]["text"])


def write_corpus(tmp_path: Path, rows: list[dict], *, jsonl: bool = False) -> str:
    path = tmp_path / ("corpus.jsonl" if jsonl else "corpus.json")
    text = "\n".join(json.dumps(r) for r in rows) if jsonl else json.dumps(rows)
    path.write_text(text, encoding="utf-8")
    return str(path)


ROW = {
    "statute": "PPC",
    "unit": "section",
    "number": "302",
    "heading": "Punishment of qatl-i-amd",
    "text": "Whoever commits qatl-i-amd shall be punished with death as qisas.",
    "in_force_from": "1997-04-11",
}


class TestHandshake:
    def test_a_supported_version_is_echoed(self):
        result = run([init("2025-03-26")])[0]["result"]
        assert result["protocolVersion"] == "2025-03-26"
        assert result["capabilities"] == {"tools": {"listChanged": False}}
        assert result["serverInfo"]["name"] == "pak-law-assistant"

    def test_an_unknown_version_gets_the_newest(self):
        assert run([init("1999-01-01")])[0]["result"]["protocolVersion"] == PROTOCOL_VERSIONS[0]

    def test_notifications_get_no_reply(self):
        replies = run(
            [
                init(),
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 7, "method": "ping"},
            ]
        )
        assert [r["id"] for r in replies] == [0, 7]
        assert replies[1]["result"] == {}

    def test_every_date_is_required_by_the_schema(self):
        """A model fills an optional date by omission, and today is the wrong default for
        a question about conduct before an amendment."""
        tools = run([init(), {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}])[1]
        by_name = {t["name"]: t for t in tools["result"]["tools"]}
        assert set(by_name) == {
            "answer_question",
            "check_citations",
            "provision_history",
            "compare_versions",
            "list_provisions",
            "changes_between",
            "parse_citations",
            "corpus_info",
        }
        for name, t in by_name.items():
            schema = t["inputSchema"]
            dates = [
                p
                for p, spec in schema["properties"].items()
                if "YYYY-MM-DD" in spec.get("description", "")
            ]
            assert set(dates) <= set(schema["required"]), name
            assert schema["additionalProperties"] is False
            assert t["annotations"]["readOnlyHint"] is True

    def test_every_tool_is_callable_through_the_protocol(self):
        """The dispatch table and the advertised list cannot drift apart."""
        tools = run([init(), {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}])[1]
        for t in tools["result"]["tools"]:
            reply = run([init(), call(t["name"], {})])[-1]
            assert "result" in reply, t["name"]  # a tool error at worst, never unknown

    def test_there_is_no_bare_search_tool(self):
        """Raw hits bypass the weak-match refusal; the only route to retrieval is one
        that can say no."""
        tools = run([init(), {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}])[1]
        assert not [t for t in tools["result"]["tools"] if "search" in t["name"]]


class TestAnswering:
    def test_the_date_decides_which_text_is_the_law(self):
        before = body(tool("answer_question", {"question": QUESTION, "as_of": "2020-06-01"}))
        after = body(tool("answer_question", {"question": QUESTION, "as_of": "2026-01-01"}))
        assert "three years" in before["passages"][0]["text"]
        assert "five years" in after["passages"][0]["text"]
        assert before["passages"][0]["in_force_to"] == "2022-02-20"
        assert after["passages"][0]["in_force_to"] is None

    def test_a_refusal_is_an_answer_not_an_error(self):
        result = tool(
            "answer_question",
            {"question": "registration of a private limited company", "as_of": "2026-01-01"},
        )
        assert result["isError"] is False
        assert body(result)["refused"] is True
        assert body(result)["passages"] == []

    def test_a_cited_provision_serialises_without_an_infinite_score(self):
        """The library scores a named provision as infinity; JSON has no such number."""
        result = tool("answer_question", {"question": "section 302 PPC", "as_of": "2026-01-01"})
        passage = body(result)["passages"][0]
        assert passage["citation"] == "Section 302 PPC"
        assert passage["score"] is None
        assert passage["matched"] == "cited (1.00)"

    def test_a_citation_before_enactment_is_refused_with_its_history(self):
        answer = body(
            tool("answer_question", {"question": "section 20 PECA", "as_of": "2015-01-01"})
        )
        assert answer["refused"]
        assert "commenced 2016-08-19" in answer["refusal_reason"]
        assert len(answer["superseded"][0]["history"]) == 2

    def test_non_ascii_survives_the_wire(self):
        answer = body(
            tool("answer_question", {"question": "section 20 PECA", "as_of": "2015-01-01"})
        )
        assert "\u2014" in answer["refusal_reason"]

    def test_statute_accepts_a_common_name(self):
        answer = body(
            tool(
                "answer_question",
                {
                    "question": "section 302",
                    "as_of": "2026-01-01",
                    "statute": "pakistan penal code",
                },
            )
        )
        assert answer["passages"][0]["citation"] == "Section 302 PPC"

    def test_structured_content_only_where_the_protocol_has_it(self):
        new = run([init("2025-06-18"), call("corpus_info", {})])[-1]["result"]
        old = run([init("2024-11-05"), call("corpus_info", {})])[-1]["result"]
        assert new["structuredContent"] == body(new)
        assert "structuredContent" not in old


class TestBadArguments:
    @pytest.mark.parametrize("as_of", ["yesterday", "20260101", "2026-02-30", ""])
    def test_a_bad_date_is_a_tool_error_the_model_can_fix(self, as_of):
        result = tool("answer_question", {"question": QUESTION, "as_of": as_of})
        assert result["isError"] is True
        assert "as_of" in result["content"][0]["text"]

    def test_an_unknown_statute_lists_the_known_ones(self):
        result = tool(
            "answer_question", {"question": QUESTION, "as_of": "2026-01-01", "statute": "XYZ"}
        )
        assert result["isError"] is True
        assert "PECA" in result["content"][0]["text"]

    def test_history_of_a_bare_section_is_refused_rather_than_guessed(self):
        result = tool("provision_history", {"citation": "section 302"})
        assert result["isError"] is True
        assert "names no Act" in result["content"][0]["text"]

    def test_history_with_the_statute_supplied(self):
        result = body(tool("provision_history", {"citation": "section 20", "statute": "PECA"}))
        assert result["versions"] == 2
        assert result["currently_in_force"] is True


class TestParsing:
    def test_every_family_and_a_bare_section(self):
        found = body(
            tool(
                "parse_citations",
                {
                    "text": "PLD 2015 SC 401, read with s. 302 of the Pakistan Penal Code, "
                    "Order XXXIX Rule 1 CPC and SRO 1125(I)/2011; see section 21."
                },
            )
        )["citations"]
        assert [c["kind"] for c in found] == [
            "reported",
            "statutory",
            "statutory",
            "subordinate",
            "statutory",
        ]
        assert found[1] | {"in_corpus": True} == found[1]
        assert found[2]["citation"] == "Order XXXIX Rule 1 CPC"
        assert found[4]["statute"] is None and "default_statute" in found[4]["note"]

    def test_default_statute_attaches_only_to_bare_sections(self):
        found = body(
            tool(
                "parse_citations",
                {"text": "section 20 and section 302 PPC", "default_statute": "PECA"},
            )
        )["citations"]
        assert [c["key"] for c in found] == ["PECA:section:20", "PPC:section:302"]


class TestProtocolErrors:
    def test_unknown_tool_and_method(self):
        replies = run(
            [init(), call("search", {}, 1), {"jsonrpc": "2.0", "id": 2, "method": "resources/list"}]
        )
        assert replies[1]["error"]["code"] == INVALID_PARAMS
        assert replies[2]["error"]["code"] == METHOD_NOT_FOUND

    def test_a_malformed_line_does_not_end_the_session(self):
        replies = run([init(), b"{not json", {"jsonrpc": "2.0", "id": 9, "method": "ping"}])
        assert replies[1]["error"]["code"] == PARSE_ERROR
        assert replies[2] == {"jsonrpc": "2.0", "id": 9, "result": {}}

    def test_a_batch_gets_a_batch(self):
        replies = run([[init(), {"jsonrpc": "2.0", "method": "notifications/initialized"}]])
        assert isinstance(replies[0], list) and len(replies[0]) == 1


class TestCorpus:
    def test_the_sample_says_so_in_every_result(self):
        assert body(tool("corpus_info", {}))["corpus_warning"] == SAMPLE_WARNING

    def test_a_real_corpus_carries_no_warning(self, tmp_path):
        info = body(tool("corpus_info", {}, corpus_path=write_corpus(tmp_path, [ROW])))
        assert "corpus_warning" not in info
        assert info["statutes"] == [
            {"statute": "PPC", "provisions": 1, "versions": 1, "currently_in_force": 1}
        ]

    def test_json_lines(self, tmp_path):
        corpus, _ = load_corpus(write_corpus(tmp_path, [ROW], jsonl=True))
        assert len(corpus) == 1

    def test_overlapping_versions_are_refused_at_start(self, tmp_path):
        """Two versions in force on one date make the answer depend on iteration order."""
        rows = [ROW | {"in_force_to": "2010-01-01"}, ROW | {"in_force_from": "2005-01-01"}]
        with pytest.raises(CorpusError, match="overlap"):
            load_corpus(write_corpus(tmp_path, rows))

    def test_an_unknown_field_names_the_row(self, tmp_path):
        with pytest.raises(CorpusError, match="provision 2 has unknown fields"):
            load_corpus(write_corpus(tmp_path, [ROW, ROW | {"sectoin": "x"}]))


def start(*args: str, env_corpus: str | None = None) -> subprocess.Popen:
    # Inherited, minus any corpus the developer has configured: the tests pin the sample.
    env = {k: v for k, v in os.environ.items() if k != "PAKLAW_CORPUS"}
    env["PYTHONPATH"] = str(SRC)
    if env_corpus:
        env["PAKLAW_CORPUS"] = env_corpus
    return subprocess.Popen(
        [sys.executable, "-m", "paklaw.mcp_server", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


class TestProcess:
    def test_stdout_carries_only_protocol(self):
        messages = [
            init(),
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            call("answer_question", {"question": QUESTION, "as_of": "2020-06-01"}),
        ]
        process = start()
        out, err = process.communicate(
            b"".join(json.dumps(m).encode() + b"\n" for m in messages), timeout=30
        )
        lines = out.splitlines()
        assert len(lines) == 2
        assert all(json.loads(line)["jsonrpc"] == "2.0" for line in lines)
        assert b"sample corpus" in err  # diagnostics go to stderr, never stdout

    def test_a_bad_corpus_exits_before_serving(self, tmp_path):
        process = start(env_corpus=str(tmp_path / "missing.json"))
        out, err = process.communicate(json.dumps(init()).encode() + b"\n", timeout=30)
        assert process.returncode == 2
        assert out == b""
        assert b"cannot load corpus" in err


class TestNewTools:
    def test_check_citations_flags_law_that_was_not_in_force(self):
        draft = (
            "The accused is liable under section 20 PECA and Article 25 of the Constitution; "
            "see also section 302 PPC, PLD 2015 SC 401 and section 9."
        )
        report = body(tool("check_citations", {"text": draft, "as_of": "2015-01-01"}))
        by = {c["citation"]: c["status"] for c in report["citations"]}
        assert by["Section 20 PECA"] == "not_yet_in_force"
        assert by["Article 25 CONST"] == "in_force"
        assert by["PLD 2015 SC 401"] == "not_checkable"
        assert by["Section 9"] == "no_act_named"
        assert report["verdict"] == "problems"
        assert report["problems"] == 2

    def test_check_citations_never_calls_an_unchecked_draft_clean(self):
        report = body(
            tool("check_citations", {"text": "see section 5 CrPC", "as_of": "2026-01-01"})
        )
        assert report["citations"][0]["status"] == "statute_not_loaded"
        assert report["verdict"] == "unverified"
        empty = body(tool("check_citations", {"text": "no citations", "as_of": "2026-01-01"}))
        assert empty["verdict"] == "unverified"

    def test_compare_versions_names_the_changed_words(self):
        result = body(
            tool(
                "compare_versions",
                {"citation": "section 20 PECA", "before": "2020-01-01", "after": "2026-01-01"},
            )
        )
        changes = {(c["before"], c["after"]) for c in result["changes"]}
        assert ("three", "five") in changes
        assert result["amended_by"] == ["Ordinance II of 2022"]

    def test_compare_versions_accepts_dates_in_either_order(self):
        forward = tool(
            "compare_versions",
            {"citation": "section 20 PECA", "before": "2020-01-01", "after": "2026-01-01"},
        )
        backward = tool(
            "compare_versions",
            {"citation": "section 20 PECA", "before": "2026-01-01", "after": "2020-01-01"},
        )
        assert body(forward)["changes"] == body(backward)["changes"]

    def test_list_provisions_for_a_statute_not_loaded_is_a_refusal_not_an_error(self):
        """A well-formed argument naming something absent is a refusal.

        It used to be a tool error, which tells a model it called the tool wrongly -
        and a model told that retries rather than reporting. `answer_question` calls
        the same condition `no_act_named` and `check_citations` calls it
        `statute_not_loaded`; this is the third name for one thing, so it uses the
        second. It is also the only path that returned no structuredContent, which the
        README's "every tool on every path" schema claim did not survive.
        """
        result = tool("list_provisions", {"statute": "CrPC", "as_of": "2026-01-01"})
        assert result.get("isError") is not True
        structured = body(result)
        assert structured["refused"] is True
        assert structured["refusal_status"] == "statute_not_loaded"
        assert "not in this corpus" in structured["refusal_reason"]
        assert structured["provisions"] == []

    def test_list_provisions(self):
        result = body(tool("list_provisions", {"statute": "peca", "as_of": "2026-01-01"}))
        assert result["provisions"][0]["citation"] == "Section 20 PECA"
        assert result["provisions"][0]["amended"] is True

    def test_changes_between_reports_a_substitution_once(self):
        result = body(tool("changes_between", {"start": "2022-01-01", "end": "2022-12-31"}))
        assert [(e["citation"], e["event"]) for e in result["events"]] == [
            ("Section 20 PECA", "substituted")
        ]

    def test_a_subsection_citation_finds_its_section(self):
        answer = body(
            tool("answer_question", {"question": "section 20(1) PECA", "as_of": "2026-01-01"})
        )
        assert not answer["refused"]
        assert answer["passages"][0]["citation"] == "Section 20 PECA"
        assert "subdivision" in answer["warnings"][0]

    def test_a_statute_only_the_corpus_knows_is_accepted(self, tmp_path):
        row = ROW | {"statute": "PRPA", "number": "5", "heading": "Tenancy"}
        result = tool(
            "list_provisions",
            {"statute": "prpa", "as_of": "2026-01-01"},
            corpus_path=write_corpus(tmp_path, [row]),
        )
        assert body(result)["provisions"][0]["citation"] == "Section 5 PRPA"

    def test_list_provisions_pages(self, tmp_path):
        rows = [ROW | {"number": str(n)} for n in range(1, 251)]
        path = write_corpus(tmp_path, rows)
        first = body(
            tool("list_provisions", {"statute": "PPC", "as_of": "2026-01-01"}, corpus_path=path)
        )
        assert len(first["provisions"]) == 200 and first["next_offset"] == 200
        rest = body(
            tool(
                "list_provisions",
                {"statute": "PPC", "as_of": "2026-01-01", "offset": 200},
                corpus_path=path,
            )
        )
        assert len(rest["provisions"]) == 50 and "next_offset" not in rest
        # statute order, not string order
        assert [p["citation"] for p in first["provisions"][:3]] == [
            "Section 1 PPC",
            "Section 2 PPC",
            "Section 3 PPC",
        ]


class TestActsTheParserDoesNotKnow:
    """A user's corpus can hold any Act. Before the parser learned the corpus's own
    statute names, "section 5 PRPA" parsed as "section 5" of no Act at all."""

    @pytest.fixture
    def path(self, tmp_path):
        row = ROW | {"statute": "PRPA", "number": "5", "heading": "Tenancy", "text": "rent"}
        return write_corpus(tmp_path, [row])

    def test_answer_by_citation(self, path):
        answer = body(
            tool(
                "answer_question",
                {"question": "section 5 PRPA", "as_of": "2026-01-01"},
                corpus_path=path,
            )
        )
        assert answer["passages"][0]["citation"] == "Section 5 PRPA"

    def test_check_citations(self, path):
        report = body(
            tool(
                "check_citations",
                {"text": "under s. 5 PRPA", "as_of": "2026-01-01"},
                corpus_path=path,
            )
        )
        assert report["citations"][0]["status"] == "in_force"
        assert report["verdict"] == "all_in_force"

    def test_history(self, path):
        result = body(tool("provision_history", {"citation": "section 5 prpa"}, corpus_path=path))
        assert result["versions"] == 1


def test_check_citations_reports_where_each_citation_is():
    text = "First section 302 PPC, then Article 25."
    report = body(tool("check_citations", {"text": text, "as_of": "2026-01-01"}))
    for c in report["citations"]:
        assert text[c["offset"] :].startswith(c["raw"])


def test_a_commencement_is_not_attributed_to_the_instrument_that_ended_it():
    """`amended_by` records what ENDED a version. Read as what began it, the 2016
    commencement of PECA s.20 was reported as made "by Ordinance II of 2022"."""
    events = body(tool("changes_between", {"start": "2016-01-01", "end": "2026-01-01"}))["events"]
    by = {(e["date"], e["event"]): e["by"] for e in events if e["citation"] == "Section 20 PECA"}
    assert by == {
        ("2016-08-19", "commenced"): None,
        ("2022-02-20", "substituted"): "Ordinance II of 2022",
    }


def _check_against_schema(value, schema, path="structuredContent"):
    """Enough of JSON Schema to prove the declared shape is the shape returned.

    Written out rather than imported: the package has no dependencies, and a schema
    nobody validates is a claim rather than a contract.
    """
    kinds = schema.get("type")
    kinds = [kinds] if isinstance(kinds, str) else list(kinds or [])
    python = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    if kinds:
        # bool is an int in Python, so an integer field must not accept True.
        if isinstance(value, bool) and "boolean" not in kinds:
            raise AssertionError(f"{path}: boolean where {kinds} was declared")
        allowed = tuple(python[k] for k in kinds if k in python)
        assert isinstance(value, allowed), f"{path}: {type(value).__name__} not in {kinds}"
    if "enum" in schema and value is not None:
        assert value in schema["enum"], f"{path}: {value!r} not in {schema['enum']}"
    if isinstance(value, dict):
        for key in schema.get("required", []):
            assert key in value, f"{path}: required key {key!r} missing"
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                _check_against_schema(item, properties[key], f"{path}.{key}")
            elif isinstance(schema.get("additionalProperties"), dict):
                _check_against_schema(item, schema["additionalProperties"], f"{path}.{key}")
            else:
                # The direction nobody checked. This walked returned -> declared only,
                # so a tool could return a key its schema never mentions and pass: three
                # tools returned `warnings` and none of them declared it. A client
                # generating types from these schemas drops the field.
                raise AssertionError(f"{path}.{key} is returned and not declared")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _check_against_schema(item, schema["items"], f"{path}[{index}]")


# Every path a tool can take, including the refusals: an outputSchema that only holds
# for the happy case is worse than none, because a client validating against it rejects
# a perfectly good refusal.
_OUTPUT_CASES = [
    ("answer_question", {"question": "punishment for murder", "as_of": "2026-01-01"}),
    ("answer_question", {"question": "cryptocurrency licensing", "as_of": "2026-01-01"}),
    ("answer_question", {"question": "section 20 PECA", "as_of": "2010-01-01"}),
    ("answer_question", {"question": "section 999 PPC", "as_of": "2026-01-01"}),
    ("check_citations", {"text": "section 20 PECA and section 302 PPC", "as_of": "2026-01-01"}),
    ("check_citations", {"text": "nothing cited here", "as_of": "2026-01-01"}),
    (
        "check_citations",
        {
            "text": "section 302 of the Indian Penal Code",
            "as_of": "2026-01-01",
            "default_statute": "PPC",
        },
    ),
    ("provision_history", {"citation": "section 20 PECA"}),
    ("provision_history", {"citation": "section 20 PECA", "with_text": True}),
    ("provision_history", {"citation": "section 999 PECA"}),
    (
        "compare_versions",
        {"citation": "section 20 PECA", "before": "2018-01-01", "after": "2026-01-01"},
    ),
    (
        "compare_versions",
        {"citation": "section 20 PECA", "before": "1990-01-01", "after": "1991-01-01"},
    ),
    (
        "compare_versions",
        {"citation": "section 999 PECA", "before": "2018-01-01", "after": "2026-01-01"},
    ),
    ("list_provisions", {"statute": "PECA", "as_of": "2026-01-01"}),
    ("list_provisions", {"statute": "PECA", "as_of": "1990-01-01"}),
    ("changes_between", {"start": "2015-01-01", "end": "2026-01-01"}),
    ("changes_between", {"start": "2024-01-01", "end": "2024-02-01"}),
    ("parse_citations", {"text": "ss. 302-304 PPC and PLD 2015 Lahore 401"}),
    ("parse_citations", {"text": "nothing"}),
    ("corpus_info", {}),
    # Paths that produce the keys the drift was hiding in. `warnings` was returned by
    # three tools and declared by none, and no case here could produce it: the list had
    # no reversed date range and no subdivision citation, so the check ran only over
    # payloads that happened to be complete.
    ("changes_between", {"start": "2026-01-01", "end": "1860-01-01"}),
    ("provision_history", {"citation": "section 20(1) PECA"}),
    (
        "compare_versions",
        {"citation": "section 20(1) PECA", "before": "2018-01-01", "after": "2026-01-01"},
    ),
    ("answer_question", {"question": "punishment for dacoity", "as_of": "2026-01-01"}),
    (
        "answer_question",
        {"question": "punishment under the Indian Penal Code", "as_of": "2026-01-01"},
    ),
    # check_citations grows three fields on branches no case reached: a subsection
    # citation (checked_as, subdivision_checked) and a citation not in force on the date
    # (history). All three were returned and undeclared, which is the same defect the
    # bidirectional check was added for - the check was right and the cases were thin.
    ("check_citations", {"text": "section 20(1) PECA", "as_of": "2026-01-01"}),
    ("check_citations", {"text": "section 20 PECA", "as_of": "2010-01-01"}),
    ("check_citations", {"text": "section 999 PECA and Part 3", "as_of": "2026-01-01"}),
]


@pytest.mark.parametrize("name,arguments", _OUTPUT_CASES)
def test_structured_output_matches_the_declared_output_schema(name, arguments):
    schemas = {t["name"]: t.get("outputSchema") for t in TOOLS}
    schema = schemas[name]
    assert schema is not None, f"{name} returns structuredContent but declares no outputSchema"
    server = LawServer(*load_corpus(None))
    _check_against_schema(getattr(server, name)(arguments), schema)


def test_every_tool_declares_an_output_schema():
    missing = [t["name"] for t in TOOLS if "outputSchema" not in t]
    assert missing == []


# --- numbers a client is given, and can act on ------------------------------------------


def test_a_client_can_reproduce_the_order_it_was_given():
    """`ranking` decided the order and was never sent.

    Its own comment said it was "kept as a field rather than recomputed, so the order a
    client sees can be checked against a number it was given" - and it was in no
    `Passage`, no output schema and no test. The only number a client did get, `score`,
    is not monotonic down the list, so sorting on it reorders the answer.
    """
    server = LawServer(*load_corpus(None))
    answer = server.answer_question({"question": "punishment for murder", "as_of": "2026-01-01"})
    passages = answer["passages"]
    for passage in passages:
        assert passage["coverage"] is not None
        assert passage["ranking"] is not None
    given = [p["citation"] for p in passages]
    rebuilt = [
        p["citation"] for p in sorted(passages, key=lambda p: (-p["coverage"], -p["ranking"]))
    ]
    assert given == rebuilt


def test_the_citation_count_is_the_page_and_the_total_is_the_whole():
    """`count` is declared "Citations on this page" and was set before the cut, so a
    text with 600 citations reported count 600 alongside 500 of them."""
    server = LawServer(*load_corpus(None))
    text = " and ".join(["section 302 PPC"] * 600)

    first = server.parse_citations({"text": text})
    assert first["count"] == len(first["citations"]) == CITATION_LIMIT
    assert first["total"] == 600
    assert first["next_offset"] == CITATION_LIMIT

    last = server.parse_citations({"text": text, "offset": CITATION_LIMIT})
    assert last["count"] == len(last["citations"]) == 100
    assert last["total"] == 600
    assert "next_offset" not in last


def test_a_refusal_carries_the_token_its_schema_tells_clients_to_branch_on():
    """`_REFUSAL` declares `refusal_status` and says "branch on this". `_refusal()`
    never set it, so every refusal from these two tools declared the key and omitted
    it."""
    server = LawServer(*load_corpus(None))
    for name, arguments in (
        ("provision_history", {"citation": "section 999 PECA"}),
        (
            "compare_versions",
            {"citation": "section 999 PECA", "before": "2018-01-01", "after": "2026-01-01"},
        ),
    ):
        result = getattr(server, name)(arguments)
        assert result["refused"] is True, name
        assert result["refusal_status"] == "unknown_provision", (name, result)

    # A citation that does not parse at all is rejected before the library is reached,
    # as a tool error rather than a refusal - which is why "no_citation_found" is not in
    # the enum. The library returns it; this tool cannot.
    with pytest.raises(ToolError):
        server.provision_history({"citation": "hello"})


def test_every_declared_refusal_status_is_one_the_code_can_produce():
    """An enum listing statuses nothing emits is as misleading as a missing one."""
    import paklaw.answer as answer_module

    declared = {
        value
        for tool in TOOLS
        for field, schema in (tool.get("outputSchema") or {}).get("properties", {}).items()
        if field == "refusal_status"
        for value in schema["enum"]
        if value is not None
    }
    source = Path(answer_module.__file__).read_text(encoding="utf-8")
    source += Path(Path(answer_module.__file__).parent / "audit.py").read_text(encoding="utf-8")
    source += Path(Path(answer_module.__file__).parent / "mcp_server.py").read_text(
        encoding="utf-8"
    )
    missing = sorted(status for status in declared if f'"{status}"' not in source)
    assert missing == [], missing


def test_compare_versions_does_not_say_both_when_it_means_either():
    """The condition is `or` and the summary said "both".

    So the common and interesting case - it did not exist on the earlier date and does
    now - was summarised as though the provision had never been in force at all.
    `summary` is the field a model paraphrases to a reader, and a wrong statement about
    whether a provision was in force on a date is the harm this repository is about.
    """
    server = LawServer(*load_corpus(None))

    one_side = server.compare_versions(
        {"citation": "section 20 PECA", "before": "2015-01-01", "after": "2026-01-01"}
    )
    assert one_side["before"]["in_force"] is False
    assert one_side["after"]["in_force"] is True
    assert "not in force on 2015-01-01, in force on 2026-01-01" in one_side["summary"]
    assert "both" not in one_side["summary"]

    neither = server.compare_versions(
        {"citation": "section 20 PECA", "before": "1990-01-01", "after": "1991-01-01"}
    )
    assert neither["summary"] == "in force on neither date; nothing to compare"

    # `changes` on every path, so a client reads one shape rather than branching.
    for result in (one_side, neither):
        assert result["changes"] == []


def test_an_old_protocol_is_not_offered_a_field_it_does_not_have():
    """`outputSchema` arrived in 2025-06-18, alongside `structuredContent`.

    The payload was correctly withheld from older clients and the schema was advertised
    to them anyway — a contract offered and then declined.
    """
    for version in PROTOCOL_VERSIONS:
        server = LawServer(*load_corpus(None))
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {"protocolVersion": version},
            }
        )
        listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        advertised = any("outputSchema" in t for t in listed["result"]["tools"])
        called = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "corpus_info", "arguments": {}},
            }
        )
        sent = "structuredContent" in called["result"]
        assert advertised == sent, (version, advertised, sent)
        assert sent is (version in {"2025-11-25", "2025-06-18"}), version


def test_an_id_that_can_be_answered_is_answered():
    """A wrong envelope is still attributable when it carried a usable id.

    `{"jsonrpc": "1.0", "id": 7}` was answered with id null, so a client could not match
    the error to the call. A null id is for a message too malformed to attribute.
    """
    server = LawServer(*load_corpus(None))
    answered = server.handle({"jsonrpc": "1.0", "id": 7, "method": "ping"})
    assert answered["id"] == 7
    assert answered["error"]["code"] == INVALID_REQUEST

    for unusable in ({"jsonrpc": "1.0", "method": "ping"}, {"jsonrpc": "1.0", "id": None}):
        assert server.handle(unusable)["id"] is None


def test_a_call_with_no_tool_named_says_so():
    """ "unknown tool: None" named the wrong problem: nothing was asked for, rather than
    something unrecognised being asked for. Both messages now list the tools."""
    server = LawServer(*load_corpus(None))
    missing = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call"})
    assert missing["error"]["code"] == INVALID_PARAMS
    assert "needs params.name" in missing["error"]["message"]
    assert "answer_question" in missing["error"]["message"]

    unknown = server.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "nope"}}
    )
    assert "unknown tool: 'nope'" in unknown["error"]["message"]
    assert "answer_question" in unknown["error"]["message"]
