"""Defects found by using the installed server the way an MCP user does (second review).

Each test reproduces one of them: a Constitution imported as sections, Urdu and Roman Urdu
citations that parsed to nothing, a whitespace question answered with a refusal about
nothing, reversed dates taken silently, and the packaging a registry client relies on.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from paklaw.citation import normalise_statute, parse
from paklaw.corpus import CorpusError
from paklaw.ingest import build_checked
from paklaw.ingest import main as corpus_main
from tests.test_mcp_server import body, call, init, run, tool

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10 has no tomllib; the packaging test is skipped there
    tomllib = None

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


# ---- the Constitution is numbered in articles ----------------------------------------


def test_importing_the_constitution_stores_articles(tmp_path, capsys):
    out = tmp_path / "statutes.jsonl"
    rc = corpus_main(
        [
            "import",
            str(FIXTURES / "constitution_pakistan_code_pp18-20.txt"),
            "--source",
            "pakistan-code",
            "--statute",
            "CONST",
            "--in-force-from",
            "1973-08-14",
            "-o",
            str(out),
        ]
    )
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows and {r["unit"] for r in rows} == {"article"}

    # and the README's own question now finds it
    result = body(
        tool(
            "answer_question",
            {"question": "Article 6", "as_of": "2024-01-01"},
            corpus_path=str(out),
        )
    )
    assert not result["refused"]
    assert result["passages"][0]["citation"] == "Article 6 CONST"


def test_explicit_unit_still_wins(tmp_path):
    out = tmp_path / "s.jsonl"
    rc = corpus_main(
        [
            "import",
            str(FIXTURES / "ppc_pakistan_code_pp39-41.txt"),
            "--source",
            "pakistan-code",
            "--statute",
            "PPC",
            "--in-force-from",
            "1860-10-06",
            "--unit",
            "section",
            "-o",
            str(out),
        ]
    )
    assert rc == 0
    assert '"unit": "section"' in out.read_text(encoding="utf-8")


def test_a_constitution_stored_as_sections_is_refused_at_load():
    row = {
        "statute": "CONST",
        "unit": "section",
        "number": "6",
        "heading": "High treason",
        "text": "...",
        "in_force_from": "1973-08-14",
    }
    with pytest.raises(CorpusError, match="numbered in articles"):
        build_checked([row])


# ---- Urdu and Roman Urdu -------------------------------------------------------------


@pytest.mark.parametrize(
    "text, key",
    [
        ("دفعہ 302 تعزیرات پاکستان", "PPC:section:302"),
        ("دفعہ ۳۰۲ تعزیرات پاکستان", "PPC:section:302"),
        ("dafa 302 PPC ki saza", "PPC:section:302"),
        ("dafa 489-F PPC", "PPC:section:489F"),
        ("آرٹیکل 25 آئین", "CONST:article:25"),
        ("section 497 Criminal Procedure Code", "CrPC:section:497"),
        ("section 302 of the Penal Code", "PPC:section:302"),
    ],
)
def test_urdu_and_roman_urdu_citations(text, key):
    assert [c.key for c in parse(text)] == [key]


def test_words_that_begin_like_dafa_are_not_citations():
    assert parse("daftar 5") == []


def test_urdu_question_is_a_lookup():
    result = body(
        tool(
            "answer_question",
            {"question": "دفعہ 302 تعزیرات پاکستان کی سزا", "as_of": "2024-01-01"},
        )
    )
    assert not result["refused"]
    assert result["passages"][0]["citation"] == "Section 302 PPC"


def test_common_names_as_statute_argument():
    assert normalise_statute("Penal Code") == "PPC"
    result = tool("list_provisions", {"statute": "penal code", "as_of": "2024-01-01"})
    assert not result["isError"]


# ---- argument handling ---------------------------------------------------------------


def test_whitespace_question_is_a_tool_error():
    result = tool("answer_question", {"question": "   \n", "as_of": "2024-01-01"})
    assert result["isError"]
    assert "required" in result["content"][0]["text"]


def test_reversed_dates_are_reported():
    result = body(
        tool(
            "compare_versions",
            {"citation": "section 20 PECA", "before": "2026-01-01", "after": "2020-01-01"},
        )
    )
    assert any("date order" in w for w in result["warnings"])
    assert result["before"]["date"] == "2020-01-01"

    events = body(tool("changes_between", {"start": "2026-01-01", "end": "2000-01-01"}))
    assert any("date order" in w for w in events["warnings"])
    assert events["total"] == 2


def test_dates_in_order_carry_no_swap_warning():
    events = body(tool("changes_between", {"start": "2000-01-01", "end": "2026-01-01"}))
    assert not any("date order" in w for w in events.get("warnings", []))


# ---- protocol ------------------------------------------------------------------------


def test_newest_protocol_gets_structured_content():
    replies = run([init("2025-11-25"), call("corpus_info", {})])
    assert replies[0]["result"]["protocolVersion"] == "2025-11-25"
    assert "structuredContent" in replies[1]["result"]


def test_old_protocol_gets_text_only():
    replies = run([init("2025-03-26"), call("corpus_info", {})])
    assert "structuredContent" not in replies[1]["result"]


# ---- packaging for the MCP registry --------------------------------------------------


@pytest.mark.skipif(tomllib is None, reason="tomllib needs Python 3.11")
def test_registry_manifest_matches_the_package():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    manifest = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    package = manifest["packages"][0]

    assert package["identifier"] == project["name"]
    assert package["version"] == manifest["version"] == project["version"]
    # a registry client runs `uvx <identifier>`, which needs a script of that name
    assert project["scripts"][project["name"]] == "paklaw.mcp_server:main"
    assert len(manifest["description"]) <= 100
    assert [v["name"] for v in package["environmentVariables"]] == ["PAKLAW_CORPUS"]

    # PyPI ownership check: the README (the PyPI description) names the server
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(r"mcp-name: " + re.escape(manifest["name"]) + r"(\s|-->)", readme)


def test_package_version_matches_the_library():
    from paklaw import __version__

    manifest = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    assert manifest["version"] == __version__
