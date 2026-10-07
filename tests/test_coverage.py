"""Where a corpus's record begins and ends, and the source history parser.

A corpus built from a consolidated text holds a statute only from the date its record
starts (`start_known: false`), and only up to the date it was last brought up to date
(`_corpus.as_at`). An answer outside that window must say so rather than report a
provision as "not yet in force" or silently current.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from paklaw import audit
from paklaw.corpus import CorpusError, Provision, build
from paklaw.ingest import main as corpus_main
from paklaw.ingest import read_corpus, read_rows, write_jsonl
from paklaw.mcp_server import LawServer, load_corpus
from paklaw.mcp_server import main as server_main
from paklaw.sources import pakistani_org_history, text_on

FIXTURES = Path(__file__).parent / "fixtures"


def row(**over) -> dict:
    base = {
        "statute": "PPC",
        "unit": "section",
        "number": "302",
        "heading": "Punishment of qatl-i-amd",
        "text": "Whoever commits qatl-i-amd shall be punished with death.",
        "in_force_from": "2016-01-01",
        "start_known": False,
    }
    return base | over


def test_start_known_must_be_a_bool():
    with pytest.raises(CorpusError, match="start_known"):
        Provision(**row(start_known="no"))


def test_before_the_record_is_unknown_not_not_yet_in_force():
    p = Provision(**row())
    note = p.status_note(dt.date(2010, 1, 1))
    assert "not recorded before 2016-01-01" in note
    assert "not yet in force" not in note


def test_coverage_warnings_before_record_and_after_as_at():
    corpus = build([row()], as_at="2025-06-30")
    assert corpus.coverage_warnings(dt.date(2020, 1, 1)) == []
    before = corpus.coverage_warnings(dt.date(2010, 1, 1))
    assert len(before) == 1 and "only from 2016-01-01" in before[0]
    after = corpus.coverage_warnings(dt.date(2026, 1, 1))
    assert len(after) == 1 and "up to 2025-06-30" in after[0]


def test_validate_rejects_two_record_starts_for_one_statute():
    corpus = build([row(), row(number="303", in_force_from="2017-01-01")])
    assert any("record begins on 2 different dates" in p for p in corpus.validate())


def test_validate_rejects_a_later_version_starting_at_the_record():
    corpus = build(
        [
            row(in_force_to="2020-01-01", manner="substituted"),
            row(in_force_from="2020-01-01"),
        ]
    )
    assert any("only the first version" in p for p in corpus.validate())


def test_corpus_header_round_trips_and_plain_readers_skip_it(tmp_path):
    path = tmp_path / "c.jsonl"
    write_jsonl([row()], path, meta={"as_at": "2025-06-30", "source": "pakistani.org"})
    rows, meta = read_corpus(path)
    assert meta == {"as_at": "2025-06-30", "source": "pakistani.org"}
    assert len(rows) == 1
    assert len(read_rows(path)) == 1


def test_corpus_header_only_first(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps(row()) + "\n" + json.dumps({"_corpus": {"as_at": "2025-01-01"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CorpusError, match="only as the first entry"):
        read_corpus(path)


def test_bad_as_at_is_named(tmp_path):
    path = tmp_path / "c.jsonl"
    write_jsonl([row()], path, meta={"as_at": "June 2025"})
    with pytest.raises(CorpusError, match="as_at"):
        load_corpus(str(path))


def test_amending_keeps_the_header(tmp_path):
    path = tmp_path / "c.jsonl"
    write_jsonl([row(start_known=True)], path, meta={"as_at": "2025-06-30"})
    new = tmp_path / "new.txt"
    new.write_text("Whoever commits qatl-i-amd shall be punished with death.", encoding="utf-8")
    assert (
        corpus_main(
            [
                "substitute",
                str(path),
                "section 302 PPC",
                "--on",
                "2024-01-01",
                "--by",
                "Act I",
                "--text-file",
                str(new),
            ]
        )
        == 0
    )
    _, meta = read_corpus(path)
    assert meta == {"as_at": "2025-06-30"}


@pytest.fixture
def server(tmp_path) -> LawServer:
    path = tmp_path / "c.jsonl"
    write_jsonl([row()], path, meta={"as_at": "2025-06-30"})
    return LawServer(*load_corpus(str(path)))


def test_server_keeps_as_at_and_reports_it(server):
    info = server.corpus_info({})
    assert info["as_at"] == "2025-06-30"
    assert info["recorded_from"] == {"PPC": "2016-01-01"}


def test_answer_after_as_at_warns(server):
    result = server.answer_question({"question": "section 302 PPC", "as_of": "2026-01-01"})
    assert not result["refused"]
    assert any("up to 2025-06-30" in w for w in result["warnings"])


def test_answer_inside_the_window_has_no_coverage_warning(server):
    result = server.answer_question({"question": "section 302 PPC", "as_of": "2020-01-01"})
    assert not any("recorded" in w or "up to" in w for w in result["warnings"])


def test_audit_before_record_is_unverified_not_a_problem(server):
    result = server.check_citations({"text": "s. 302 PPC", "as_of": "2010-01-01"})
    assert result["citations"][0]["status"] == audit.BEFORE_RECORD
    assert result["verdict"] == "unverified"


def test_audit_before_a_known_commencement_is_still_a_problem(tmp_path):
    path = tmp_path / "c.jsonl"
    write_jsonl([row(start_known=True)], path)
    srv = LawServer(*load_corpus(str(path)))
    result = srv.check_citations({"text": "s. 302 PPC", "as_of": "2010-01-01"})
    assert result["citations"][0]["status"] == audit.NOT_YET
    assert result["verdict"] == "problems"


def test_mcp_check_takes_the_corpus_as_a_positional_path(tmp_path, capsys):
    path = tmp_path / "c.jsonl"
    write_jsonl([row()], path)
    assert server_main(["--check", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["source"] == str(path)


def test_import_with_no_provisions_points_at_source(tmp_path, capsys):
    text = (FIXTURES / "ppc_pakistan_code_pp39-41.txt").read_text(encoding="utf-8")
    src = tmp_path / "ppc.txt"
    src.write_text(text, encoding="utf-8")
    code = corpus_main(
        [
            "import",
            str(src),
            "--statute",
            "PPC",
            "--in-force-from",
            "1860-10-06",
            "-o",
            str(tmp_path / "out.jsonl"),
        ]
    )
    if code != 0:  # the plain reader found nothing: the error must name the fix
        assert "--source pakistan-code" in capsys.readouterr().err
    assert (
        corpus_main(
            [
                "import",
                str(src),
                "--source",
                "pakistan-code",
                "--statute",
                "PPC",
                "--in-force-from",
                "1860-10-06",
                "-o",
                str(tmp_path / "out2.jsonl"),
            ]
        )
        == 0
    )


def test_pakistani_org_history_undoes_amendments_by_date():
    page = (FIXTURES / "ppc_pakistani_org_ss300-303.html").read_text(encoding="utf-8")
    tree, notes = pakistani_org_history(page)
    assert any(n.kind == "inserted" and n.year == 2016 for n in notes.values())
    now = text_on(tree, notes, lambda note: True)
    before_2016 = text_on(tree, notes, lambda note: (note.year or 0) < 2016)
    assert "fasad-fil-arz" in now
    assert "fasad-fil-arz" not in before_2016
    assert "\x01" not in now and "\x02" not in before_2016


def _sample_assistant():
    """A LawAssistant over the shipped three-provision sample."""
    import json as _json
    from importlib import resources

    from paklaw.answer import LawAssistant

    rows = _json.loads(resources.files("paklaw").joinpath("sample_corpus.json").read_text("utf-8"))
    return LawAssistant(build(rows if isinstance(rows, list) else rows["provisions"]))


# -- all nine refusal conditions, on the tool a caller actually uses ------


def test_the_readme_lists_the_refusal_statuses_the_code_can_set():
    """The README calls them "Nine refusal conditions" and tabulates the token for
    each. Nothing checked the two lists against each other, so a status could be
    added, renamed or dropped and the table would still read as an inventory.
    """
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src" / "paklaw" / "answer.py").read_text(
        encoding="utf-8"
    )
    in_code = set(re.findall(r'refusal_status = "([a-z_]+)"', source))

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("## Nine refusal conditions") :]
    section = section[: section.index("Every refusal carries")]
    # The header cell is literally `refusal_status`, so drop the column's own name.
    in_readme = set(re.findall(r"`([a-z_]+)`\s*\|", section)) - {"refusal_status"}

    assert in_code == in_readme, (
        f"only in the code: {sorted(in_code - in_readme)}; "
        f"only in the README: {sorted(in_readme - in_code)}"
    )
    assert len(in_code) == 9, f"the heading says nine; the code sets {len(in_code)}"


@pytest.mark.parametrize(
    ("question", "as_of", "statute", "status"),
    [
        # Nothing in the corpus is about this subject at all.
        (
            "what is the punishment for dacoity with murder?",
            "2026-01-01",
            None,
            "subject_not_in_corpus",
        ),
        # A citation with no Act. s.302 IPC is murder; s.302 PPC is qatl-i-amd.
        (
            "what is the punishment for murder under section 302?",
            "2026-01-01",
            None,
            "no_act_named",
        ),
        # An Act that was named and is not one this corpus holds.
        (
            "what is the punishment for murder under the Indian Penal Code?",
            "2026-01-01",
            None,
            "act_not_recognised",
        ),
        # A cited section this corpus does not have.
        ("what does section 500 PPC say?", "2026-01-01", None, "unknown_provision"),
        # The provision exists and was not in force on the date asked about.
        ("what is the penalty under section 20 PECA?", "1990-01-01", None, "not_in_force"),
        # A qualifier that selects a neighbouring offence the corpus does not hold.
        ("what is the punishment for attempt to murder?", "2026-01-01", None, "different_offence"),
    ],
)
def test_each_refusal_condition_fires_on_the_answer_path(question, as_of, statute, status):
    """Six of the nine were only asserted behaviourally, or on another tool.

    `no_act_named` was tested through `provision_history` and through ingest, and not
    through `answer_question` - which is the call a client makes and the path a bare
    "section 302" actually arrives on. A refusal condition tested somewhere else is
    not tested where it matters.
    """
    assistant = _sample_assistant()
    kwargs = {"statute": statute} if statute else {}
    answer = assistant.answer(question, as_of=as_of, **kwargs)

    assert answer.refused, f"expected a refusal, got {answer.passages[0].citation}"
    assert answer.refusal_status == status, answer.refusal_reason
    assert answer.refusal_reason, "a refusal with no reason is not a refusal a caller can act on"
    assert answer.passages == []


def test_a_refusal_never_carries_a_passage_and_an_answer_always_does():
    """The two halves of the contract, over every question in this file's cases."""
    assistant = _sample_assistant()
    for question in (
        "what is the punishment for murder?",
        "what is the punishment for dacoity with murder?",
        "what are the offences against the dignity of a natural person?",
        "what is the punishment for murder under section 302?",
    ):
        answer = assistant.answer(question, as_of="2026-01-01")
        if answer.refused:
            assert answer.passages == []
            assert answer.refusal_status and answer.refusal_reason
        else:
            assert answer.passages
            assert answer.refusal_status == ""
            assert answer.refusal_reason == ""


# -- the caveat belongs to the answer, not to one transport -------------------------


def _library(tmp_path):
    """A `LawAssistant` over a corpus whose record stops before the questions asked."""
    import sys

    from paklaw.answer import LawAssistant
    from paklaw.mcp_server import load_corpus

    del sys
    path = tmp_path / "c.jsonl"
    write_jsonl([row()], path, meta={"as_at": "2025-06-30"})
    corpus, _ = load_corpus(str(path))
    return LawAssistant(corpus=corpus), corpus


#: Question shapes that take different return paths out of `answer`. A citation
#: spelled out leaves four lines before the ordinary path does, which is how the first
#: fix for this missed it.
SHAPES = {
    "a citation spelled out": "section 302 PPC",
    "an ordinary question": "what is the punishment for qatl-i-amd?",
    "a question that refuses": "what is the punishment for dacoity?",
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_library_attaches_the_coverage_caveat_too(tmp_path, shape):
    """`Corpus.coverage_warnings` had one caller: the MCP server.

    So `demo.py` and every library consumer got the confident answer with no caveat,
    while the README's "record coverage - what the corpus does and does not claim to
    know" row reads as a property of the library. A reader asking about 2026 against a
    corpus recorded to 2025-06-30 was told the text in force without being told the
    record stops before the date.
    """
    assistant, _ = _library(tmp_path)
    answer = assistant.answer(SHAPES[shape], as_of="2026-01-01")
    assert any("up to 2025-06-30" in w for w in answer.warnings), (shape, answer.warnings)


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_library_and_the_server_say_the_same_thing(tmp_path, shape):
    """Two paths to one answer, and the caveat was on one of them. Compared rather
    than asserted twice, because "both say it" is the claim."""
    from paklaw.mcp_server import LawServer, load_corpus

    assistant, _ = _library(tmp_path)
    path = tmp_path / "c.jsonl"
    server = LawServer(*load_corpus(str(path)))

    from_library = [w for w in assistant.answer(SHAPES[shape], as_of="2026-01-01").warnings]
    over_protocol = server.answer_question({"question": SHAPES[shape], "as_of": "2026-01-01"})[
        "warnings"
    ]
    coverage = lambda notes: sorted(w for w in notes if "records amendments up to" in w)  # noqa: E731
    assert coverage(from_library) == coverage(over_protocol), (shape, from_library, over_protocol)


def test_a_date_inside_the_record_gets_no_caveat_from_either(tmp_path):
    """A caveat on every answer is a caveat nobody reads."""
    from paklaw.mcp_server import LawServer, load_corpus

    assistant, _ = _library(tmp_path)
    path = tmp_path / "c.jsonl"
    server = LawServer(*load_corpus(str(path)))

    inside = assistant.answer("section 302 PPC", as_of="2020-01-01")
    assert not any("up to" in w for w in inside.warnings), inside.warnings
    over = server.answer_question({"question": "section 302 PPC", "as_of": "2020-01-01"})
    assert not any("up to" in w for w in over["warnings"]), over["warnings"]


def test_the_caveat_is_attached_at_one_place_not_per_return():
    """`_answer` has eleven returns, and eleven is how many chances there are to miss
    one. `answer` wraps it, so a new return path cannot be the one that forgets - which
    is the mistake the first fix for this made, and the reason the call was in the
    server to begin with.
    """
    import ast
    import inspect

    from paklaw.answer import LawAssistant

    source = inspect.getsource(LawAssistant)
    tree = ast.parse("class X:\n" + "\n".join("    " + line for line in source.split("\n")[1:]))
    bodies = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in ("answer", "_answer")
    }
    assert set(bodies) == {"answer", "_answer"}, sorted(bodies)

    calls_in = {
        name: sum(
            1
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call) and getattr(sub.func, "attr", None) == "coverage_warnings"
        )
        for name, node in bodies.items()
    }
    assert calls_in == {"answer": 1, "_answer": 0}, calls_in

    returns = sum(1 for sub in ast.walk(bodies["_answer"]) if isinstance(sub, ast.Return))
    assert returns >= 8, f"_answer has {returns} returns; the wrapper covers all of them"
