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
