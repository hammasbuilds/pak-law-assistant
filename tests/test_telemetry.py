"""The call log, and the console's arithmetic over it.

Three properties carry the weight here, and each one is a promise made elsewhere in the
repository that this could quietly break:

  * **Off unless asked.** The badge says `dependencies: zero` and the README says the
    server reads a corpus and answers; a build that starts writing a file the moment it
    is installed is a different thing than the one described. Asserted by running real
    calls with the variable unset and watching the directory stay empty.
  * **The question never reaches the log.** A question put to a legal corpus is a
    person's question and a draft pasted into `check_citations` is their draft. The log
    holds the shape of the traffic - tool, duration, verdict - and this asserts that a
    distinctive question and a distinctive draft do not appear in the file afterwards.
  * **Telemetry cannot break a call.** A full disk during a citation check must not turn
    into a failed citation check.

And one about the console: its figures are checked against the events on disk, not
against another call to the same function.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from paklaw import telemetry
from paklaw.mcp_server import LawServer, load_corpus

ROOT = Path(__file__).resolve().parents[1]


def _collect():
    """`observability/collect.py`, which is not part of the installed package.

    It lives beside the console rather than in `src/`, because it is the console's
    arithmetic and nothing the server imports.
    """
    spec = importlib.util.spec_from_file_location(
        "paklaw_observability_collect", ROOT / "observability" / "collect.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def server():
    return LawServer(*load_corpus(None))


#: A question and a draft with words that appear nowhere else, so finding either of them
#: in the log is unambiguous.
QUESTION = "zarcon pelagic underwriting syndicate rules"
DRAFT = "Relying on section 20 PECA and on section 302 of the Indian Penal Code, quixotic."


def _exercise(server) -> None:
    server._call_tool({"name": "corpus_info", "arguments": {}})
    server._call_tool(
        {"name": "answer_question", "arguments": {"question": QUESTION, "as_of": "2026-01-01"}}
    )
    server._call_tool(
        {"name": "check_citations", "arguments": {"text": DRAFT, "as_of": "2026-01-01"}}
    )
    server._call_tool({"name": "corpus_info", "arguments": {"not_a_field": 1}})


def test_nothing_is_written_when_the_variable_is_unset(server, tmp_path, monkeypatch):
    """The default. `dependencies: zero` and a file appearing uninvited do not sit
    together, so this is asserted over a directory rather than over the function."""
    monkeypatch.delenv(telemetry.ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert telemetry.destination() is None
    _exercise(server)
    assert list(tmp_path.rglob("*")) == []


def test_every_outcome_reaches_the_log(server, tmp_path, monkeypatch):
    """Including the ones that are not successes. A console that records what worked and
    drops what did not is the wrong way round: the refusals are the subject."""
    log = tmp_path / "nested" / "calls.jsonl"
    monkeypatch.setenv(telemetry.ENV, str(log))
    _exercise(server)

    written = log.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in written if line.strip()]
    assert len(rows) == 4, rows
    assert {r["outcome"] for r in rows} >= {"ok", "refused", "rejected"}
    assert all("at" in r and "ms" in r and "tool" in r for r in rows)
    # The parent directory did not exist and was created.
    assert log.parent.is_dir()


def test_the_question_and_the_draft_do_not_reach_the_log(server, tmp_path, monkeypatch):
    """The one that would be embarrassing to get wrong."""
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv(telemetry.ENV, str(log))
    _exercise(server)

    written = log.read_text(encoding="utf-8")
    assert "zarcon" not in written, "the question reached the log"
    assert "quixotic" not in written, "the draft reached the log"
    assert "Indian Penal Code" not in written, "the draft reached the log"
    # ...while the VERDICT on that draft did, which is the point of recording at all.
    assert "act_not_recognised" in written


def test_an_unwritable_destination_does_not_break_the_call(server, tmp_path, monkeypatch):
    """A directory where the file should be. Every failure path in `record` ends in a
    return, and this is the one a reader is most likely to produce by accident."""
    blocked = tmp_path / "a-directory"
    blocked.mkdir()
    monkeypatch.setenv(telemetry.ENV, str(blocked))
    answer = server._call_tool({"name": "corpus_info", "arguments": {}})
    assert answer["isError"] is False


def test_the_log_rotates_rather_than_growing_without_bound(tmp_path, monkeypatch):
    """One generation kept, so a long-running server cannot fill a disk."""
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv(telemetry.ENV, str(log))
    monkeypatch.setattr(telemetry, "MAX_BYTES", 400)
    for index in range(80):
        telemetry.record({"tool": "corpus_info", "outcome": "ok", "ms": 1.0, "n": index})
    assert log.exists()
    assert log.with_suffix(".jsonl.1").exists(), "nothing was rotated"
    assert log.stat().st_size < 4000


def test_a_half_written_line_does_not_empty_the_console(tmp_path):
    """A live process appends while this reads, so the last line can be truncated. One
    damaged line must cost one event, not the whole file."""
    collect = _collect()
    log = tmp_path / "calls.jsonl"
    good = json.dumps({"at": 1.0, "tool": "corpus_info", "outcome": "ok", "ms": 1.0})
    log.write_text(good + "\n" + good + "\n" + '{"at": 3.0, "tool": "corp', encoding="utf-8")
    assert len(collect.read(log)) == 2


def test_the_console_figures_match_the_events(server, tmp_path, monkeypatch):
    """The summary against the file, rather than against another call to `summarise`."""
    collect = _collect()
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv(telemetry.ENV, str(log))
    _exercise(server)
    _exercise(server)

    written = log.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in written if line.strip()]
    summary = collect.summarise(collect.read(log))

    assert summary["calls"] == len(rows)
    assert summary["refused"] == sum(1 for r in rows if r["outcome"] == "refused")
    assert sum(summary["refusals"].values()) == summary["refused"]
    assert sum(row["calls"] for row in summary["tools"]) == len(rows)
    assert summary["outcomes"]["rejected"] == sum(1 for r in rows if r["outcome"] == "rejected")
    assert summary["citation_statuses"]["act_not_recognised"] == 2  # one per `_exercise`


def test_a_status_the_console_has_never_seen_is_still_counted(tmp_path):
    """Grouped by what the log holds, never by a list typed into the console.

    The first version of `collect` carried eleven statuses read off a grep of the source.
    The first real call produced `subject_not_in_corpus`, which was not among them - so a
    fixed list would have dropped the most common refusal on the page built to show
    refusals.
    """
    collect = _collect()
    log = tmp_path / "calls.jsonl"
    log.write_text(
        json.dumps(
            {
                "at": 1.0,
                "tool": "answer_question",
                "outcome": "refused",
                "ms": 1.0,
                "refused": True,
                "refusal_status": "a_status_invented_for_this_test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = collect.summarise(collect.read(log))
    assert summary["refusals"] == {"a_status_invented_for_this_test": 1}


def test_a_thin_sample_is_declared_rather_than_dressed_up(tmp_path):
    """`p95` of three calls is the largest of three. The console says so."""
    collect = _collect()
    log = tmp_path / "calls.jsonl"
    log.write_text(
        "".join(
            json.dumps({"at": float(i), "tool": "corpus_info", "outcome": "ok", "ms": float(i)})
            + "\n"
            for i in range(3)
        ),
        encoding="utf-8",
    )
    summary = collect.summarise(collect.read(log))
    assert summary["thin"] is True
    assert summary["thin_at"] == collect.THIN
    assert summary["tools"][0]["thin"] is True


def test_no_calls_and_no_log_are_different_states(tmp_path):
    """An empty console should say which of the two it is looking at: a log that is not
    there wants a different fix than a log with nothing in it yet."""
    collect = _collect()
    missing = collect.summarise(collect.read(tmp_path / "absent.jsonl"))
    assert missing["calls"] == 0
    assert missing["first"] is None
    assert missing["sample_corpus"] is None
