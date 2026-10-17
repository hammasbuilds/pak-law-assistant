"""The federal statute book that ships with this repository.

`data/pakistan-code.jsonl.gz` is built by `scripts/build_corpus.py` from the Pakistan
Code. These tests are about the corpus as a *file* - that it loads, that it validates,
and that the properties the build script works to guarantee actually hold in the thing
that shipped. A build script that silently stopped doing one of them would otherwise be
found by a reader asking a question.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pytest

from paklaw.act_index import MIN_KEY
from paklaw.citation import NOT_A_STATUTE_NAME
from paklaw.ingest import ARTICLE_STATUTES
from paklaw.mcp_server import load_corpus

CORPUS = Path(__file__).resolve().parents[1] / "data" / "pakistan-code.jsonl.gz"


@pytest.fixture(scope="module")
def corpus_path(shipped_path):
    return shipped_path


@pytest.fixture(scope="module")
def shipped_path():
    """Where the built corpus is, or a skip saying how to make one."""
    path = Path(os.environ.get("PAKLAW_CORPUS") or CORPUS)
    if not path.is_file():
        pytest.skip(
            f"no corpus at {path}: build one with `python scripts/build_corpus.py "
            "catalogue | fetch | build`, or set PAKLAW_CORPUS to one"
        )
    return path


@pytest.fixture(scope="module")
def shipped(shipped_path):
    """The built corpus, when it is on this machine.

    It is not in the repository: it is published as a dataset, and a 7.7 MB file is not
    what a zero-dependency package should carry. So these skip by name rather than fail,
    the way every other test here that needs data the repo does not hold behaves - and
    they skip with the command that produces it, so a reader who wants them run knows
    what to run. `PAKLAW_CORPUS` points at it anywhere else.
    """
    corpus, _ = load_corpus(str(shipped_path))
    return corpus


def test_it_loads_and_validates(shipped):
    """`load_corpus` already refuses a corpus with structural problems, so reaching
    here is most of the assertion; the count guards against shipping a stub."""
    assert len(shipped) > 20_000
    assert shipped.validate() == []


def test_it_holds_the_codes_a_question_is_most_likely_to_be_about(shipped):
    held = {provision.statute for provision in shipped.provisions}
    for statute in ("PPC", "CrPC", "CPC", "CONST", "QSO", "CONTRACT"):
        assert statute in held, statute


def test_the_instruments_numbered_in_articles_are_stored_that_way(shipped):
    """Stored as sections, "Article 25 of the Constitution" finds nothing."""
    for statute, unit in ARTICLE_STATUTES.items():
        held = [p for p in shipped.provisions if p.statute == statute]
        assert held, statute
        assert {p.unit for p in held} == {unit}, statute


def test_no_statute_key_is_short_enough_to_be_an_english_word(shipped):
    """A key of "A" matched the article "a" in any question and took it out of the
    question - see `test_regressions.test_a_one_letter_statute_key_does_not_eat_the_question`."""
    for statute in {p.statute for p in shipped.provisions}:
        assert len(statute) >= MIN_KEY, statute
        assert statute.lower() not in NOT_A_STATUTE_NAME, statute


def test_an_act_the_index_calls_repealed_is_not_served_as_current_law(shipped):
    """The failure this whole package exists to prevent, as a property of the file."""
    today = dt.date.today()
    for provision in shipped.provisions:
        if provision.manner == "repealed":
            assert not provision.in_force_on(today), provision.key
            assert provision.in_force_to is not None, provision.key


def test_a_repeal_whose_day_is_unknown_says_so(shipped):
    """Most of the Code's repeal notes name the instrument and not the day."""
    imprecise = [p for p in shipped.provisions if not p.end_known]
    assert imprecise, "no provision carries an imprecise end date; the flag is unused"
    for provision in imprecise[:50]:
        note = provision.status_note(dt.date.today())
        assert "unknown here" in note, provision.key


def test_every_provision_says_where_it_came_from(shipped):
    for provision in shipped.provisions:
        assert provision.source.startswith("Pakistan Code: "), provision.key


def test_the_file_describes_itself(shipped):
    """`_corpus.as_at` is what lets an answer about a later date say that a later
    amendment would not be in this corpus."""
    assert shipped.as_at is not None
    sources = shipped.meta.get("sources") or []
    assert sources and sources[0]["url"] == "https://pakistancode.gov.pk/"


def test_the_metadata_row_is_the_first_line_and_only_the_first(shipped, corpus_path):
    """`read_corpus` takes `_corpus` from row one; anywhere else it is a provision with
    unknown fields, and the corpus would refuse to load."""
    import gzip

    with gzip.open(corpus_path, "rt", encoding="utf-8") as handle:
        first = json.loads(handle.readline())
        assert set(first) == {"_corpus"}
        for line, _ in zip(handle, range(500), strict=False):
            if line.strip():
                assert "_corpus" not in json.loads(line)
