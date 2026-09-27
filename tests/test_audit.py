"""Audit functions on edge cases the sample corpus does not reach."""

from __future__ import annotations

from paklaw.audit import changes_between, check_citations, provision_order, word_diff
from paklaw.corpus import build

BASE = {"statute": "PPC", "unit": "section", "heading": "h", "text": "t"}


def test_repealed_then_reinstated():
    corpus = build(
        [
            BASE
            | {
                "number": "7",
                "in_force_from": "2000-01-01",
                "in_force_to": "2005-01-01",
                "manner": "omitted",
                "amended_by": "Act I of 2005",
            },
            BASE | {"number": "7", "in_force_from": "2010-01-01", "enacted_by": "Act II of 2010"},
        ]
    )
    assert corpus.validate() == []  # a re-enactment is not a data error
    events = changes_between(corpus, "1999-01-01", "2020-01-01")["events"]
    assert [e["event"] for e in events] == ["commenced", "omitted", "reinstated"]

    gap = check_citations(corpus, "section 7 PPC", as_of="2007-01-01")["citations"][0]
    assert gap["status"] == "not_in_force"
    assert "later version has been in force since 2010-01-01" in gap["note"]


def test_in_force_but_amended_later_is_flagged():
    corpus = build(
        [
            BASE | {"number": "7", "in_force_from": "2000-01-01", "in_force_to": "2020-01-01"},
            BASE | {"number": "7", "in_force_from": "2020-01-01", "text": "new"},
        ]
    )
    report = check_citations(corpus, "section 7 PPC", as_of="2010-01-01")
    c = report["citations"][0]
    # Good law on the date, but a draft may be quoting the later words: never "clean".
    assert c["status"] == "amended_since"
    assert report["verdict"] == "review"
    assert "amended with effect from 2020-01-01" in c["note"]


def test_word_diff_is_word_level():
    assert word_diff("up to three years", "up to five years") == [
        {"change": "replaced", "before": "three", "after": "five"}
    ]
    assert word_diff("a b", "a b c") == [{"change": "added", "before": None, "after": "c"}]


def test_statute_order():
    numbers = ["10", "2", "10A", "XL/1", "IX/2", "1"]
    assert sorted(numbers, key=provision_order) == ["1", "2", "10", "10A", "IX/2", "XL/1"]
