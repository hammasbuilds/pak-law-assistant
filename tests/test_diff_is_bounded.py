"""One call cannot hold the server for a minute, return megabytes, or lie about a clause.

SECURITY.md said "Text arguments are capped at 200,000 characters and large results are
paged, so one call cannot make the server produce megabytes of output." `TEXT_LIMIT`
caps arguments; `compare_versions` compares two versions of a provision, and the CORPUS
sets that size.

What the precise comparison costs, measured on word lists with the shape named, because
the first version of this file quoted figures without one and a reviewer measuring a
different shape could not reproduce any of them:

    words   alternating same/unique   statute prose, 1 word in 50 changed
      500        1.25s                      0.03s
    1,000        8.67s                      0.24s
    2,000       84.59s                      2.17s
    6,000           -                      58.89s

Both columns are real. They are different inputs, and that is the point: the cost is in
the content and not in the length, so a figure without its shape says nothing and a cap
on the length is the wrong instrument. Growth is far worse than quadratic.

Three bounds, and a correctness case that the first two broke:

  * `MAX_DIFF_WORK` is a budget in pairwise comparisons for the WHOLE call. Per passage
    it was no bound at all - total work was the cap times the number of passages.
  * A passage past the budget is compared COARSELY, with a row saying so. It used to
    come back "not compared", which is a worse answer than a slow one and worse than
    the answer the unbounded code gave.
  * `MAX_DIFF_CHANGES`, `MAX_DIFF_CHARS` and `MAX_ROW_CHARS` bound the reply. The row
    count alone did not: "all of this was replaced by all of that" is ONE row holding
    both texts, and that row measured 4.38 MB.
  * And the sentence pass, which makes all of this affordable, anchored on the clause
    numbers and reported a sentence present unchanged in both texts as replaced.
"""

from __future__ import annotations

import json
import time

import pytest

from paklaw.audit import (
    ALL_CHANGE_KINDS,
    MAX_DIFF_CHANGES,
    MAX_DIFF_CHARS,
    MAX_DIFF_WORK,
    MAX_ROW_CHARS,
    word_diff,
)

#: Every wall-clock assertion here. Generous on purpose: the numbers above are seconds
#: to minutes and the bound is meant to put every input under a second, so a limit this
#: loose still fails on anything the bounds have stopped working for, and does not fail
#: on a loaded machine.
BUDGET_SECONDS = 10.0

#: The eight words a Pakistani penal provision is mostly made of.
_PROSE = ("whoever", "commits", "the", "offence", "shall", "be", "punished", "with")


def _prose(words: int, *, changed_every: int = 50, sentence_every: int = 0) -> tuple[str, str]:
    """Statute-like text, and the same text with one word in `changed_every` altered.

    `sentence_every` puts a full stop every so many words. It matters more than the
    length does: the sentence pass is what keeps the word comparison working on short
    inputs, so text with sentences and text without are the two different cases, and
    the first version of this file generated neither - it emitted `word123 word456`
    with no punctuation anywhere, so two of its three timing cases were timing the
    early return.
    """
    before = [_PROSE[i % len(_PROSE)] for i in range(words)]
    if sentence_every:
        for i in range(sentence_every - 1, words, sentence_every):
            before[i] += "."
    after = list(before)
    for i in range(0, words, changed_every):
        after[i] = "altered"
    return " ".join(before), " ".join(after)


def _alternating(words: int, *, sentences: bool) -> tuple[str, str]:
    """The pathological shape: every other word differs, every other word is shared.

    Which is what the cost blows up on, and roughly what two versions of an amended
    schedule look like.
    """
    join = ". " if sentences else " "
    return (
        join.join("same" if i % 2 else f"alpha{i}" for i in range(words)),
        join.join("same" if i % 2 else f"beta{i}" for i in range(words)),
    )


def _size(rows: list[dict]) -> int:
    return sum(len(r["before"] or "") + len(r["after"] or "") for r in rows)


def _timed(before: str, after: str) -> tuple[list[dict], float]:
    started = time.monotonic()
    rows = word_diff(before, after)
    return rows, time.monotonic() - started


# -- the answer, which no bound may change -------------------------------------------


def test_a_short_diff_is_the_diff():
    """The bound must not change the answer for an ordinary provision."""
    changes = word_diff(
        "shall be punished with imprisonment for three years or with fine",
        "shall be punished with imprisonment for five years or with fine",
    )
    assert changes == [{"change": "replaced", "before": "three", "after": "five"}], changes


def test_a_sentence_in_both_texts_is_not_reported_as_replaced():
    """The sentence pass anchored on the clause numbers.

    `C.` and `D.` are in both texts - a numbering survives a renumbering, which is what
    a numbering is for - so the matcher pinned them as equal and lined up the shifted
    clauses either side of them against each other:

        replaced  'penalty is three years.'        -> 'court may award costs.'
        replaced  'The court may award costs.'     -> 'Appeals lie to the High Court.'
        replaced  'Appeals lie to the High Court.' -> 'Interest accrues.'

    One clause was struck and one appended. "The court may award costs." is in both
    texts, unchanged, and the tool said an amendment replaced it with a sentence about
    appeals - a false statement about what an Act did, from the tool whose only job is
    to say what an Act did, and every row after the deletion named the wrong clause.
    """
    before = (
        "A. Definitions apply. B. The penalty is three years. "
        "C. The court may award costs. D. Appeals lie to the High Court."
    )
    after = (
        "A. Definitions apply. B. The court may award costs. "
        "C. Appeals lie to the High Court. D. Interest accrues."
    )
    changes = word_diff(before, after)

    removed = " ".join(r["before"] or "" for r in changes)
    added = " ".join(r["after"] or "" for r in changes)
    # The struck clause is reported struck and the appended one appended. Asserted on
    # the substance and not the exact span, because which shared word a run starts on
    # is cosmetic and the defect was not.
    assert "penalty is three years" in removed, changes
    assert "Interest accrues" in added, changes
    for row in changes:
        assert (row["before"], row["after"]) != (
            "The court may award costs.",
            "Appeals lie to the High Court.",
        ), changes


def test_an_amendment_inside_a_sentence_is_still_reported_word_by_word():
    """The other direction: the pairing that the test above constrains must survive.

    Sentences that each are the other amended stay paired, so the diff is "three" ->
    "five" and not two whole sentences.
    """
    changes = word_diff(
        "The penalty is three years. The court may award costs.",
        "The penalty is five years. The court may award double costs.",
    )
    assert {(r["before"], r["after"]) for r in changes} == {
        ("three", "five"),
        (None, "double"),
    }, changes


# -- the work budget -----------------------------------------------------------------


@pytest.mark.parametrize("words", [500, 2_000, 6_000, 20_000])
def test_statute_text_of_any_size_is_compared_precisely_and_quickly(words: int):
    """Real statute text has sentences, and every size of it stays precise.

    These are the sizes that used to be the slow case: 2,000 words of this shape took
    2.17s unbounded and 6,000 took 58.89s. The sentence pass cuts them into passages the
    precise comparison handles, so none of them is approximated.
    """
    changes, elapsed = _timed(*_prose(words, sentence_every=10))
    assert elapsed < BUDGET_SECONDS, (words, elapsed)
    assert changes, words
    assert not any(r["change"] == "approximate" for r in changes), (
        f"{words} words of ordinary statute text was approximated; the budget is too "
        "tight for the input this tool exists for"
    )


@pytest.mark.parametrize("words", [2_000, 20_000, 200_000])
def test_text_with_no_sentence_structure_is_approximated_not_refused(words: int):
    """One enormous sentence of the pathological shape: 200,000 words of it.

    Unbounded this is minutes. The old bound returned a row reading "not compared",
    which is a worse answer than a slow one - the same input, through the code before
    that bound existed, came back as a diff. A coarse comparison is a true statement
    about what differs; declining to look is not a statement about the law at all.
    """
    changes, elapsed = _timed(*_alternating(words, sentences=False))
    assert elapsed < BUDGET_SECONDS, (words, elapsed)
    assert any(r["change"] == "approximate" for r in changes), changes[:2]
    assert any(r["change"] in ("added", "removed", "replaced") for r in changes), (
        "approximated and then reported nothing, which is a refusal wearing another word"
    )


def test_no_row_ever_declines_to_compare():
    """There is no outcome in which the tool was handed two texts and compared neither."""
    assert "not compared" not in ALL_CHANGE_KINDS
    for before, after in (
        _alternating(50_000, sentences=False),
        _alternating(50_000, sentences=True),
        _prose(50_000, sentence_every=0),
        _prose(50_000, sentence_every=10),
    ):
        kinds = {r["change"] for r in word_diff(before, after)}
        assert kinds <= set(ALL_CHANGE_KINDS), kinds - set(ALL_CHANGE_KINDS)
        assert kinds & {"added", "removed", "replaced"}, kinds


def test_the_budget_is_one_budget_for_the_whole_call():
    """Per passage, the cap was no bound: total work was the cap times the passages.

    Many passages, each individually affordable and jointly not. The time is the
    assertion - with a per-passage cap this input is the cap paid over and over.
    """
    chunks = 400
    before = ". ".join(_alternating(60, sentences=False)[0] for _ in range(chunks))
    after = ". ".join(_alternating(60, sentences=False)[1] for _ in range(chunks))
    changes, elapsed = _timed(before, after)
    assert elapsed < BUDGET_SECONDS, elapsed
    assert changes


def test_the_budget_is_spent_on_the_early_passages_and_not_the_late_ones():
    """Which passages get precision is a decision, so it should be the stated one:
    first come, first served, until the budget is gone."""
    # One passage big enough to take the whole budget, then a small one after it.
    big_before, big_after = _alternating(2_000, sentences=False)
    before = f"{big_before}. three years"
    after = f"{big_after}. five years"
    changes = word_diff(before, after)
    assert changes[0]["change"] == "approximate", changes[0]
    assert MAX_DIFF_WORK < 2_000 * 2_000


# -- the reply ------------------------------------------------------------------------


def test_too_many_changes_are_cut_off_with_a_count():
    """Many SEPARATE changes, which is what fills a reply.

    Every other sentence differs, so every other sentence is its own passage.
    """
    before = ". ".join(
        f"clause {i} of this section applies" if i % 2 else f"clause {i} is omitted"
        for i in range(MAX_DIFF_CHANGES * 4)
    )
    after = before.replace("applies", "does not apply")
    changes = word_diff(before, after)
    assert len(changes) <= MAX_DIFF_CHANGES + 1, len(changes)
    assert changes[-1]["change"] == "truncated", changes[-1]
    assert f"{MAX_DIFF_CHANGES:,}" in changes[-1]["after"]
    assert "not listed" in changes[-1]["after"]


def test_one_row_cannot_be_megabytes():
    """A row count cannot bound a reply whose rows are unbounded.

    Two texts with nothing in common are a single opcode - "all of this replaced all of
    that" - carrying both texts entire. One row, 4.38 MB, from an input of 1.4 MB, and
    a character budget checked between PASSAGES never saw it.
    """
    before = " ".join(f"alpha{i}" for i in range(200_000))
    after = " ".join(f"beta{i}" for i in range(200_000))
    changes, elapsed = _timed(before, after)
    assert elapsed < BUDGET_SECONDS, elapsed
    assert len(before) > 1_400_000, len(before)
    for row in changes:
        for side in ("before", "after"):
            text = row[side] or ""
            assert len(text) <= MAX_ROW_CHARS + 200, (side, len(text))
    abridged = [r for r in changes if "abridged" in (r["before"] or "") + (r["after"] or "")]
    assert abridged, "a 1.4 MB side came back whole and did not say it was cut"
    assert f"{len(before):,}" in (abridged[0]["before"] or "") + (abridged[0]["after"] or "")


@pytest.mark.parametrize(
    ("label", "pair"),
    [
        ("nothing in common", (" ".join(f"a{i}" for i in range(200_000)), " ".join(f"b{i}" for i in range(200_000)))),
        ("alternating, one sentence", _alternating(200_000, sentences=False)),
        ("alternating, many sentences", _alternating(200_000, sentences=True)),
        ("repeated provision text", ("shall be punished with seven years. " * 20_000, "shall be punished with ten years. " * 20_000)),
    ],
)
def test_the_reply_cannot_be_megabytes(label: str, pair: tuple[str, str]):
    """The size of the serialised result, which is what a client receives.

    Four shapes, because the first version of this tested one - and passed it for a
    reason that had nothing to do with the size: the row cap cut it off long before any
    character count mattered, so the assertion could not have failed if the character
    bound had never been written.
    """
    before, after = pair
    assert len(before) > 600_000, (label, len(before))
    payload = json.dumps(word_diff(before, after))
    assert len(payload) < MAX_DIFF_CHARS + 50_000, (label, len(payload))


def test_the_character_bound_is_the_one_that_binds_on_many_long_rows():
    """Under the row cap and over the character cap, so only the character cap can
    stop it: a few hundred passages whose changes are each long."""
    sentence = " ".join(f"word{i}" for i in range(400))
    before = ". ".join(f"{sentence} alpha{i}" for i in range(300))
    after = ". ".join(f"{sentence} beta{i}" for i in range(300))
    changes = word_diff(before, after)
    assert _size(changes) <= MAX_DIFF_CHARS + MAX_ROW_CHARS * 2 + 500, _size(changes)
    if any(r["change"] == "truncated" for r in changes):
        assert len(changes) < MAX_DIFF_CHANGES, (
            "the row cap stopped it first, so this is not testing the character bound"
        )


# -- what the documents claim ---------------------------------------------------------


def test_security_md_describes_the_bounds_that_exist():
    """The claim this is here to make true was about arguments only, and the figures
    that replaced it named no shape, which is why they could not be reproduced."""
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "SECURITY.md").read_text(encoding="utf-8")
    assert f"{MAX_DIFF_WORK:,}" in text
    assert f"{MAX_DIFF_CHANGES}" in text
    assert f"{MAX_ROW_CHARS:,}" in text
    # The shape, without which the timings mean nothing.
    assert "alternating" in text
    # And no claim that anything is left uncompared.
    assert "not compared" not in text


# -- the other half of the reply, which the bound above does not touch ----------------
#
# A reviewer measured `compare_versions` on the unbounded code: 170,000 characters of
# provision text held the single-threaded server for 8,043 seconds (2h 14m) from about
# 90 bytes of arguments, and 1,460,000 characters produced a 5.8 MB reply. The budget
# above fixed the time - the same 170,000 characters is now 0.03s - and the `changes`
# array is ~30 KB whatever the input size.
#
# It did not fix the reply, because most of that reply was never the diff. The tool
# returns the provision as it stood on each date, so two texts; `provision_history
# --with_text` is not paged and returns one per VERSION. 2.9 MB of the 2.93 MB was
# those texts. Bounding one component of a reply and calling the reply bounded is the
# same error as bounding one passage and calling the work bounded.


def _two_version_corpus(characters: int):
    import datetime as dt

    from paklaw.corpus import Corpus, Provision

    sentence = (
        "Whoever commits an offence punishable under this section shall be punished "
        "with imprisonment of either description for a term which may extend to seven "
        "years or with fine or with both. "
    )
    before = (sentence * (characters // len(sentence) + 1))[:characters]
    after = before.replace("seven years", "ten years")

    def version(text: str, start, end=None):
        return Provision(
            statute="PPC",
            unit="section",
            number="302",
            heading="A long section",
            text=text,
            in_force_from=start,
            in_force_to=end,
        )

    return Corpus(
        provisions=[
            version(before, dt.date(2010, 1, 1), dt.date(2019, 12, 31)),
            version(after, dt.date(2020, 1, 1)),
        ],
        as_at=dt.date(2026, 1, 1),
    )


@pytest.mark.parametrize("characters", [170_000, 1_460_000])
def test_one_compare_call_is_neither_slow_nor_megabytes(characters: int):
    """The reviewer's two worst rows, through the tool rather than through `word_diff`.

    170,000 characters is still UNDER the server's own 200,000-character `TEXT_LIMIT`,
    which is the point of that finding: the limit applies to arguments and a provision's
    length is a property of the statute book.
    """
    from paklaw.audit import MAX_RESULT_TEXT, compare_versions

    corpus = _two_version_corpus(characters)
    started = time.monotonic()
    result = compare_versions(
        corpus, "PPC:section:302", before="2015-01-01", after="2021-01-01"
    )
    elapsed = time.monotonic() - started

    assert elapsed < BUDGET_SECONDS, (characters, elapsed)
    payload = len(json.dumps(result))
    assert payload < 4 * MAX_RESULT_TEXT, (characters, payload)
    assert result["changes"], "bounded into saying nothing"
    assert any(r["change"] in ("added", "removed", "replaced") for r in result["changes"])


def test_a_long_real_section_is_returned_whole():
    """The bound must not cut law. The longest sections in a tax, companies or
    procedure ordinance run past 50,000 characters, so those come back entire."""
    from paklaw.audit import MAX_RESULT_TEXT, compare_versions

    assert MAX_RESULT_TEXT > 50_000, MAX_RESULT_TEXT
    result = compare_versions(
        _two_version_corpus(50_000),
        "PPC:section:302",
        before="2015-01-01",
        after="2021-01-01",
    )
    for side in ("before", "after"):
        assert "abridged" not in result[side]["text"], side
        # Not exactly 50,000 on the `after` side: the amendment substitutes a shorter
        # phrase, which is what an amendment does.
        assert 48_000 < len(result[side]["text"]) <= 50_000, (
            side,
            len(result[side]["text"]),
        )


def test_an_abridged_text_says_its_real_length_and_where_to_read_it():
    from paklaw.audit import compare_versions

    result = compare_versions(
        _two_version_corpus(1_460_000),
        "PPC:section:302",
        before="2015-01-01",
        after="2021-01-01",
    )
    text = result["before"]["text"]
    assert "abridged" in text
    assert f"{1_460_000:,}" in text, text[-200:]
    assert "search_text" in text


def test_provision_history_with_text_is_bounded_per_version():
    """The site that was never measured and is the worst of the three: not paged, and
    one full provision text per version."""
    from paklaw.audit import MAX_RESULT_TEXT, abridge_text

    corpus = _two_version_corpus(1_460_000)
    for provision in corpus.provisions:
        assert len(abridge_text(provision.text)) < MAX_RESULT_TEXT + 500
