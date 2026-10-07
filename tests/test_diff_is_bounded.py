"""One call cannot hold the server for a minute or return megabytes.

SECURITY.md said "Text arguments are capped at 200,000 characters and large results are
paged, so one call cannot make the server produce megabytes of output." `TEXT_LIMIT` caps
arguments; `compare_versions` compares two versions of a provision, and the CORPUS sets
that size. A reviewer measured, with about 90 bytes of arguments:

      4,250 chars   0.12 s    12 KB reply
     17,000 chars   7.75 s    47 KB
     34,000 chars   56.8 s    94 KB
  1,460,000 chars      -      5.8 MB in one reply

`difflib.SequenceMatcher` with `autojunk=False` is roughly quadratic, and legal text is
the input that makes it so. 34,000 characters is ordinary for a tax or companies
ordinance.

`autojunk=False` is right and stays: the heuristic it disables drops any word appearing
in more than 1% of a long sequence, which in a statute is "shall", "the" and "section" -
the words a legal diff lines up on. The bound goes on the size instead.
"""

from __future__ import annotations

import time

import pytest

from paklaw.audit import MAX_DIFF_CHANGES, MAX_DIFF_WORDS, word_diff


def _text(words: int, seed: int = 0) -> str:
    return " ".join(f"word{(i * 7 + seed) % 997}" for i in range(words))


def test_a_short_diff_is_the_diff():
    """The bound must not change the answer for an ordinary provision."""
    before = "shall be punished with imprisonment for three years or with fine"
    after = "shall be punished with imprisonment for five years or with fine"
    changes = word_diff(before, after)
    assert len(changes) == 1, changes
    assert changes[0]["change"] == "replaced"
    assert changes[0]["before"] == "three"
    assert changes[0]["after"] == "five"


@pytest.mark.parametrize("words", [500, 2_000, 5_000])
def test_a_provision_sized_diff_is_fast(words: int):
    """Each of these is well inside the budget, and each used to be the slow case as
    the sizes doubled: 4,250 characters took 0.12s and 34,000 took 56.8s."""
    started = time.monotonic()
    word_diff(_text(words), _text(words, seed=3))
    assert time.monotonic() - started < 5.0, words


def test_one_sentence_over_the_word_budget_is_not_compared_and_says_so():
    """The budget applies to a PASSAGE, so reaching it takes a sentence this long.

    Not truncated silently: a diff that stopped early without saying so is worse than
    no diff, because a reader takes "no further changes" as a fact about the law.
    """
    # No sentence-ending punctuation anywhere, so there is nothing to cut it at.
    huge = _text(MAX_DIFF_WORDS + 50)
    started = time.monotonic()
    changes = word_diff(huge, _text(MAX_DIFF_WORDS + 50, seed=5))
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, elapsed
    assert len(changes) == 1, changes
    assert changes[0]["change"] == "not compared"
    assert f"{MAX_DIFF_WORDS:,}" in changes[0]["after"]
    assert "cubic" in changes[0]["after"]
    assert "no sentence break" in changes[0]["after"]


def test_the_same_text_with_sentence_breaks_is_compared():
    """The coarse pass is what makes a long provision affordable, so a text of the same
    length that HAS sentences must come back as a diff rather than as a refusal."""
    words = MAX_DIFF_WORDS * 3
    before = ". ".join(_text(12, seed=i) for i in range(words // 12))
    after = ". ".join(_text(12, seed=i + 1) for i in range(words // 12))
    started = time.monotonic()
    changes = word_diff(before, after)
    assert time.monotonic() - started < 5.0
    assert changes
    assert changes[0]["change"] != "not compared", changes[0]


def test_just_under_the_budget_is_still_compared():
    """A bound that refused everything would be a different way of answering nothing."""
    changes = word_diff(_text(MAX_DIFF_WORDS - 10), _text(MAX_DIFF_WORDS - 10, seed=9))
    assert changes
    assert changes[0]["change"] != "not compared"


def test_too_many_changes_are_cut_off_with_a_count():
    """Many SEPARATE changes, which is what fills a reply.

    Two texts with nothing at all in common are one opcode - "all of this replaced all
    of that" - and one row. What produces hundreds of rows is text that keeps agreeing
    and disagreeing, which is what two versions of a long provision look like: every
    other word differs, so every other word is its own passage.
    """
    before = ". ".join(("same" if i % 2 else f"alpha{i}") for i in range(MAX_DIFF_CHANGES * 4))
    after = ". ".join(("same" if i % 2 else f"beta{i}") for i in range(MAX_DIFF_CHANGES * 4))
    changes = word_diff(before, after)
    assert len(changes) <= MAX_DIFF_CHANGES + 1, len(changes)
    assert changes[-1]["change"] == "truncated"
    assert f"{MAX_DIFF_CHANGES:,}" in changes[-1]["after"]


def test_the_reply_cannot_be_megabytes():
    """The size of the serialised result, which is what a client receives.

    1,460,000 characters of corpus text produced 5.8 MB in one reply, from 90 bytes of
    arguments.
    """
    import json

    base = (
        "shall be punished with imprisonment of either description for a term which "
        "may extend to seven years or with fine. "
    )
    before = base * 6_000
    after = before.replace("seven years", "ten years")
    payload = json.dumps(word_diff(before, after))
    assert len(payload) < 500_000, len(payload)


def test_security_md_describes_both_bounds():
    """The claim this is here to make true was about arguments only."""
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "SECURITY.md").read_text(encoding="utf-8")
    assert f"{MAX_DIFF_WORDS:,} words" in text
    assert f"{MAX_DIFF_CHANGES} changed passages" in text
    assert "sentence" in text
