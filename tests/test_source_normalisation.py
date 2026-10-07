"""Punctuation that renders as punctuation this package looks for, and is not it.

A review listed "non-ASCII fixture glyphs" as a minor point. Two of them are not minor:
the Pakistan Code writes a semicolon as `U+037E GREEK QUESTION MARK` throughout, and
carries `U+00AD SOFT HYPHEN` inside words. Both render as something a reader recognises -
one as a semicolon, one as nothing at all - which is why neither was noticed.

They change what the code does. `audit._SENTENCE` splits on `[.;:]`, so an enumerated
provision whose items end in `U+037E` is one unsplittable sentence as far as the version
diff is concerned, and that is the shape the diff compares coarsely instead of word by
word. A soft hyphen inside a word makes it a different word from the one anybody types.

Fixed at import and not in the fixture: the fixture is a verbatim copy of the source,
and the next corpus anyone imports carries the same artifacts, because this is the
Pakistan Code's own encoding rather than a mistake in the copy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paklaw.audit import _SENTENCE
from paklaw.ingest import build_checked
from paklaw.corpus import normalise_source
from paklaw.split import split_act
from tests.test_retrieval_quality import _corpus_rows

FIXTURES = Path(__file__).parent / "fixtures"

#: Every character the normaliser removes or replaces, and what it stands for.
LOOKALIKES = {
    ";": "GREEK QUESTION MARK, written for a semicolon",
    "­": "SOFT HYPHEN, invisible, inside words",
    "​": "ZERO WIDTH SPACE",
    "⁠": "WORD JOINER",
    "﻿": "ZERO WIDTH NO-BREAK SPACE, a BOM mid-file",
    " ": "NO-BREAK SPACE",
    "−": "MINUS SIGN, written for a hyphen",
}


def test_the_fixture_really_does_contain_the_character():
    """The premise. If the source is cleaned up, this file is testing a hypothetical.

    PPC s.53's list of punishments - "Firstly, Qisas ; Secondly, Diyat ;" - is where it
    was found, and it is every item of every enumerated list in the excerpt.
    """
    source = (FIXTURES / "ppc_pakistan_code_pp39-41.txt").read_text(encoding="utf-8")
    assert ";" in source
    assert source.count(";") >= 8, source.count(";")
    assert ";" not in source.split("Firstly")[1][:40], (
        "the source now uses a real semicolon here, so this file measures nothing"
    )


@pytest.mark.parametrize("character", sorted(LOOKALIKES))
def test_each_lookalike_is_normalised(character: str):
    cleaned = normalise_source(f"Qisas{character}Diyat")
    assert character not in cleaned, LOOKALIKES[character]


def test_the_greek_question_mark_becomes_a_semicolon_and_not_nothing():
    """It is punctuation and it means something: dropping it would join two clauses."""
    assert normalise_source("Qisas; Diyat") == "Qisas; Diyat"


def test_typography_the_statute_is_actually_written_with_is_left_alone():
    """Curly quotes and apostrophes are not lookalikes - they are what the text uses,
    and the text is served to a reader. A normaliser that rewrites them is editing the
    statute."""
    text = "Nothing is said to be done in “good faith” or in Ta’zir"
    assert normalise_source(text) == text


def test_no_provision_in_the_imported_corpus_carries_one():
    """End to end, through the importer the fixtures actually go through."""
    corpus = build_checked(_corpus_rows(), source="fixtures")
    offenders = [
        (p.citation().pretty(), hex(ord(character)))
        for p in corpus.provisions
        for character in LOOKALIKES
        if character in p.text or character in p.heading
    ]
    assert offenders == [], offenders


def test_the_semicolons_are_real_semicolons_in_the_imported_text():
    """Not merely absent: present as the character they stood for."""
    corpus = build_checked(_corpus_rows(), source="fixtures")
    punishments = [p for p in corpus.provisions if p.number == "53"]
    assert punishments, "PPC s.53 is not in the fixture corpus; this test moved"
    text = punishments[0].text
    assert "Qisas ;" in text or "Qisas;" in text, text[:120]


def test_a_list_written_with_the_lookalike_now_splits_into_sentences():
    """The consequence that made this worth fixing.

    The version diff matches sentences first and compares the words of each changed
    one, which is what keeps a long provision affordable. Split on `[.;:]`, a list
    punctuated with `U+037E` is a single sentence - so a provision like PPC s.53 was
    one blob, and changing one item of it was reported as the whole list replaced.
    """
    raw = "Firstly, Qisas ; Secondly, Diyat ; Thirdly, Arsh ; Fourthly, Daman"
    assert len([s for s in _SENTENCE.split(raw) if s.strip()]) == 1

    cleaned = normalise_source(raw)
    assert len([s for s in _SENTENCE.split(cleaned) if s.strip()]) == 4


def test_the_diff_of_one_item_of_such_a_list_names_only_that_item():
    """Which is the whole point: the sentence pass exists so a change is reported at
    the size it happened at."""
    from paklaw.audit import word_diff

    before = normalise_source(
        "Firstly, Qisas ; Secondly, Diyat ; Thirdly, Arsh ; Fourthly, Daman"
    )
    after = before.replace("Arsh", "Compensation")
    changes = word_diff(before, after)
    assert changes == [{"change": "replaced", "before": "Arsh", "after": "Compensation"}], changes


def test_split_act_normalises_what_it_is_handed():
    """The entry point for raw statute text, which is where a new corpus arrives."""
    rows, _report = split_act(
        "1. Short title. This Act may be called the Test Act.\n\n"
        "2. Punishments. The punishments are; Firstly, a fine; Secondly, costs.\n",
        statute="TEST",
        in_force_from="2020-01-01",
    )
    joined = " ".join(row["text"] for row in rows)
    assert ";" not in joined
    assert ";" in joined


# -- the entry point the server actually uses -----------------------------------------
#
# The first version of this fix called `normalise_source` from `split_act`, which is the
# entry point for raw statute TEXT. The one the server uses is `PAKLAW_CORPUS`: a corpus
# FILE, built by some other tool or by an older version of this importer, and that path
# never went near it. The fix covered the path nobody runs in production.


def test_a_corpus_file_is_normalised_on_load(tmp_path: Path):
    """Through `load_corpus`, which is what `PAKLAW_CORPUS` goes through."""
    import json

    from paklaw.mcp_server import load_corpus

    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps(
            [
                {
                    "statute": "PPC",
                    "unit": "section",
                    "number": "53",
                    "heading": "Punishments;",
                    "text": "The punishments are; Firstly, Qisas; Secondly, Diyat.",
                    "in_force_from": "1860-01-01",
                }
            ]
        ),
        encoding="utf-8",
    )
    corpus, _source = load_corpus(str(path))
    provision = corpus.provisions[0]
    assert ";" not in provision.text
    assert ";" not in provision.heading
    assert len([s for s in _SENTENCE.split(provision.text) if s.strip()]) == 3


def test_a_provision_built_by_hand_cannot_carry_one():
    """Not the importer, not `split_act`: the constructor.

    Normalising in one importer leaves every other way of building a corpus - another
    tool's exporter, a test, a script - free to produce provisions the diff cannot
    split. `Provision.__post_init__` is the one place they all pass through.
    """
    import datetime as dt

    from paklaw.corpus import Provision

    provision = Provision(
        statute="PPC",
        unit="section",
        number="53",
        heading="Punishments;",
        text="Firstly, Qisas; Secondly, Diyat­.",
        in_force_from=dt.date(1860, 1, 1),
    )
    assert ";" not in provision.text
    assert "­" not in provision.text
    assert ";" not in provision.heading
    assert provision.text.count(";") == 1, provision.text
