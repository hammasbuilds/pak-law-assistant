"""The importer on real statute text, in the layouts the two public sources really use.

The fixtures are excerpts of the real sources, unedited except for being cut short:

  ppc_pakistani_org_ss300-303.html      pakistani.org's Penal Code page: header, sections
                                         300-303, its source list and two of its notes
  ppc_pakistan_code_pp39-41.txt         `pdftotext -layout` of pages 39-41 of the Pakistan
                                         Code PPC PDF, after the contents lines for those
                                         sections
  constitution_pakistan_code_pp18-20.txt the same for the Constitution, pages 18-20, where
                                         a footnote number is glued to Article 1 ("11.")

The importer this replaced read the whole pakistani.org Penal Code as one section 237
(84 KB) and reported no problem: its heading pattern needed the number and heading on
one line, and its greedy "must follow the last number" rule let every later heading be
rejected after one bad one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paklaw.ingest import main
from paklaw.sources import pakistan_code, pakistani_org, strip_stars
from paklaw.split import provision_spans, split_act

FIXTURES = Path(__file__).parent / "fixtures"


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def by_number(rows: list[dict]) -> dict[str, dict]:
    return {r["number"]: r for r in rows}


class TestPakistaniOrg:
    @pytest.fixture
    def parsed(self):
        text, notes = pakistani_org(read("ppc_pakistani_org_ss300-303.html"))
        rows, report = split_act(text, statute="PPC", in_force_from="2016-01-01")
        return text, notes, rows, report

    def test_the_number_alone_on_its_line_is_a_heading(self, parsed):
        """'302.' on one line and 'Punishment of qatl-i-amd:' on the next."""
        _, _, rows, _ = parsed
        assert [r["number"] for r in rows] == ["300", "301", "302", "303"]
        assert by_number(rows)["302"]["heading"] == "Punishment of qatl-i-amd"
        assert by_number(rows)["302"]["text"].startswith("Whoever commits qatl-e-amd")

    def test_amendment_markers_leave_the_text_and_the_notes_are_kept(self, parsed):
        text, notes, rows, _ = parsed
        s302 = by_number(rows)["302"]["text"]
        assert "133" not in s302 and "134" not in s302
        assert "not applicable: Provided that nothing in clause (c)" in s302
        assert "XLIII of 2016" in notes[134]
        assert 'for Full-stop: "."' in notes[133]

    def test_every_span_not_imported_is_reported_with_its_size(self, parsed):
        text, _, rows, report = parsed
        spans = {s["what"]: s for s in report["not_imported"]}
        assert spans["text before the first provision"]["characters"] > 1000
        trailer = spans["text after the last provision (schedules, forms or notes)"]
        assert trailer["starts"].startswith("Source::")
        # the site's source list is not the law: it must not end up in section 303
        assert "Nakhoda" not in by_number(rows)["303"]["text"]
        accounted = report["imported_characters"] + sum(
            s["characters"] for s in report["not_imported"]
        )
        # headings' own numbers and whitespace aside, every character is accounted for
        assert accounted / len(text.replace("\n", "").replace(" ", "")) > 0.9


class TestPakistanCode:
    @pytest.fixture
    def source(self):
        return pakistan_code(read("ppc_pakistan_code_pp39-41.txt"))

    @pytest.fixture
    def parsed(self, source):
        return split_act(strip_stars(source.text), statute="PPC", in_force_from="2016-01-01")

    def test_page_footers_and_footnotes_do_not_reach_the_law(self, parsed):
        rows, _ = parsed
        for row in rows:
            assert "Page 40 of 179" not in row["text"]
            assert "Indian Penal Code Amdt. Act" not in row["text"]
            assert "Subs. by" not in row["text"]

    def test_footnotes_are_read_per_page_and_markers_point_at_them(self, source):
        notes = {(f.page, f.number): f.text for f in source.footnotes}
        assert notes[(3, 5)] == "Subs. by Act XLIV of 2016, s. 2."
        marked = [a for a in source.amended if (a.page, a.number) == (3, 5)]
        assert len(marked) == 2  # a colon, and the proviso it introduces
        second_proviso = max(marked, key=lambda a: a.end - a.start)
        words = source.text[second_proviso.start : second_proviso.end]
        assert words.startswith("Provided further that in a case in which the sentence")
        assert "2[" not in source.text and "5[" not in source.text

    def test_a_wrapped_footnote_line_is_not_a_new_footnote(self, source):
        """'23rd March, 1956).' continues note 8; read as note 23 it broke the page."""
        note = next(f for f in source.footnotes if f.page == 3 and f.number == 8)
        assert note.text.endswith("23rd March, 1956).")

    def test_the_contents_decide_numbers_and_headings(self, parsed):
        """And where the contents stop, an ascending run of headings continues them.

        This listed thirteen provisions, which is what the CONTENTS of a three-page
        excerpt lists. The text runs on to s.66, and those nine were folded into s.57 -
        reported as `swallowed_headings` and left there. ss.58, 59, 61 and 62 are
        omitted or repealed in the source, so they are still reported rather than
        loaded; 60, 63, 64, 65 and 66 are provisions and are now parsed as provisions.
        """
        rows, report = parsed
        numbers = [r["number"] for r in rows]
        assert numbers == [
            "46",
            "47",
            "48",
            "49",
            "50",
            "51",
            "52",
            "52A",
            "53",
            "54",
            "55",
            "55A",
            "57",
            "60",
            "63",
            "64",
            "65",
            "66",
        ]
        # s.57 is its own section again, not ten of them. 141 characters against the
        # 1,800 it held, which is the figure the retrieval failure came from: BM25
        # penalises length, so the provision that answers "how many years is
        # imprisonment for life" lost to s.302 for being nine sections long.
        assert len(by_number(rows)["57"]["text"]) < 200, len(by_number(rows)["57"]["text"])
        assert by_number(rows)["57"]["text"].startswith("In calculating fractions")
        assert by_number(rows)["66"]["heading"].startswith("Description of imprisonment")
        assert by_number(rows)["47"]["heading"] == '"Animal"'
        assert by_number(rows)["47"]["text"].startswith("The word “animal” denotes")
        assert report["missing_from_contents"] == []

    def test_a_repealed_section_is_reported_not_loaded(self, parsed):
        rows, report = parsed
        assert "56" not in by_number(rows)
        assert any(line.startswith("section 56:") for line in report["omitted_or_repealed"])

    def test_the_fonts_semicolons_are_semicolons(self, parsed):
        rows, _ = parsed
        assert "Qisas ;" in by_number(rows)["53"]["text"]
        assert ";" not in json.dumps(rows, ensure_ascii=False)

    def test_a_chapter_heading_labels_what_follows_and_leaves_the_body(self, parsed):
        rows, _ = parsed
        assert by_number(rows)["53"]["chapter"] == "Chapter III OF PUNISHMENTS"
        assert "CHAPTER" not in by_number(rows)["52A"]["text"]


class TestConstitution:
    @pytest.fixture
    def parsed(self):
        text = strip_stars(pakistan_code(read("constitution_pakistan_code_pp18-20.txt")).text)
        return split_act(text, statute="CONST", in_force_from="2026-01-01", unit="article")

    def test_a_footnote_number_glued_to_article_1_is_recognised(self, parsed):
        """The PDF prints '11. The Republic and its territories' for Article 1, note 1."""
        rows, report = parsed
        assert rows[0]["number"] == "1"
        assert rows[0]["heading"] == "The Republic and its territories"
        assert rows[0]["text"].startswith("(1) Pakistan shall be a Federal Republic")
        assert "11" not in [r["number"] for r in rows]
        assert report["missing_from_contents"] == []

    def test_parts_and_chapters_label_articles(self, parsed):
        rows, _ = parsed
        assert by_number(rows)["7"]["chapter"].startswith("Part II FUNDAMENTAL RIGHTS")
        assert by_number(rows)["8"]["chapter"] == "Chapter 1 FUNDAMENTAL RIGHTS"


class TestSplitting:
    def test_one_bad_number_does_not_reject_every_heading_after_it(self):
        """The greedy rule accepted '900.' and then refused 2 and 3 as out of order."""
        text = (
            "1. First.— one\n900. A numbered line inside section 1.— stray\n"
            "2. Second.— two\n3. Third.— three\n"
        )
        rows, report = split_act(text, statute="X", in_force_from="2020-01-01")
        assert [r["number"] for r in rows] == ["1", "2", "3"]
        assert "900. A numbered line" in rows[0]["text"]
        assert len(report["rejected_headings"]) == 1

    def test_text_after_a_chapter_heading_stays_in_the_body(self):
        """The old importer kept only what came before a CHAPTER line."""
        text = (
            "1. First.— one\nCHAPTER II\nOFFENCES\nA closing paragraph that is still law.\n"
            "2. Second.— two\n"
        )
        rows, report = split_act(text, statute="X", in_force_from="2020-01-01")
        assert rows[0]["text"] == "one A closing paragraph that is still law."
        assert rows[1]["chapter"] == "Chapter II OFFENCES"
        labels = [s for s in report["not_imported"] if "chapter" in s["what"]]
        assert labels[0]["characters"] == len("CHAPTER II\nOFFENCES")

    def test_text_before_the_first_provision_is_reported(self):
        rows, report = split_act(
            "An Act to make provision.\n1. First.— one\n", statute="X", in_force_from="2020-01-01"
        )
        assert report["not_imported"][0] == {
            "what": "text before the first provision",
            "characters": len("An Act to make provision."),
            "starts": "An Act to make provision.",
        }

    def test_a_suspiciously_long_provision_is_flagged(self):
        text = "1. First.— " + "word " * 50 + "\n2. Second.— two\n"
        _, report = split_act(text, statute="X", in_force_from="2020-01-01", too_long=100)
        assert report["too_long"] == [{"provision": "section 1", "characters": 249}]


def test_the_command_line_reads_a_pakistan_code_pdf_text(tmp_path, capsys):
    corpus = tmp_path / "c.jsonl"
    source = FIXTURES / "ppc_pakistan_code_pp39-41.txt"
    args = ["import", str(source), "--source", "pakistan-code", "--statute", "PPC"]
    assert main([*args, "--in-force-from", "2016-01-01", "-o", str(corpus)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["provisions"] == 18
    assert "imported 18 provision(s)" in captured.err
    # Five: s.56, which the contents list as repealed, and ss.58, 59, 61 and 62, which
    # the text brackets as omitted and which used to be invisible inside s.57's body.
    assert "5 omitted or repealed" in captured.err


# --- an exported helper with no caller had already drifted ----------------------------


@pytest.mark.parametrize(
    "fixture,statute,unit",
    [
        ("ppc_pakistan_code_pp39-41.txt", "PPC", "section"),
        ("constitution_pakistan_code_pp18-20.txt", "CONST", "article"),
    ],
)
def test_provision_spans_agrees_with_the_importer(fixture, statute, unit):
    """`provision_spans` is exported, had no caller and no test, and had drifted.

    On the real Penal Code pages it returned 14 spans where `split_act` imported 13 -
    the extra one being s.56, which reads "[Sentence of Europeans and Americans to
    penal servitude.] Rep. by the Criminal Law ... Act, 1949". A build script
    attributing amendment markers by offset would have attributed some to a provision
    the importer never created.

    This is the second time an unused export turned out to disagree with the code it
    has to agree with; `resolve_bare` was the first. The test is the point: a helper
    nothing calls is a helper nothing checks.
    """
    text = strip_stars(pakistan_code(read(fixture)).text)
    rows, _ = split_act(text, statute=statute, in_force_from="1900-01-01", unit=unit)
    spans = provision_spans(text)
    assert [number for number, _, _ in spans] == [r["number"] for r in rows]


def test_include_repealed_asks_for_the_layout_instead():
    """ "What does this offset sit under" and "which provision is this" are different
    questions, and the caller has to say which it means."""
    text = strip_stars(pakistan_code(read("ppc_pakistan_code_pp39-41.txt")).text)
    default = [n for n, _, _ in provision_spans(text)]
    every = [n for n, _, _ in provision_spans(text, include_repealed=True)]
    assert "56" not in default
    assert "56" in every
    # Five, not one. ss.58, 59, 61 and 62 are "[...] Omitted by" or "Rep. by" in the
    # source and were invisible to this comparison while all nine sat inside s.57's
    # body; now that the run after the contents is parsed, they are repealed headings
    # like s.56 and behave like it.
    assert set(every) - set(default) == {"56", "58", "59", "61", "62"}


def test_spans_cover_the_text_without_overlapping():
    """An offset must fall in exactly one provision, or attributing by offset is
    ambiguous in a way no caller could detect."""
    text = strip_stars(pakistan_code(read("ppc_pakistan_code_pp39-41.txt")).text)
    spans = provision_spans(text, include_repealed=True)
    # strict=False on purpose: spans[1:] is one shorter, which is the pairing wanted.
    for (_, _, end), (_, next_start, _) in zip(spans, spans[1:], strict=False):
        assert end == next_start
    assert spans[-1][2] == len(text)


def test_a_provision_holding_nine_others_is_reported():
    """Reported for a while, and now parsed.

    `missing_from_contents` catches a contents entry with no body. Nothing caught a
    body with no contents entry - so when the contents stop at s.57 and the text runs
    on to s.66, nine sections were folded into s.57's body and the import reported no
    problem. Then it reported one, and the report was all it did: the nine stayed
    inside s.57, which made it 1,800 characters holding ten provisions. That is not
    cosmetic, and it cost a citation - BM25 penalises length, so s.57, whose own words
    answer "how many years is imprisonment for life", lost to s.302.

    The detector's own rule - an ascending run of numbers each followed by a
    capitalised title - is now what the parser uses to continue a contents list that
    stopped, so there is nothing left to report here.
    """
    text = strip_stars(pakistan_code(read("ppc_pakistan_code_pp39-41.txt")).text)
    rows, report = split_act(text, statute="PPC", in_force_from="2016-01-01")
    assert report["swallowed_headings"] == []
    # The nine are accounted for: five are provisions, four are repealed headings the
    # source itself brackets, and none of them is inside s.57 any more.
    numbers = {r["number"] for r in rows}
    assert {"60", "63", "64", "65", "66"} <= numbers
    assert not {"58", "59", "61", "62"} & numbers
    for number in ("58", "59", "61", "62"):
        assert any(line.startswith(f"section {number}:") for line in report["omitted_or_repealed"])
    assert "58." not in by_number(rows)["57"]["text"]


def test_a_run_the_parser_cannot_claim_is_still_reported():
    """The detector has to keep working, or removing its only live case retires it.

    The parser continues a contents list through CONSECUTIVE numbers, because that is
    what a page-range excerpt looks like. A body holding non-consecutive later headings
    is a different shape - a gap means something else is going on - and it is still
    reported rather than guessed at.
    """
    from paklaw.split import buried_numbers

    body = (
        "In calculating fractions of terms of punishment, imprisonment for life "
        "shall be reckoned as equivalent to imprisonment for twenty-five years.\n"
        "59. Transportation instead of imprisonment. Where any person is sentenced "
        "to transportation, the Court may award imprisonment.\n"
        "63. Amount of fine. Where no sum is expressed to which a fine may extend, "
        "the amount is unlimited.\n"
        "71. Limit of punishment of offence made up of several offences. Where "
        "anything is an offence falling within two definitions.\n"
    )
    assert buried_numbers(body, "57") == ["59", "63", "71"]

    # And a citation of an earlier section is not a heading, which is what keeps this
    # from firing on every provision in a statute book.
    assert buried_numbers("punishable under section 304 of this Code.", "57") == []


@pytest.mark.parametrize(
    "fixture,reader",
    [
        ("constitution_pakistan_code_pp18-20.txt", "pakistan-code"),
        ("ppc_pakistani_org_ss300-303.html", "pakistani-org"),
    ],
)
def test_a_well_formed_import_reports_nothing_swallowed(fixture, reader):
    """The check has to be quiet on the sources that are fine, or it is noise.

    A statute cites earlier sections constantly ("specified in Section 304"), so only
    an ascending run of numbers greater than the provision's own counts.
    """
    raw = read(fixture)
    if reader == "pakistan-code":
        text = strip_stars(pakistan_code(raw).text)
        unit, statute = "article", "CONST"
    else:
        text, _ = pakistani_org(raw)
        unit, statute = "section", "PPC"
    _, report = split_act(text, statute=statute, in_force_from="1900-01-01", unit=unit)
    assert report["swallowed_headings"] == []


# --- the rule that continues a contents list, and where it stops ----------------------


def _act(contents: list[str], body: str) -> str:
    """A minimal Act with a CONTENTS block, in the layout the importer reads."""
    listed = "\n".join(contents)
    return f"THE TEST ACT\n\n            CONTENTS\n\n{listed}\n\n{body}\n"


def _candidate(number: str, start: int, heading: str = "A heading"):
    """A weak candidate: the shape a run-in or bracketed heading produces."""
    from paklaw.split import _Candidate

    return _Candidate(
        start=start,
        body_start=start + 10,
        number=number,
        heading=heading,
        weight=1.0,
        order=(int("".join(c for c in number if c.isdigit())), 0),
    )


def test_two_consecutive_weak_headings_are_not_enough():
    """One ascending number is a citation; two could be two citations in order.

    A statute cites later sections constantly - "the procedure in sections 11 and 12
    applies" - so a rule that split on two ascending numbers would invent provisions
    out of a sentence. Three is the minimum, and this is what `MIN_TAIL_RUN` buys.
    """
    from paklaw.split import _tail_run

    chain = [_candidate("57", 0)]
    assert _tail_run(chain, [_candidate("58", 100), _candidate("59", 200)], "x" * 4000) == []


def test_three_consecutive_weak_headings_are_a_contents_list_that_stopped():
    from paklaw.split import _tail_run

    chain = [_candidate("57", 0)]
    weak = [_candidate("58", 100), _candidate("59", 200), _candidate("60", 300)]
    assert [c.number for c in _tail_run(chain, weak, "x" * 4000)] == ["58", "59", "60"]


def test_a_gap_ends_the_run():
    """3, 4, 5 then 9 is not a page-range excerpt. The run stops at the gap, and what
    is past it stays where it was found - which the detector then reports."""
    from paklaw.split import _tail_run

    chain = [_candidate("57", 0)]
    weak = [
        _candidate("58", 100),
        _candidate("59", 200),
        _candidate("60", 300),
        _candidate("64", 400),
    ]
    assert [c.number for c in _tail_run(chain, weak, "x" * 4000)] == ["58", "59", "60"]


def test_the_run_only_looks_after_the_last_accepted_provision():
    """A number earlier in the text than the last heading is a cross-reference in a
    body that has already been parsed, not a continuation of the contents."""
    from paklaw.split import _tail_run

    chain = [_candidate("57", 500)]
    weak = [_candidate("58", 100), _candidate("59", 200), _candidate("60", 300)]
    assert _tail_run(chain, weak, "x" * 4000) == []


def test_nothing_to_continue_is_not_a_run():
    """A sweep over an empty list passes, so both empty cases are stated."""
    from paklaw.split import _tail_run

    assert _tail_run([], [_candidate("58", 100)], "x" * 4000) == []
    assert _tail_run([_candidate("57", 0)], [], "x" * 4000) == []


def test_three_consecutive_unlisted_headings_continue_the_contents():
    """The shape the Pakistan Code excerpt has: a contents list that ends early."""
    text = _act(
        ["1. Short title", "2. Definitions"],
        "1. Short title. This Act may be cited as the Test Act.\n"
        "2. Definitions. In this Act, words have their ordinary meaning.\n"
        "3. Procedure. An application shall be made in writing.\n"
        "4. Appeals. An appeal lies to the High Court.\n"
        "5. Rules. The Federal Government may make rules.\n",
    )
    rows, report = split_act(text, statute="TEST", in_force_from="2020-01-01")
    assert [r["number"] for r in rows] == ["1", "2", "3", "4", "5"], [r["number"] for r in rows]
    assert report["swallowed_headings"] == []
    assert "Short title" in by_number(rows)["1"]["heading"]
    # And nothing from s.3 is left in s.2's body, which is the failure being fixed.
    assert "Procedure" not in by_number(rows)["2"]["text"]


def test_the_other_two_fixtures_are_unchanged_by_the_rule():
    """A rule that fires on a well-formed source is a rule that invents provisions.

    Both of these have contents that match their text, so the tail run must find
    nothing: the Constitution excerpt parses to articles 1-8 and the pakistani.org page
    to ss.300-303, as they did before.
    """
    constitution = strip_stars(pakistan_code(read("constitution_pakistan_code_pp18-20.txt")).text)
    rows, report = split_act(
        constitution, statute="CONST", in_force_from="1973-08-14", unit="article"
    )
    assert [r["number"] for r in rows] == ["1", "2", "2A", "3", "4", "5", "6", "7", "8"]
    assert report["swallowed_headings"] == []

    text, _ = pakistani_org(read("ppc_pakistani_org_ss300-303.html"))
    rows, report = split_act(text, statute="PPC", in_force_from="2016-01-01")
    assert [r["number"] for r in rows] == ["300", "301", "302", "303"]
    assert report["swallowed_headings"] == []


# -- the rule that separates a contents list from a numbered list ---------------------
#
# A default argument of "" turned this rule off, and every unit test above took the
# default, so the rule shipped with no test executing it and these are the first.


def test_a_run_of_bare_titles_is_a_list_and_not_a_run_of_sections():
    """The shape that found it, inside the last listed section's own text:

        3. Remedies. The court may award the following, in this order:
        4. Compensation for loss actually suffered.
        5. Costs of the proceedings.
        6. Interest from the date of the decree.

    4, 5 and 6 are clauses of s.3. Parsed as sections they truncated s.3 at the colon,
    dropped three substantive clauses, and were reported as omitted sections "(0
    characters)" - which is the discriminator, because a genuinely omitted section
    carries the note that says so and has a body.
    """
    from paklaw.split import _tail_run

    # Offsets chosen so each candidate's body runs up to the next candidate's start,
    # and every one of those spans is whitespace: a title and then immediately the
    # next title.
    text = " " * 400
    chain = [_candidate("3", 0)]
    weak = [_candidate("4", 100), _candidate("5", 200), _candidate("6", 300)]
    assert _tail_run(chain, weak, text) == []


def test_a_run_whose_members_have_bodies_is_a_run_of_sections():
    """The other direction, or the rule above is refusing every tail run rather than
    discriminating between the two shapes."""
    from paklaw.split import _tail_run

    text = "x" * 400
    chain = [_candidate("3", 0)]
    weak = [_candidate("4", 100), _candidate("5", 200), _candidate("6", 300)]
    assert [c.number for c in _tail_run(chain, weak, text)] == ["4", "5", "6"]


def test_one_member_with_a_body_is_enough():
    """`any`, not `all`: the last section of an excerpt is often the one that got cut
    off, and a run where some members have text is a run of sections."""
    from paklaw.split import _tail_run

    text = " " * 250 + "x" * 150
    chain = [_candidate("3", 0)]
    weak = [_candidate("4", 100), _candidate("5", 200), _candidate("6", 300)]
    assert [c.number for c in _tail_run(chain, weak, text)] == ["4", "5", "6"]


def test_the_text_is_not_optional():
    """It used to default to "", which turned the rule above off for every caller that
    forgot it - and every test in this file forgot it."""
    import inspect

    from paklaw.split import _tail_run

    text = inspect.signature(_tail_run).parameters["text"]
    assert text.default is inspect.Parameter.empty
