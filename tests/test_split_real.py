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
        ]
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
    assert json.loads(captured.out)["provisions"] == 13
    assert "imported 13 provision(s)" in captured.err
    assert "1 omitted or repealed" in captured.err


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
    assert set(every) - set(default) == {"56"}


def test_spans_cover_the_text_without_overlapping():
    """An offset must fall in exactly one provision, or attributing by offset is
    ambiguous in a way no caller could detect."""
    text = strip_stars(pakistan_code(read("ppc_pakistan_code_pp39-41.txt")).text)
    spans = provision_spans(text, include_repealed=True)
    # strict=False on purpose: spans[1:] is one shorter, which is the pairing wanted.
    for (_, _, end), (_, next_start, _) in zip(spans, spans[1:], strict=False):
        assert end == next_start
    assert spans[-1][2] == len(text)
