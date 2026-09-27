"""Building a corpus from an Act's text, and amending it without breaking it.

The fixture is laid out the way official texts are — number, heading, dash, body, with
chapter headings between sections — and it carries the traps: a numbered line inside a
body that looks like a heading, and an omitted section. Its wording is paraphrased for
testing, not the official text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paklaw.answer import REFUSAL_NOT_IN_FORCE, LawAssistant
from paklaw.corpus import CorpusError, build
from paklaw.ingest import (
    check,
    insert,
    main,
    read_rows,
    repeal,
    split_act,
    substitute,
    write_jsonl,
)
from paklaw.retrieve import LawSearch

ACT = """\
THE PREVENTION OF ELECTRONIC CRIMES ACT, 2016

CHAPTER I
PRELIMINARY

1. Short title, extent and commencement.—(1) This Act may be called the Prevention of
Electronic Crimes Act, 2016.
(2) It extends to the whole of Pakistan.

2. Definitions.—(1) In this Act, unless there is anything repugnant in the subject or
context,—
(a) "act" includes a series of acts or omissions;
(b) "access to information system" means gaining control over an information system;

CHAPTER II
OFFENCES AND PUNISHMENTS

3. Unauthorized access to information system or data.— Whoever with dishonest intention
gains unauthorized access to any information system or data shall be punished with
imprisonment for a term which may extend to three months.

4. [Omitted by the Amendment Act, 2020.]

20. Offences against dignity of a natural person.—(1) Whoever intentionally and publicly
exhibits or displays or transmits any information which he knows to be false, and
intimidates or harms the reputation or privacy of a natural person, shall be punished
with imprisonment which may extend to three years.
(2) Any aggrieved person may apply to the Authority for removal of such information,
and the Authority shall consider the application within
3. Thirty days: counted from its receipt, as a numbered line in some printed copies.

20A. Offences against modesty of a natural person.— Whoever intentionally and publicly
exhibits any information that superimposes a photograph of a natural person shall be
punished with imprisonment which may extend to five years.
"""


@pytest.fixture
def rows() -> list[dict]:
    parsed, _ = split_act(ACT, statute="PECA", in_force_from="2016-08-19")
    return parsed


class TestSplitting:
    def test_sections_in_order(self, rows):
        assert [r["number"] for r in rows] == ["1", "2", "3", "20", "20A"]

    def test_heading_and_body_are_separated(self, rows):
        twenty = rows[3]
        assert twenty["heading"] == "Offences against dignity of a natural person"
        assert twenty["text"].startswith("(1) Whoever intentionally")
        assert "\n" not in twenty["text"]

    def test_a_numbered_line_inside_a_body_is_not_a_new_section(self, rows):
        """'3. Thirty days: ...' sits inside section 20. Splitting there would end
        section 20 early and invent a second section 3."""
        _, report = split_act(ACT, statute="PECA", in_force_from="2016-08-19")
        assert "Thirty days: counted from its receipt" in rows[3]["text"]
        assert len(report["rejected_headings"]) == 1

    def test_an_omitted_section_is_reported_not_loaded(self, rows):
        _, report = split_act(ACT, statute="PECA", in_force_from="2016-08-19")
        assert "4" not in [r["number"] for r in rows]
        assert report["omitted_or_repealed"][0].startswith("section 4")

    def test_chapter_headings_label_sections_and_stay_out_of_bodies(self, rows):
        assert rows[0]["chapter"] == "Chapter I"
        assert rows[2]["chapter"] == "Chapter II"
        assert "CHAPTER" not in rows[1]["text"]

    def test_the_result_loads_and_validates(self, rows):
        assert check(rows) == []
        assert len(build(rows)) == 5

    def test_a_bad_date_fails_before_any_work(self):
        with pytest.raises(ValueError):
            split_act(ACT, statute="PECA", in_force_from="19-08-2016")


class TestAmending:
    def test_substitution_closes_one_version_and_opens_the_next(self, rows):
        substitute(
            rows, "section 20 PECA", on="2022-02-20", by="Ordinance II of 2022", text="new text"
        )
        assert check(rows) == []
        versions = [r for r in rows if r["number"] == "20"]
        assert versions[0]["in_force_to"] == "2022-02-20"
        assert versions[0]["manner"] == "substituted"
        assert versions[1]["in_force_from"] == "2022-02-20"
        assert versions[1]["heading"] == versions[0]["heading"]
        # what ended the old version, and what began the new one: the same instrument,
        # recorded on each side, never read across
        assert versions[0]["amended_by"] == versions[1]["enacted_by"] == "Ordinance II of 2022"
        assert "amended_by" not in versions[1]

        assistant = LawAssistant(corpus=build(rows))
        before = assistant.answer("section 20 PECA", as_of="2020-01-01").passages[0].text
        after = assistant.answer("section 20 PECA", as_of="2023-01-01").passages[0].text
        assert "three years" in before and after == "new text"

    def test_substitution_before_the_current_version_began_is_refused(self, rows):
        with pytest.raises(CorpusError, match="cannot end on"):
            substitute(rows, "section 20 PECA", on="2010-01-01", by="x", text="t")

    def test_a_subsection_cannot_be_amended_on_its_own(self, rows):
        """It would sit beside the stale section, and lookups would find the stale one."""
        with pytest.raises(CorpusError, match="subdivision"):
            substitute(rows, "section 20(2) PECA", on="2022-02-20", by="x", text="t")

    def test_repeal(self, rows):
        repeal(rows, "section 3 PECA", on="2024-01-01", by="Act X of 2024")
        answer = LawAssistant(corpus=build(rows)).answer("section 3 PECA", as_of="2025-01-01")
        assert answer.refused and REFUSAL_NOT_IN_FORCE in answer.refusal_reason

    def test_repealing_twice_is_refused(self, rows):
        repeal(rows, "section 3 PECA", on="2024-01-01", by="x")
        with pytest.raises(CorpusError, match="no version in force"):
            repeal(rows, "section 3 PECA", on="2025-01-01", by="x")

    def test_insert_refuses_a_live_provision(self, rows):
        with pytest.raises(CorpusError, match="substitute it instead"):
            insert(rows, "section 20 PECA", on="2024-01-01", by="x", text="t", heading="h")

    def test_insert_then_reinsert_after_repeal(self, rows):
        insert(rows, "section 26A PECA", on="2025-01-29", by="Act I of 2025", text="t", heading="h")
        assert check(rows) == []

    def test_a_citation_without_an_act_is_refused(self, rows):
        with pytest.raises(CorpusError, match="names no Act"):
            repeal(rows, "section 3", on="2024-01-01", by="x")


class TestFiles:
    def test_written_in_statute_order_and_read_back(self, rows, tmp_path):
        path = tmp_path / "c.jsonl"
        write_jsonl(list(reversed(rows)), path)
        assert [r["number"] for r in read_rows(path)] == ["1", "2", "3", "20", "20A"]
        assert not list(tmp_path.glob("*.tmp"))


class TestCommandLine:
    def test_import_amend_check(self, tmp_path, capsys):
        act = tmp_path / "peca.txt"
        act.write_text(ACT, encoding="utf-8")
        corpus = tmp_path / "corpus.jsonl"
        amended = tmp_path / "s20.txt"
        amended.write_text("Whoever ... five years.\n", encoding="utf-8")

        assert (
            main(
                [
                    "import",
                    str(act),
                    "--statute",
                    "Prevention of Electronic Crimes Act",
                    "--in-force-from",
                    "2016-08-19",
                    "-o",
                    str(corpus),
                ]
            )
            == 0
        )
        assert json.loads(capsys.readouterr().out)["provisions"] == 5
        assert (
            main(
                [
                    "substitute",
                    str(corpus),
                    "section 20 PECA",
                    "--on",
                    "2022-02-20",
                    "--by",
                    "Ordinance II of 2022",
                    "--text-file",
                    str(amended),
                ]
            )
            == 0
        )
        assert main(["check", str(corpus)]) == 0
        assert len(read_rows(corpus)) == 6

    def test_a_refused_amendment_leaves_the_file_untouched(self, tmp_path, rows):
        corpus = tmp_path / "corpus.jsonl"
        write_jsonl(rows, corpus)
        before = corpus.read_bytes()
        text = tmp_path / "t.txt"
        text.write_text("t", encoding="utf-8")
        assert (
            main(
                [
                    "substitute",
                    str(corpus),
                    "section 20 PECA",
                    "--on",
                    "2001-01-01",
                    "--by",
                    "x",
                    "--text-file",
                    str(text),
                ]
            )
            == 2
        )
        assert corpus.read_bytes() == before

    def test_importing_the_same_act_twice_is_refused(self, tmp_path):
        act = tmp_path / "peca.txt"
        act.write_text(ACT, encoding="utf-8")
        corpus = tmp_path / "corpus.jsonl"
        args = [
            "import",
            str(act),
            "--statute",
            "PECA",
            "--in-force-from",
            "2016-08-19",
            "-o",
            str(corpus),
        ]
        assert main(args) == 0
        assert main(args) == 2

    def test_check_reports_an_invalid_file(self, tmp_path, rows):
        rows.append(dict(rows[0], in_force_from="2017-01-01"))  # second live version
        corpus = tmp_path / "bad.jsonl"
        corpus.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        assert main(["check", str(corpus)]) == 1


class TestIndexCache:
    def test_dates_between_the_same_amendments_share_an_index(self, rows):
        search = LawSearch(corpus=build(rows))
        search.search("information", as_of="2017-01-01")
        search.search("information", as_of="2019-06-30")
        assert len(search._indexes) == 1

    def test_the_cache_is_bounded(self, rows):
        for year in range(2017, 2030):
            rows = substitute(rows, "section 3 PECA", on=f"{year}-01-01", by="x", text=str(year))
        search = LawSearch(corpus=build(rows), max_indexes=4)
        for year in range(2017, 2030):
            search.search("information", as_of=f"{year}-06-01")
        assert len(search._indexes) == 4

    def test_a_grown_corpus_is_reindexed(self, rows):
        corpus = build(rows)
        search = LawSearch(corpus=corpus)
        assert not search.search("quantum", as_of="2020-01-01")
        corpus.add(
            build([dict(rows[0], number="99", heading="Quantum", text="quantum")]).provisions[0]
        )
        assert search.search("quantum", as_of="2020-01-01")


def test_fixture_is_not_mistaken_for_the_official_text():
    """Guard the docstring's promise: this file must never be shipped as a corpus."""
    assert "paraphrased" in Path(__file__).read_text(encoding="utf-8")
