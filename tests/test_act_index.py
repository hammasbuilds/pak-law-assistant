"""Reading the Pakistan Code's header and its index annotations.

Every title quoted here is one the Code actually publishes, spelling and stray brackets
included. They are the input this parser exists for, and each of the odd ones broke an
earlier version of it.
"""

from __future__ import annotations

import datetime as dt

from paklaw import act_index

# ---- the date in the header ----------------------------------------------------------

HEADER = """
                       THE ANTI-TERRORISM ACT, 1997

                               The Anti-Terrorism Act, 1997

                                   1Act No. XXVII of 1997

                                                          [16th August, 1997]

An Act to provide for the prevention of terrorism.
"""


def test_the_assent_date_is_read_from_the_header():
    assert act_index.assent_date(HEADER) == dt.date(1997, 8, 16)


def test_the_act_number_is_read_from_the_header():
    assert act_index.act_number(HEADER) == "Act XXVII of 1997"


def test_a_header_without_a_date_gives_none_rather_than_the_title_year():
    """The caller then records start_known=False. A year turned into 1 January would
    answer a question about March with an Act that did not exist until August."""
    assert act_index.assent_date("THE ANTI-TERRORISM ACT, 1997\n\nAn Act to provide") is None


def test_an_impossible_date_in_brackets_is_not_a_date():
    assert act_index.assent_date("Act No. I of 1997 [31st June, 1997]") is None


def test_a_date_buried_in_running_text_is_not_the_acts_own():
    """A bracketed date deep inside a sentence is a commencement notified for one
    section, or part of an illustration."""
    buried = "An Act\n" + ("x " * act_index.HEADER) + " shall commence [16th August, 1997] and"
    assert act_index.assent_date(buried) is None


def test_a_date_alone_on_a_line_is_the_acts_own_however_far_in():
    """The Constitution's 176-page table of contents puts its "[12th April, 1973]" some
    30,000 characters in, and it was left out of the corpus entirely for having "no date
    in the header and no year in the title"."""
    contents = "THE CONSTITUTION\n\nCONTENTS\n" + ("1. Some article heading\n" * 2000)
    assert act_index.assent_date(contents + "\n          [12th April, 1973]\n") == dt.date(
        1973, 4, 12
    )


# ---- the repeal note in the title ------------------------------------------------------

EXACT = "Agricultural Census Act, 1958 (Repealed by Act XiV of 2011 s.3(w.e.f 31-05-2011))"
YEAR_ONLY = "Alternative Energy Development Board Act, 2010 (Repealed by Act No.XXVI of 2023)"
TWO_DIGIT = "Trade Marks Act,1940(Act No.V of 1940)(Repealed by Ord.19 of 01 s.134)"
TYPO = "National Insurance Corporation Act, 1976(Repeal by ord.37 of 200,s.9(when notified))"


def test_a_title_with_no_note_is_not_repealed():
    assert act_index.repeal("Contract Act, 1872") is None


def test_an_exact_w_e_f_date_is_taken_as_exact():
    found = act_index.repeal(EXACT)
    assert found == {"date": dt.date(2011, 5, 31), "exact": True, "by": "Act XIV of 2011"}
    assert act_index.ended(dt.date(1958, 4, 1), found) == (dt.date(2011, 5, 31), True)


def test_a_year_only_note_is_the_first_of_that_year_and_not_exact():
    found = act_index.repeal(YEAR_ONLY)
    assert found["date"] == dt.date(2023, 1, 1)
    assert found["exact"] is False
    assert found["by"] == "Act XXVI of 2023"


def test_a_two_digit_repealing_year_is_resolved():
    """Eight Acts were left looking live because "of 01" is not four digits."""
    found = act_index.repeal(TWO_DIGIT)
    assert found["date"] == dt.date(2001, 1, 1)
    assert found["by"] == "Ordinance 19 of 2001"


def test_a_three_digit_year_is_a_typo_not_a_year():
    """ "ord.37 of 200" once matched its first two digits and recorded the repeal as
    2020 - asserting that a 1976 Act was in force for another 44 years."""
    found = act_index.repeal(TYPO)
    assert found is not None, "the title says repealed, so it must not read as live"
    assert found["date"] is None
    assert act_index.ended(dt.date(1976, 1, 1), found) == (dt.date(1976, 1, 2), False)


def test_a_repeal_in_the_year_of_commencement_ends_after_it_begins():
    """1 January of the repeal year precedes commencement, and a provision that ends
    before it starts is refused by the corpus."""
    found = act_index.repeal("Some Act, 2010 (Repealed by Act I of 2010)")
    end, exact = act_index.ended(dt.date(2010, 6, 1), found)
    assert end == dt.date(2010, 6, 2)
    assert exact is False


def test_every_published_form_of_the_note_is_recognised():
    """The eight below are every shape the Code's own index uses, and all eight were
    unparsed at some point. An unparsed note is an Act served as current law."""
    for title in (
        EXACT,
        YEAR_ONLY,
        TWO_DIGIT,
        TYPO,
        "Agricultural Development Bank Ordinance 1961(Repealed by Ord. 60 of 2002,s.2)",
        "Apprenticeship Ordinance, 1962 (Repeal by Act I of 2019,s.18)",
        "Khushali Bank Ordinance 2000 (Repealed by Act I of 08)",
        "University Grants Commission Act, 1974 (Repealed by Ord 53 of 02,s.22)",
    ):
        assert act_index.repeal(title) is not None, title


# ---- the title itself -------------------------------------------------------------------


def test_the_repeal_note_is_not_part_of_the_acts_name():
    assert act_index.strip_annotations(EXACT) == "Agricultural Census Act, 1958"


def test_stripping_leaves_the_brackets_balanced():
    """`.strip(")")` leaves "(QSO", which would then appear in the provenance line of
    every provision of that Order."""
    assert (
        act_index.strip_annotations("Qanun-e-Shahadat Order, 1984 (QSO)")
        == "Qanun-e-Shahadat Order, 1984 (QSO)"
    )
    assert act_index.strip_annotations(
        "Trade Marks Act,1940(Act No.V of 1940)(Repealed by Ord.19 of 01 s.134)"
    ) == ("Trade Marks Act,1940(Act No.V of 1940)")


def test_an_editorial_mark_is_not_part_of_the_name_either():
    assert act_index.strip_annotations("Pakistan Penal Code (PPC),1860 (Under Review)") == (
        "Pakistan Penal Code (PPC),1860"
    )


# ---- statute keys -------------------------------------------------------------------------


def test_the_codes_people_cite_are_not_generated():
    keys = act_index.assign_keys(
        [
            "Pakistan Penal Code (PPC),1860",
            "Constitution of the Islamic  Republic of Pakistan",
            "Code of Criminal Procedure (CrPC), 1898",
            "Qanun-e-Shahadat Order, 1984 (QSO)",
        ]
    )
    assert keys["Pakistan Penal Code (PPC),1860"] == "PPC"
    assert keys["Constitution of the Islamic  Republic of Pakistan"] == "CONST"
    assert keys["Code of Criminal Procedure (CrPC), 1898"] == "CrPC"
    assert keys["Qanun-e-Shahadat Order, 1984 (QSO)"] == "QSO"


def test_an_ordinary_act_gets_its_initials():
    title = "Anti-Money Laundering Act, 2010"
    assert act_index.assign_keys([title])[title] == "AML"


def test_a_key_is_never_short_enough_to_be_an_english_word():
    """ "Antiquities Act, 1975" initialised to "A", which then matched the article "a"
    as a whole word in every question and scoped it to that Act."""
    from paklaw.citation import NOT_A_STATUTE_NAME

    for title in ("Antiquities Act, 1975", "Boilers Act, 1923", "Anti-Terrorism Act, 1997"):
        key = act_index.assign_keys([title])[title]
        assert len(key) >= act_index.MIN_KEY, (title, key)
        assert key.lower() not in NOT_A_STATUTE_NAME, (title, key)


def test_two_acts_never_share_a_key():
    """A shared key merges two statutes: every lookup of either returns both."""
    titles = [
        "Pakistan Academy Act, 1950",
        "Pakistan Academy Act, 1976",
        "Pakistan Arbitration Act, 1950",
    ]
    keys = act_index.assign_keys(titles)
    assert len(set(keys.values())) == len(titles)


def test_a_title_with_no_significant_words_still_gets_a_key():
    assert act_index.assign_keys(["The Act, 1950"])["The Act, 1950"]


def test_an_act_whose_purpose_is_to_repeal_is_not_an_annotation():
    """ "... (Conversion) (Repeal) Act, 2026" is an Act whose purpose is to repeal
    something: "(Repeal)" is part of its name. Read as an annotation it stripped back to
    the Conversion Act - which the repealing ORDINANCE of the same year also stripped to
    - so two different instruments shared one statute key and the corpus held two
    conflicting "section 1"s."""
    title = "Pakistan International Airlines Corporation (Conversion) (Repeal) Act, 2026"
    assert act_index.repeal(title) is None
    assert act_index.strip_annotations(title) == title


def test_a_bare_repealed_in_brackets_is_still_an_annotation():
    title = "Some Act, 1960 (Repealed)"
    assert act_index.repeal(title) is not None
    assert act_index.strip_annotations(title) == "Some Act, 1960"
