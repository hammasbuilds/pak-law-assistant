"""Print the retrieval-quality table the README quotes.

Not a test. Run it to see the figures:

    .venv/Scripts/python tests/bench.py

The README's "Measured" row states three question sets and their results - 17 in a
person's words, 34 paraphrases, 20 written later again, plus the subjects the corpus does
not hold. Those numbers were asserted by `test_retrieval_quality.py` and printed by
nothing: the thresholds there are floors (`right >= MIN_PARAPHRASE_RIGHT`), so a set that
had quietly got *better* than the README said would pass, and so would one whose refusal
split had moved entirely. A reader could not see the table without reading the test
source and running the numbers by hand.

This prints it, and `test_the_readme_table_is_the_one_this_prints` checks the README
against what it prints, so the two cannot drift apart.

The question sets themselves stay in the test module, because that is where the
assertions about them live and one definition is the point.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_retrieval_quality import (  # noqa: E402
    ANSWERABLE,
    FRESH,
    FRESH_MUST_REFUSE,
    MUST_REFUSE,
    PARAPHRASES,
    UNANSWERABLE,
    _corpus_rows,
)

from paklaw.answer import LawAssistant  # noqa: E402
from paklaw.ingest import build_checked  # noqa: E402


class Result:
    """One question set's outcome, with every question accounted for."""

    def __init__(self, name: str, note: str):
        self.name = name
        self.note = note
        self.right: list[str] = []
        self.wrong: list[tuple[str, str, str]] = []
        self.refused: list[str] = []

    @property
    def total(self) -> int:
        return len(self.right) + len(self.wrong) + len(self.refused)

    def row(self) -> str:
        return (
            f"  {self.name:<34} {self.total:>4}  {len(self.right):>6} "
            f"{len(self.wrong):>6} {len(self.refused):>8}"
        )


def score(assistant: LawAssistant, name: str, note: str, pairs) -> Result:
    out = Result(name, note)
    for question, expected in pairs:
        answer = assistant.answer(question, as_of="2026-01-01")
        if answer.refused:
            out.refused.append(question)
        elif answer.passages[0].citation == expected:
            out.right.append(question)
        else:
            out.wrong.append((question, expected, answer.passages[0].citation))
    return out


def refusals(assistant: LawAssistant, name: str, note: str, questions) -> Result:
    """A set where the only right answer is a refusal."""
    out = Result(name, note)
    for question in questions:
        answer = assistant.answer(question, as_of="2026-01-01")
        if answer.refused:
            out.refused.append(question)
        else:
            out.wrong.append((question, "a refusal", answer.passages[0].citation))
    return out


def main() -> int:
    assistant = LawAssistant(corpus=build_checked(_corpus_rows(), source="fixtures"))

    answerable = [
        score(assistant, "in a person's words", "written with the retriever", ANSWERABLE),
        score(assistant, "paraphrases", "written against the provisions", PARAPHRASES),
        score(assistant, "written later again", "written from the provisions alone", FRESH),
    ]
    must_refuse = [
        refusals(assistant, "not in corpus (set 1)", "", UNANSWERABLE),
        refusals(assistant, "not in corpus (set 2)", "", MUST_REFUSE),
        refusals(assistant, "not in corpus (set 3)", "", FRESH_MUST_REFUSE),
    ]

    print("\nRETRIEVAL QUALITY")
    print("=" * 70)
    print(f"  {'question set':<34} {'n':>4}  {'right':>6} {'wrong':>6} {'refused':>8}")
    print("  " + "-" * 64)
    for result in answerable:
        print(result.row())
    print("  " + "-" * 64)
    for result in must_refuse:
        # Refusal is the right answer here, so `refused` is the score and `wrong` is an
        # answer that should not have been given. Printed in the same columns rather
        # than relabelled, because the shape of the table is the claim.
        print(result.row())
    print("  " + "-" * 64)

    right = sum(len(r.right) for r in answerable)
    wrong = sum(len(r.wrong) for r in answerable + must_refuse)
    asked = sum(r.total for r in answerable + must_refuse)
    held_back = sum(len(r.refused) for r in must_refuse)
    print(
        f"  {asked} questions asked: {right} answered correctly, {wrong} wrong, "
        f"{held_back} of {sum(r.total for r in must_refuse)} correctly refused"
    )
    print(
        "  'wrong' is a confident answer citing a different provision. It is the only\n"
        "  column that matters: a refusal costs a reader a lookup, a wrong citation\n"
        "  costs them the argument."
    )

    if wrong:
        print("\n  WRONG ANSWERS")
        for result in answerable + must_refuse:
            for question, expected, got in result.wrong:
                print(f"    {result.name}: {question!r}\n      expected {expected}, got {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
