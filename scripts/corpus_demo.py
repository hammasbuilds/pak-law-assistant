"""What the federal statute book answers, and what it refuses.

    python scripts/corpus_demo.py                      # the shipped corpus
    python scripts/corpus_demo.py path/to/corpus.jsonl

Every line below is printed by this script from the corpus named above. No network, no
model, no dependencies.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from paklaw.mcp_server import LawServer, load_corpus  # noqa: E402

# Windows defaults the console to the ANSI code page, and statute prose is not ASCII.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

SHIPPED = ROOT / "data" / "pakistan-code.jsonl.gz"
AS_OF = "2026-10-17"

#: Chosen to exercise the corpus rather than to flatter it: four that it answers from
#: four different Acts, and two it should refuse - one naming another country's statute,
#: one about a subject the federal statute book does not legislate on.
QUESTIONS = [
    "Are all citizens equal before the law?",
    "What is the offence of money laundering?",
    "What rights does a deaf person have to access the media?",
    "What is the punishment for an act of terrorism?",
    "What is the punishment for murder under the Indian Penal Code?",
    "How much should I pay for a used car in Lahore?",
]


def main(argv: list[str]) -> int:
    path = argv[0] if argv else str(SHIPPED)
    if not Path(path).exists():
        print(f"  {path} is not here.")
        print("  Build it with: python scripts/build_corpus.py catalogue | fetch | build")
        return 1

    corpus, source = load_corpus(path)
    server = LawServer(corpus, source)
    statutes = {provision.statute for provision in corpus.provisions}
    repealed = sum(1 for provision in corpus.provisions if provision.manner == "repealed")
    imprecise = sum(1 for provision in corpus.provisions if not provision.end_known)
    unknown_start = sum(1 for provision in corpus.provisions if not provision.start_known)

    print(f"\n  {len(corpus):,} provisions from {len(statutes)} Acts, as at {corpus.as_at}")
    print(f"  {repealed:,} of them repealed, {imprecise:,} on a date the source does not give")
    print(f"  {unknown_start:,} whose record starts before the date held for them\n")

    for question in QUESTIONS:
        answer = server.answer_question({"question": question, "as_of": AS_OF})
        print(f"  Q  {question}")
        if answer["refused"]:
            print(f"     REFUSED [{answer['refusal_status']}]")
            print(f"     {answer['refusal_reason'][:150]}")
        for passage in answer["passages"][:1]:
            print(f"     {passage['citation']} - {passage['heading']}")
            print(f"     {passage['text'][:160]}")
            if passage.get("status_note"):
                print(f"     ! {passage['status_note'][:120]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
