"""Three questions: one answered, one answered as of a past date, one refused.

    python demo.py

The corpus carries PECA s.20 twice -- the text in force until 20 Feb 2022 and
the text substituted that day. Which one is correct depends entirely on the
date the question is about. No network, no model, no dependencies.
"""

import contextlib
import sys

sys.path.insert(0, "src")

from paklaw.answer import LawAssistant
from paklaw.corpus import Corpus, Provision

# Windows defaults the console to the ANSI code page - cp437 on a US install, cp850
# in parts of Europe - and the corpus prose this prints contains an em dash.
# `demo_mcp.py` exited 1 with `UnicodeEncodeError` on exactly that console while
# the server's `--check` and its protocol channel had both been fixed for it. This
# one survives today because of which strings it happens to print, which is not a
# reason to leave it depending on that.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PECA_OLD = (
    "Whoever intentionally and publicly exhibits or displays or transmits any information "
    "which he knows to be false, and intimidates or harms the reputation or privacy of a "
    "natural person, shall be punished with imprisonment which may extend to three years "
    "or with fine which may extend to one million rupees or with both."
)
PECA_NEW = (
    "Whoever intentionally and publicly exhibits or displays or transmits any information "
    "through any information system, which he knows to be false, and intimidates or harms "
    "the reputation or privacy of a natural person, shall be punished with imprisonment "
    "which may extend to five years or with fine which may extend to ten million rupees "
    "or with both."
)


def build() -> Corpus:
    c = Corpus()
    c.add(
        Provision(
            statute="PECA",
            unit="section",
            number="20",
            heading="Offences against dignity of a natural person",
            text=PECA_OLD,
            in_force_from="2016-08-19",
            in_force_to="2022-02-20",
            manner="substituted",
            amended_by="Ordinance II of 2022",
            superseded_by="PECA s.20 as amended 2022",
        )
    )
    c.add(
        Provision(
            statute="PECA",
            unit="section",
            number="20",
            heading="Offences against dignity of a natural person",
            text=PECA_NEW,
            in_force_from="2022-02-20",
        )
    )
    c.add(
        Provision(
            statute="PPC",
            unit="section",
            number="302",
            heading="Punishment of qatl-i-amd",
            text="Whoever commits qatl-i-amd shall be punished with death as qisas, or with "
            "death or imprisonment for life as tazir.",
            in_force_from="1997-04-11",
        )
    )
    c.add(
        Provision(
            statute="CONST",
            unit="article",
            number="25",
            heading="Equality of citizens",
            text="All citizens are equal before law and are entitled to equal protection of "
            "law. There shall be no discrimination on the basis of sex.",
            in_force_from="1973-08-14",
        )
    )
    return c


assistant = LawAssistant(corpus=build())

ASKS = [
    (
        "What is the penalty for publicly transmitting false information about a person?",
        "2026-01-01",
        "asked about today",
    ),
    (
        "What is the penalty for publicly transmitting false information about a person?",
        "2020-06-01",
        "the same question, asked about 2020",
    ),
    (
        "What are the registration requirements for a private limited company?",
        "2026-01-01",
        "nothing in the corpus covers this",
    ),
]

corpus = build()
provisions = len({(p.statute, p.unit, p.number) for p in corpus.provisions})
versions = len(corpus.provisions)

print("INPUT")
print(
    f"   corpus             {provisions} provisions, {versions} versions "
    "(PECA s.20 twice: pre- and post-2022)"
)
for q, as_of, _note in ASKS:
    print(f'   as_of {as_of}   "{q[:58]}..."')
print()

print("OUTPUT")
for n, (q, as_of, note) in enumerate(ASKS, 1):
    a = assistant.answer(q, as_of=as_of)
    print(f"   [{n}] {note}")
    if a.refused:
        print(f"       REFUSED    {a.refusal_reason}")
    else:
        for p in a.passages:
            note_txt = f"  [{p.status_note}]" if p.status_note else ""
            print(f"       {p.citation}{note_txt}")
            print(f"       in force   {p.in_force_from} -> {p.in_force_to or 'present'}")
            # The operative clause is the penalty, and it is at the end of the text.
            penalty = p.text[p.text.index("shall be punished") :]
            print(f"       ...{penalty}")
    print()
