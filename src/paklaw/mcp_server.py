"""The statute book as an MCP server — standard library only.

    paklaw-mcp --corpus statutes.jsonl
    python -m paklaw.mcp_server --corpus statutes.jsonl

Speaks the Model Context Protocol over stdio: newline-delimited JSON-RPC 2.0 on stdin
and stdout, diagnostics on stderr. Written against the protocol rather than an SDK so the
package keeps `dependencies = []`; the tests drive it as a subprocess, byte for byte, the
way a client does.

What it exposes is the library's behaviour, not a looser version of it:

  **answer_question**    cite a provision in force on a date, or refuse — the nine
                         refusal conditions come through as answers, not errors
  **check_citations**    every citation in a draft, checked against the law on a date
  **provision_history**  the amendment trail of a named provision
  **compare_versions**   what an amendment changed, word by word
  **list_provisions**    a statute's table of contents on a date
  **changes_between**    every commencement, substitution and repeal in a period
  **parse_citations**    every citation in a passage, canonicalised; needs no corpus
  **corpus_info**        what the loaded corpus actually covers

Two choices are deliberate.

**Dates are required.** The library defaults `as_of` to today, but a model calling a tool
fills an optional argument by omission, and the question most likely to need a past
date — conduct before an amendment — is exactly the one where today is wrong.

**There is no bare search tool.** Raw BM25 hits are the nearest provisions, whether or
not they are relevant, and a model handed a ranked list cites the top one. The only way
to reach retrieval is through `answer_question`, which applies the weak-match refusal.

A corpus is loaded once, at start, and **validated before the server accepts a
request**: overlapping versions make answers depend on iteration order, so a corpus
with any structural problem is refused rather than served. With no corpus configured
the server runs on a three-provision sample, and says so in every result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import traceback
from collections import Counter
from importlib import resources
from typing import Any, BinaryIO

from . import __version__
from .answer import Answer, LawAssistant
from .audit import (
    ALL_CHANGE_KINDS,
    ALL_STATUSES,
    changes_between,
    check_citations,
    compare_versions,
    contents,
)
from .citation import CANONICAL_STATUTES, normalise_statute, parse, statute_aliases
from .corpus import Corpus, CorpusError
from .ingest import build_checked, read_corpus

# Newest first. A client asking for one of these gets it; anything else gets the newest.
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
#: Asked for work before the protocol version was agreed. Not one of JSON-RPC's own
#: codes - the -32000..-32099 range is reserved for the server to define, and -32002
#: is what MCP implementations conventionally use for this.
NOT_INITIALIZED = -32002
#: The two methods that are answerable before the handshake: one *is* the handshake,
#: and `ping` has no response shape to negotiate.
BEFORE_INITIALIZE = ("initialize", "ping")
# structuredContent arrived in 2025-06-18; older clients get the text block only.
_STRUCTURED = {"2025-11-25", "2025-06-18"}

SERVER_NAME = "pak-law-assistant"
CORPUS_ENV = "PAKLAW_CORPUS"

SAMPLE_WARNING = (
    "sample corpus: three demonstration provisions (PECA s.20 in two versions, PPC s.302, "
    "Article 25) with illustrative, unofficial wording — not the statute book and not to "
    f"be relied on. Set {CORPUS_ENV} to a corpus file to answer real questions."
)

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

LIST_LIMIT = 200
EVENT_LIMIT = 500
CITATION_LIMIT = 500
# A draft, a judgment, a chapter of an Act: comfortably under this. Past it the answer
# would be several megabytes, which no client displays and no model reads.
TEXT_LIMIT = 200_000
# How much of the question comes back in the result. A client sends the question, so
# echoing all of it doubles it on the wire for no information: a 100,000-character
# question produced a 98 KB reply to say "refused". Enough to correlate a result with a
# request, and the result says when it has been shortened.
ECHO_LIMIT = 2_000

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

INSTRUCTIONS = (
    "Pakistani statutes, versioned by date. Every provision carries the dates it was in "
    "force, so always pass the date the question is ABOUT — the date of the conduct, the "
    "contract or the hearing — not today's date by habit. "
    "A refusal is an answer: it means the corpus cannot support one, and must not be "
    "replaced with a provision from memory. Quote passage text as returned and cite it "
    "exactly; the text is the law, anything you add is not. "
    "Before relying on citations in any draft — yours or the user's — run check_citations. "
    "Call corpus_info first to see which statutes are loaded; a statute that is not loaded "
    "cannot be answered from or verified."
)


def _date_property(what: str) -> dict:
    return {"type": "string", "description": f"YYYY-MM-DD. {what}"}


_STATUTE = {
    "type": "string",
    "description": (
        "One statute, by key or common name (PPC, CrPC, CPC, CONST, PECA, "
        "'Pakistan Penal Code', ...). corpus_info lists what is loaded."
    ),
}
_READ_ONLY = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}


# Every result carries this when the corpus is the demonstration sample, and nothing
# when it is a real one - so it is declared and never required.
_CORPUS_WARNING = {
    "type": "string",
    "description": "Present only when the loaded corpus is the demonstration sample.",
}
# Paging. `total` is the whole result; `next_offset` appears only while more remains, so
# a page without it is the last page and not the whole answer.
_PAGING = {
    "total": {"type": "integer", "description": "Size of the whole result, not this page."},
    "next_offset": {
        "type": "integer",
        "description": "Offset of the next page. Absent on the last page.",
    },
}
_WARNINGS = {
    "warnings": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Coverage limits that bear on this answer, such as a statute the "
        "question named that the corpus does not hold, or a date range taken in the "
        "order it was meant rather than the order it was given.",
    },
}
_REFUSAL = {
    "refused": {"type": "boolean"},
    "refusal_reason": {
        "type": ["string", "null"],
        "description": "Why no answer is given, in prose. A refusal is an answer; it is "
        "not an error.",
    },
    "refusal_status": {
        "type": ["string", "null"],
        "enum": [
            "not_in_force",
            "unknown_provision",
            "act_not_recognised",
            "no_act_named",
            "nothing_matched",
            "weak_match",
            "different_offence",
            "subject_not_in_corpus",
            "citation_unreliable",
            "statute_not_loaded",
            None,
        ],
        "description": "The SAME refusal, as a token to branch on. `refusal_reason` is "
        "prose and will be reworded; branch on this. Names match check_citations' "
        "statuses where the two tools mean the same thing. Null when answered.",
    },
}


def _tool(
    name: str,
    title: str,
    description: str,
    properties: dict,
    required: list,
    output: dict | None = None,
    output_required: list | None = None,
) -> dict:
    """One tool. `output` describes `structuredContent`, which every tool here returns.

    Declared because a client that is handed structured results and no schema for them
    has to guess, and `additionalProperties` is left open: a corpus with richer metadata
    adds keys, and a client must not reject a result for carrying more than this.
    """
    tool = {
        "name": name,
        "title": title,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "annotations": _READ_ONLY,
    }
    if output is not None:
        tool["outputSchema"] = {
            "type": "object",
            "properties": {"corpus_warning": _CORPUS_WARNING, **output},
            # Only keys present on every path, refusals included.
            "required": output_required if output_required is not None else sorted(output),
        }
    return tool


_PASSAGE = {
    "type": "object",
    "properties": {
        "citation": {"type": "string"},
        "heading": {"type": ["string", "null"]},
        "text": {"type": "string", "description": "The provision's words, as recorded."},
        "statute": {"type": "string"},
        "in_force_from": {"type": "string"},
        "in_force_to": {
            "type": ["string", "null"],
            "description": "Null while still in force.",
        },
        "status_note": {"type": ["string", "null"]},
        "matched": {"type": ["string", "null"], "description": "Why this one was returned."},
        "score": {
            "type": ["number", "null"],
            "description": "BM25 alone. NOT the sort order: a later passage can score "
            "higher. Sort on (coverage, ranking) or leave the order as given.",
        },
        "coverage": {
            "type": ["number", "null"],
            "description": "Share of the question's content words this provision contains. "
            "The first sort key.",
        },
        "ranking": {
            "type": ["number", "null"],
            "description": "The second sort key: score weighted by how much of the "
            "question the provision's heading answers.",
        },
    },
    "required": ["citation", "text", "statute", "in_force_from"],
}
_VERSION = {
    "type": "object",
    "properties": {
        "from": {"type": "string"},
        "to": {"type": ["string", "null"]},
        "manner": {"type": ["string", "null"], "description": "substituted, repealed, omitted."},
        "amended_by": {"type": ["string", "null"]},
        "enacted_by": {
            "type": ["string", "null"],
            "description": "The instrument that brought this version into force.",
        },
        "heading": {"type": ["string", "null"]},
        "superseded_by": {
            "type": ["string", "null"],
            "description": "The instrument that replaced this version, when one is recorded.",
        },
        "text": {"type": "string", "description": "Only when with_text was true."},
    },
    "required": ["from"],
}
_SIDE = {
    "type": "object",
    "properties": {
        "date": {"type": "string"},
        "in_force": {"type": "boolean"},
        "in_force_from": {"type": "string"},
        "in_force_to": {"type": ["string", "null"]},
        "status": {
            "type": ["string", "null"],
            "description": "How this side stood on its date: in force, not yet in force, "
            "or no longer in force.",
        },
        "text": {"type": "string"},
    },
    "required": ["date", "in_force"],
}

TOOLS: list[dict[str, Any]] = [
    _tool(
        "answer_question",
        "Answer from the statute book",
        "Answer a legal question with the text of provisions in force on a date, each with "
        "its citation — or refuse. A question that cites a provision ('section 20 PECA') is "
        "a lookup of exactly that provision; otherwise it is a search. Refusals: nothing "
        "matched, the match was too weak, the cited provision was not in force on that "
        "date (its history is returned), or it is not in the corpus.",
        {
            "question": {
                "type": "string",
                "description": "The question, in English. A citation in it ('section 20 "
                "PECA', 'dafa 302 PPC', 'دفعہ 302 تعزیرات پاکستان') makes it a lookup.",
            },
            "as_of": _date_property(
                "The date the question is about. Law changes; the answer for 2020 and for "
                "today can differ."
            ),
            "statute": {
                **_STATUTE,
                "description": _STATUTE["description"]
                + " Restricts the search, and attaches the statute to a bare 'section 302'.",
            },
        },
        ["question", "as_of"],
        output={
            "question": {
                "type": "string",
                "description": "The question as asked, shortened past 2,000 characters - "
                "the caller already has the whole of it, and echoing it doubles a long "
                "one on the wire. `question_length` is always the full length.",
            },
            "question_length": {"type": "integer"},
            "as_of": {"type": "string"},
            **_REFUSAL,
            "passages": {"type": "array", "items": _PASSAGE},
            "superseded": {
                "type": "array",
                "description": "Cited provisions not in force on that date, with their "
                "full version history - the reason the answer was refused.",
                "items": {
                    "type": "object",
                    "properties": {
                        "citation": {"type": "string"},
                        "status": {"type": "string"},
                        "history": {"type": "array", "items": _VERSION},
                    },
                    "required": ["citation"],
                },
            },
            **_WARNINGS,
        },
    ),
    _tool(
        "check_citations",
        "Check every citation in a draft",
        "Audit a draft — a brief, a notice, an answer a model wrote — for citations that were "
        "not good law on a date. Each citation gets a status: in_force, amended_since (good "
        "law on the date but amended later, so check the draft quotes the older words), "
        "not_in_force (repealed or substituted), not_yet_in_force, unknown_provision, "
        "no_act_named, act_not_recognised (the text named an Act this corpus does not "
        "know, such as a foreign code — it is NOT read as the Pakistani provision of the "
        "same number), "
        "statute_not_loaded, not_checkable (case law, SROs) or before_record (dated before the "
        "corpus starts recording that statute). The verdict is problems, review, "
        "unverified, or 'all_in_force' only when every citation was checked and passed. "
        f"Pages of {CITATION_LIMIT} citations; pass offset to continue.",
        {
            "text": {"type": "string", "description": "The draft, up to 200,000 characters."},
            "as_of": _date_property("The date the draft speaks about."),
            "default_statute": {
                **_STATUTE,
                "description": "The Act a bare 'section 9' in the draft belongs to. "
                "Never guessed when omitted.",
            },
            "offset": {"type": "integer", "minimum": 0, "description": "Default 0."},
        },
        ["text", "as_of"],
        output={
            "as_of": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "citation": {"type": "string"},
                        "raw": {"type": "string", "description": "As written in the draft."},
                        "kind": {
                            "type": "string",
                            "enum": ["statutory", "subordinate", "reported"],
                        },
                        # The enum, declared. It was a bare string, so the test
                        # that validates every tool's output against its own schema
                        # on every path could not notice that the README listed nine
                        # of these and the code emits ten - `act_not_recognised`,
                        # the one the README calls the most serious.
                        "status": {"type": "string", "enum": sorted(ALL_STATUSES)},
                        "note": {"type": "string"},
                        "heading": {"type": ["string", "null"]},
                        "offset": {"type": "integer", "description": "Character offset."},
                        "checked_as": {
                            "type": "string",
                            "description": "Present when a subsection was asked about and "
                            "the section was checked: whether subsection (9) exists is not "
                            "recorded, so the answer is about the whole section.",
                        },
                        "subdivision_checked": {
                            "type": "boolean",
                            "description": "False alongside checked_as, so a caller cannot "
                            "read the status as a statement about the subsection.",
                        },
                        "history": {
                            "type": "array",
                            "items": _VERSION,
                            "description": "Present when the citation is not in force on "
                            "the date: the versions that do exist, which are the answer to "
                            "'then when was it'.",
                        },
                    },
                    "required": ["citation", "kind", "status"],
                },
            },
            "counts": {
                "type": "object",
                "description": "How many citations got each status.",
                "additionalProperties": {"type": "integer"},
            },
            "problems": {"type": "integer", "description": "Occurrences, not provisions."},
            "distinct_problems": {
                "type": "integer",
                "description": "Provisions to fix. One repealed section cited five times "
                "is one problem, not five.",
            },
            "distinct_citations": {"type": "integer"},
            "review": {"type": "integer"},
            "unverified": {"type": "integer"},
            "verdict": {
                "type": "string",
                "enum": ["problems", "review", "unverified", "all_in_force"],
                "description": "all_in_force only when every citation was checked and passed.",
            },
            **_PAGING,
        },
        output_required=[
            "as_of",
            "citations",
            "counts",
            "problems",
            "review",
            "total",
            "unverified",
            "verdict",
        ],
    ),
    _tool(
        "provision_history",
        "Amendment history of a provision",
        "Every version of one provision, oldest first: when each came into force, when and "
        "how it ceased (repealed, substituted, omitted), and what amended it. Use for 'when "
        "did this change' and 'which version applied then'.",
        {
            "citation": {
                "type": "string",
                "description": "A citation in any common form: 'Section 20 PECA', "
                "'s. 302 of the Pakistan Penal Code', 'Article 25'.",
            },
            "statute": {**_STATUTE, "description": "The Act, if the citation omits it."},
            "with_text": {
                "type": "boolean",
                "description": "Include each version's full text. Default false.",
            },
        },
        ["citation"],
        output={
            "citation": {"type": "string"},
            **_REFUSAL,
            **_WARNINGS,
            "versions": {"type": "integer"},
            "currently_in_force": {"type": "boolean"},
            "history": {"type": "array", "items": _VERSION, "description": "Oldest first."},
        },
        output_required=["citation", "refused"],
    ),
    _tool(
        "compare_versions",
        "What an amendment changed",
        "The text of one provision as it stood on two dates, and a word-level list of what "
        "differs — 'three years' became 'five years'. Use when asked what an amendment did, "
        "or whether conduct on one date is judged by different words than today.",
        {
            "citation": {"type": "string", "description": "e.g. 'section 20 PECA'."},
            "before": _date_property("The earlier date."),
            "after": _date_property("The later date."),
            "statute": {**_STATUTE, "description": "The Act, if the citation omits it."},
        },
        ["citation", "before", "after"],
        output={
            "citation": {"type": "string"},
            **_REFUSAL,
            **_WARNINGS,
            "before": _SIDE,
            "after": _SIDE,
            "changes": {
                "type": "array",
                "description": "Word-level differences. Absent when the provision was "
                "not in force on both dates, so there is nothing to compare.",
                "items": {
                    "type": "object",
                    "properties": {
                        "change": {"type": "string", "enum": list(ALL_CHANGE_KINDS)},
                        "before": {"type": ["string", "null"]},
                        "after": {"type": ["string", "null"]},
                    },
                    "required": ["change"],
                },
            },
            "amended_by": {"type": "array", "items": {"type": "string"}},
            "versions_between": {"type": "integer"},
            "summary": {"type": "string"},
        },
        output_required=["refused"],
    ),
    _tool(
        "list_provisions",
        "Contents of a statute",
        "The provisions of one statute in force on a date — citation, heading, chapter, and "
        f"whether it has been amended — in statute order. Pages of {LIST_LIMIT}; pass "
        "offset to continue.",
        {
            "statute": _STATUTE,
            "as_of": _date_property("The date whose statute book to list."),
            "offset": {"type": "integer", "minimum": 0, "description": "Default 0."},
        },
        ["statute", "as_of"],
        output={
            "statute": {"type": "string"},
            "as_of": {"type": "string"},
            **_REFUSAL,
            "in_force": {"type": "integer"},
            "not_in_force_on_this_date": {
                "type": "integer",
                "description": "Provisions of this statute that exist but were not in "
                "force on that date, and so are not listed.",
            },
            "provisions": {
                "type": "array",
                "description": "In statute order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "citation": {"type": "string"},
                        "heading": {"type": ["string", "null"]},
                        "chapter": {"type": ["string", "null"]},
                        "in_force_from": {"type": "string"},
                        "amended": {"type": "boolean"},
                    },
                    "required": ["citation", "in_force_from"],
                },
            },
            **_PAGING,
        },
        output_required=[
            "as_of",
            "in_force",
            "not_in_force_on_this_date",
            "provisions",
            "statute",
            "total",
        ],
    ),
    _tool(
        "changes_between",
        "What changed in a period",
        "Every commencement, substitution, omission and repeal between two dates, oldest "
        "first, optionally for one statute. A substitution is one event, not a repeal and "
        "an unrelated commencement.",
        {
            "start": _date_property("Start of the period, inclusive."),
            "end": _date_property("End of the period, inclusive."),
            "statute": _STATUTE,
            "offset": {"type": "integer", "minimum": 0, "description": "Default 0."},
        },
        ["start", "end"],
        output={
            "from": {"type": "string"},
            "to": {"type": "string", "description": "Reversed dates are normalised."},
            "statute": {"type": ["string", "null"], "description": "Null means all loaded."},
            "events": {
                "type": "array",
                "description": "Oldest first. A substitution is one event, not a repeal "
                "and an unrelated commencement.",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "citation": {"type": "string"},
                        "event": {"type": "string"},
                        "by": {"type": ["string", "null"]},
                        "heading": {"type": ["string", "null"]},
                    },
                    "required": ["date", "citation", "event"],
                },
            },
            **_PAGING,
            **_WARNINGS,
        },
        output_required=["events", "from", "statute", "to", "total"],
    ),
    _tool(
        "parse_citations",
        "Parse legal citations",
        "Find every Pakistani legal citation in a passage and resolve each to one canonical "
        "key, so 'Section 302 PPC', 's. 302 of the Pakistan Penal Code' and '§302 PPC' "
        "agree. Handles statutory provisions, Order/Rule CPC, SROs and reported judgments "
        "(PLD, SCMR, CLC, YLR ...), rules made under an Act ('rule 5 of the Companies "
        "Rules'), and foreign series such as AIR, which are marked as foreign rather "
        "than dropped. Needs no corpus; to check the citations against the "
        "law, use check_citations.",
        {
            "text": {"type": "string", "description": "Any passage of legal text."},
            "default_statute": {
                **_STATUTE,
                "description": "The Act a bare 'section 302' belongs to, when the "
                "surrounding document makes it clear. Never guessed when omitted.",
            },
            "offset": {"type": "integer", "minimum": 0, "description": "Default 0."},
        },
        ["text"],
        output={
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "citation": {"type": "string"},
                        "key": {"type": "string", "description": "Canonical identity."},
                        "kind": {
                            "type": "string",
                            "enum": ["statutory", "subordinate", "reported"],
                        },
                        "raw": {"type": "string"},
                        "statute": {"type": "string"},
                        "named_statute": {
                            "type": "string",
                            "description": "An Act the text named that is not one this "
                            "module knows. Empty when no Act was named - the two are "
                            "different, and only the second may take a default statute.",
                        },
                        "in_corpus": {"type": "boolean"},
                    },
                    "required": ["citation", "kind"],
                },
            },
            "count": {"type": "integer", "description": "Citations on this page."},
            **_PAGING,
        },
        output_required=["citations", "count", "total"],
    ),
    _tool(
        "corpus_info",
        "What the corpus covers",
        "Which statutes are loaded, how many provisions and versions each has, the date "
        "range covered, and whether this is the demonstration sample. A statute that is "
        "not listed here cannot be answered from.",
        {},
        [],
        output={
            "source": {"type": "string", "description": "'sample' or the corpus path."},
            "sample": {
                "type": "boolean",
                "description": "True means three demonstration provisions, not the "
                "statute book. Nothing from it should be relied on.",
            },
            "provisions": {"type": "integer"},
            "versions": {"type": "integer"},
            "statutes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "statute": {"type": "string"},
                        "provisions": {"type": "integer"},
                        "versions": {"type": "integer"},
                        "currently_in_force": {"type": "integer"},
                    },
                    "required": ["statute", "provisions", "versions"],
                },
            },
            "malformed": {
                "type": "array",
                "description": "Provisions whose text runs on into later provisions, so "
                "what is served under one citation is more than one section. Empty is the "
                "normal case; a non-empty list is a known gap in this corpus, not a bug "
                "in the question.",
                "items": {
                    "type": "object",
                    "properties": {
                        "citation": {"type": "string"},
                        "appears_to_contain": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["citation", "appears_to_contain"],
                },
            },
            "earliest": {"type": ["string", "null"]},
            "latest_change": {"type": ["string", "null"]},
            "as_at": {
                "type": ["string", "null"],
                "description": "The date the corpus was compiled, when it records one. "
                "Anything after it is not in the corpus.",
            },
            "recorded_from": {
                "type": "object",
                "description": "Per statute, the date from which amendments are recorded.",
                "additionalProperties": {"type": "string"},
            },
        },
        output_required=[
            "as_at",
            "earliest",
            "latest_change",
            "provisions",
            "recorded_from",
            "sample",
            "source",
            "statutes",
            "versions",
        ],
    ),
]


class ToolError(ValueError):
    """A bad argument. Reported to the model as a tool result it can correct, not a crash."""


class ProtocolError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---- corpus loading ------------------------------------------------------------------


def load_corpus(path: str | None) -> tuple[Corpus, str]:
    """The corpus and where it came from. Refuses one that fails validation."""
    if path:
        rows, meta = read_corpus(path)
        corpus = build_checked(rows, source=str(path), meta=meta)
        source = str(path)
    else:
        sample = resources.files("paklaw").joinpath("sample_corpus.json")
        corpus = build_checked(json.loads(sample.read_text(encoding="utf-8")), source="sample")
        source = "sample"

    if not len(corpus):
        raise CorpusError(f"{source}: the corpus is empty")
    problems = corpus.validate()
    if problems:
        raise CorpusError(
            f"{source}: {len(problems)} structural problem(s), refusing to serve answers "
            "that would depend on iteration order:\n  " + "\n  ".join(problems)
        )
    return corpus, source


# ---- argument handling ---------------------------------------------------------------


def _echo(question: str) -> str:
    """The question, shortened if long and with lone surrogates replaced.

    The scrub is not cosmetic. A lone surrogate is valid JSON, so `"murder \\ud800"`
    arrived intact, was echoed into the result, and then could not be encoded as UTF-8
    on the way out - `UnicodeEncodeError` from the writer, uncaught, process exit 1.
    One request with half a character in it ended the session for every request after
    it. Replaced rather than refused, because the request is well formed by every rule
    the protocol has and the question is still what the caller asked.
    """
    scrubbed = _LONE_SURROGATE.sub("\ufffd", question)
    if len(scrubbed) <= ECHO_LIMIT:
        return scrubbed
    return f"{scrubbed[:ECHO_LIMIT]}... [{len(scrubbed):,} characters; echo shortened]"


def _string(arguments: dict, name: str, *, required: bool = False) -> str | None:
    value = arguments.get(name)
    if value is not None and not isinstance(value, str):
        raise ToolError(f"'{name}' must be a string, got {type(value).__name__}")
    if value is not None and len(value) > TEXT_LIMIT:
        raise ToolError(
            f"'{name}' is {len(value):,} characters; the limit is {TEXT_LIMIT:,}. Send it in parts."
        )
    supplied = value is not None
    value = (value or "").strip()
    if not value:
        # Whitespace is as empty as "": answering it would be a refusal about nothing.
        # But the message has to say which it was. `'question' is required` sent back
        # for `{"question": ""}` describes a key the caller did include, and a model
        # reading it will add the key it already sent rather than put a question in it.
        if required and supplied:
            raise ToolError(f"'{name}' was given but is empty; it needs the actual text")
        if required:
            raise ToolError(f"'{name}' is required")
        return None
    return value


def _offset(arguments: dict) -> int:
    offset = arguments.get("offset", 0)
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ToolError("offset must be a non-negative integer")
    return offset


def _page(result: dict, field: str, offset: int, limit: int) -> dict:
    items = result[field]
    result[field] = items[offset : offset + limit]
    result["total"] = len(items)
    if offset + limit < len(items):
        result["next_offset"] = offset + limit
    return result


def _refusal(result: dict) -> dict:
    """The library's {"error": ...} as the same refusal shape answer_question uses.

    `refusal_status` comes from the library with the error, because the schema tells
    callers to branch on it and this function used to drop it: every refusal from
    `provision_history` and `compare_versions` declared the key and omitted it, so the
    one instruction the schema gives was impossible to follow.
    """
    if "error" in result:
        reason = result.pop("error")
        status = result.pop("refusal_status", None)
        return {"refused": True, "refusal_reason": reason, "refusal_status": status, **result}
    return {"refused": False, "refusal_status": None, **result}


def _date(arguments: dict, name: str) -> dt.date:
    value = _string(arguments, name, required=True)
    # fromisoformat alone accepts "20260101" on 3.11+ and not on 3.10; one format, everywhere.
    if not _ISO_DATE.match(value):
        raise ToolError(f"{name} must be a date as YYYY-MM-DD, got {value!r}")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ToolError(f"{name} is not a real date: {value!r}") from exc


def _finite(value: float) -> float | None:
    # A cited provision is resolved, not scored; the library marks that with infinity,
    # which JSON cannot carry.
    return None if math.isinf(value) else value


def _swapped(result: dict, first: str, second: str) -> None:
    # The library orders the dates itself; say so, or a model that sent them reversed
    # reads "before" in the result as the date it called "after".
    result.setdefault("warnings", []).append(
        f"{first} was later than {second}; the dates were taken in date order"
    )


def _first_citation(text: str, statute: str | None, aliases: dict[str, str]) -> str:
    """The corpus key of the first statutory citation, or a ToolError saying why not."""
    found = [c for c in parse(text, statutes=aliases) if c.kind == "statutory"]
    if not found:
        raise ToolError(f"no statutory citation in {text!r}; try 'section 20 PECA'")
    c = found[0]
    act = c.statute or statute
    if not act:
        # "section 302" of which Act? Guessing attributes it to the wrong one.
        raise ToolError(f"{c.pretty()!r} names no Act; add it to the citation or pass statute")
    return f"{act}:{c.unit}:{c.provision}"


# ---- the server ----------------------------------------------------------------------


class LawServer:
    def __init__(self, corpus: Corpus, source: str) -> None:
        self.corpus = corpus
        self.source = source
        self.assistant = LawAssistant(corpus=corpus)
        self.loaded = sorted({p.statute for p in corpus})
        self.aliases = statute_aliases(self.loaded)
        self.protocol_version: str | None = None
        #: What the client ASKED for, which is what a second handshake is compared
        #: against. `protocol_version` is what was agreed, and an unknown version agrees
        #: to the newest - so comparing that accepted a re-handshake the README says is
        #: refused.
        self.protocol_requested: str | None = None
        self._schemas = {t["name"]: t["inputSchema"] for t in TOOLS}

    @property
    def is_sample(self) -> bool:
        return self.source == "sample"

    def _statute(self, arguments: dict, name: str, *, required: bool = False) -> str | None:
        """Canonical key, from a known name or any statute the corpus itself holds."""
        value = _string(arguments, name, required=required)
        if value is None:
            return None
        for key in self.loaded:
            if key.lower() == value.lower():
                return key
        known = normalise_statute(value)
        if known:
            return known
        raise ToolError(
            f"unknown statute {value!r}; loaded: {', '.join(self.loaded)}; "
            f"recognised: {', '.join(CANONICAL_STATUTES)}"
        )

    def _with_corpus_note(self, result: dict) -> dict:
        if self.is_sample:
            result["corpus_warning"] = SAMPLE_WARNING
        return result

    # ---- tools -----------------------------------------------------------------------

    def answer_question(self, arguments: dict) -> dict:
        question = _string(arguments, "question", required=True)
        as_of = _date(arguments, "as_of")
        statute = self._statute(arguments, "statute")

        answer: Answer = self.assistant.answer(question, as_of=as_of, statute=statute)
        result = {
            "question": _echo(answer.question),
            "question_length": len(answer.question),
            "as_of": answer.as_of,
            "refused": answer.refused,
            "refusal_reason": answer.refusal_reason or None,
            "refusal_status": answer.refusal_status or None,
            "passages": [
                {
                    "citation": p.citation,
                    "heading": p.heading,
                    "text": p.text,
                    "statute": p.statute,
                    "in_force_from": p.in_force_from,
                    "in_force_to": p.in_force_to,
                    "status_note": p.status_note or None,
                    # "cited" when the question named it: a lookup, not a ranking.
                    "matched": p.matched_terms,
                    "score": _finite(p.score),
                    "coverage": _finite(p.coverage),
                    "ranking": _finite(p.ranking),
                }
                for p in answer.passages
            ],
            "superseded": answer.superseded,
            "warnings": answer.warnings,
        }
        if statute and statute not in self.loaded:
            result["warnings"].append(f"{statute} is not in this corpus")
        # `LawAssistant.answer` attaches the coverage warnings for the statutes it
        # cited, so this adds only the case the library cannot see: a question scoped
        # to a statute that returned nothing, where the statute named is still the one
        # the reader wants coverage told about.
        if statute and not answer.passages:
            for note in self.corpus.coverage_warnings(as_of, [statute]):
                if note not in result["warnings"]:
                    result["warnings"].append(note)
        return self._with_corpus_note(result)

    def check_citations(self, arguments: dict) -> dict:
        text = _string(arguments, "text", required=True)
        as_of = _date(arguments, "as_of")
        default = self._statute(arguments, "default_statute")
        report = check_citations(self.corpus, text, as_of=as_of, default_statute=default)
        # Counts and verdict cover every citation; only the listing is paged.
        return self._with_corpus_note(
            _page(report, "citations", _offset(arguments), CITATION_LIMIT)
        )

    def provision_history(self, arguments: dict) -> dict:
        citation = _string(arguments, "citation", required=True)
        statute = self._statute(arguments, "statute")
        with_text = arguments.get("with_text", False)
        if not isinstance(with_text, bool):
            raise ToolError("with_text must be true or false")
        key = _first_citation(citation, statute, self.aliases)
        result = _refusal(self.assistant.history(citation, statute=statute))
        resolved = self.corpus.resolve(key)
        if resolved != key and not result["refused"]:
            result["warnings"] = [
                f"{citation} is a subdivision; this is the history of the whole provision"
            ]
        if with_text and not result["refused"]:
            for entry, version in zip(
                result["history"], self.corpus.versions(resolved), strict=True
            ):
                entry["text"] = version.text
        return self._with_corpus_note(result)

    def compare_versions(self, arguments: dict) -> dict:
        key = _first_citation(
            _string(arguments, "citation", required=True),
            self._statute(arguments, "statute"),
            self.aliases,
        )
        before, after = _date(arguments, "before"), _date(arguments, "after")
        result = _refusal(compare_versions(self.corpus, key, before=before, after=after))
        if before > after:
            _swapped(result, "before", "after")
        return self._with_corpus_note(result)

    def list_provisions(self, arguments: dict) -> dict:
        statute = self._statute(arguments, "statute", required=True)
        as_of = _date(arguments, "as_of")
        offset = _offset(arguments)
        if statute not in self.loaded:
            # A statute this corpus does not hold is a well-formed argument naming
            # something absent, which is a refusal - the same thing `answer_question`
            # calls `no_act_named` and `check_citations` calls `statute_not_loaded`.
            # Raised as a tool error, it told a model it had called the tool wrongly,
            # and a model told that retries instead of reporting.
            return {
                "refused": True,
                "refusal_status": "statute_not_loaded",
                "refusal_reason": (
                    f"{statute} is not in this corpus; loaded: {', '.join(self.loaded)}"
                ),
                "statute": statute,
                "as_of": as_of.isoformat(),
                "in_force": 0,
                "not_in_force_on_this_date": 0,
                "provisions": [],
                "total": 0,
            }
        result = contents(self.corpus, statute, as_of=as_of)
        return self._with_corpus_note(_page(result, "provisions", offset, LIST_LIMIT))

    def changes_between(self, arguments: dict) -> dict:
        start, end = _date(arguments, "start"), _date(arguments, "end")
        statute = self._statute(arguments, "statute")
        result = changes_between(self.corpus, start, end, statute=statute)
        if start > end:
            _swapped(result, "start", "end")
        return self._with_corpus_note(_page(result, "events", _offset(arguments), EVENT_LIMIT))

    def parse_citations(self, arguments: dict) -> dict:
        text = _string(arguments, "text", required=True)
        default = self._statute(arguments, "default_statute")

        found = []
        for c in parse(text, statutes=self.aliases):
            if default and c.kind == "statutory" and not c.statute:
                c = type(c)(**{**c.__dict__, "statute": default})
            entry = {"kind": c.kind, "citation": c.pretty(), "key": c.key, "raw": c.raw}
            if c.kind == "statutory":
                entry["statute"] = c.statute or None
                entry["in_corpus"] = bool(c.statute) and bool(
                    self.corpus.versions(self.corpus.resolve(c.key))
                )
                if not c.statute:
                    entry["note"] = "no Act named; pass default_statute if the context says"
            found.append(entry)
        result = {"citations": found}
        result = _page(result, "citations", _offset(arguments), CITATION_LIMIT)
        # After the cut, not before it. `count` is declared as "citations on this page"
        # and was set to the whole total, so 600 citations reported count 600 beside 500
        # citations - two numbers for the same thing, one of them wrong. `total` is the
        # whole, and it comes from `_page`.
        result["count"] = len(result["citations"])
        return result

    def corpus_info(self, arguments: dict) -> dict:
        provisions = list(self.corpus)
        by_statute: dict[str, Counter] = {}
        for p in provisions:
            counts = by_statute.setdefault(p.statute, Counter())
            counts["versions"] += 1
            counts["in_force"] += p.currently_in_force
        distinct = Counter(p.statute for p in {p.key: p for p in provisions}.values())

        return self._with_corpus_note(
            {
                "source": self.source,
                "sample": self.is_sample,
                "versions": len(provisions),
                "provisions": sum(distinct.values()),
                "statutes": [
                    {
                        "statute": s,
                        "provisions": distinct[s],
                        "versions": by_statute[s]["versions"],
                        "currently_in_force": by_statute[s]["in_force"],
                    }
                    for s in sorted(by_statute)
                ],
                # What this corpus is known to get wrong, named rather than left for a
                # reader to notice from a provision that is nine provisions long.
                "malformed": [
                    {"citation": key, "appears_to_contain": buried}
                    for key, buried in sorted(self.corpus.swallowed_headings().items())
                ],
                "earliest": min(p.in_force_from for p in provisions).isoformat(),
                "latest_change": max(
                    max(p.in_force_from, p.in_force_to or p.in_force_from) for p in provisions
                ).isoformat(),
                "as_at": self.corpus.as_at.isoformat() if self.corpus.as_at else None,
                "recorded_from": {
                    s: d.isoformat()
                    for s in sorted(by_statute)
                    if (d := self.corpus.recorded_from(s))
                },
                **({"meta": self.corpus.meta} if self.corpus.meta else {}),
            }
        )

    # ---- protocol --------------------------------------------------------------------

    def _initialize(self, params: dict) -> dict:
        requested = params.get("protocolVersion")
        agreed = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        if self.protocol_requested is not None and requested != self.protocol_requested:
            # Initialise happens once per session. A second one used to be accepted and
            # to reset the version - so a client that re-sent `initialize` with a
            # version this server does not know had its session silently moved to the
            # newest one, which changes whether later results carry
            # `structuredContent`. The results change shape and nothing says why.
            #
            # Compared on what was REQUESTED, not on what was agreed. An unknown version
            # agrees to the newest, so a second handshake asking for "9999-01-01" landed
            # on the version already in force, differed in nothing, and was accepted -
            # while the README said a re-initialize at a different version is refused.
            # A client asking for a different version is changing its mind about the
            # session whatever the negotiation would land on.
            #
            # Re-sending the SAME version is allowed: that is a retry of a message the
            # client is not sure arrived, and answering it identically strands nobody.
            raise ProtocolError(
                INVALID_REQUEST,
                f"already initialized at protocol {self.protocol_version}; a session is "
                "initialized once, and this request would have changed the shape of "
                "every later result",
            )
        self.protocol_version = agreed
        self.protocol_requested = requested
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "title": "Pakistan law", "version": __version__},
            "instructions": INSTRUCTIONS,
        }

    def _call_tool(self, params: dict) -> dict:
        name = params.get("name")
        known = {t["name"] for t in TOOLS}
        if name is None:
            raise ProtocolError(
                INVALID_PARAMS,
                "tools/call needs params.name; one of " + ", ".join(sorted(known)),
            )
        if name not in known:
            raise ProtocolError(
                INVALID_PARAMS, f"unknown tool: {name!r}; one of " + ", ".join(sorted(known))
            )
        handler = getattr(self, name)

        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ProtocolError(INVALID_PARAMS, "arguments must be an object")

        def failed(message: str) -> dict:
            return {"content": [{"type": "text", "text": message}], "isError": True}

        # The schemas say additionalProperties: false, so honour it. A misspelt "asof"
        # silently ignored is a required date silently missing.
        allowed = set(self._schemas[name]["properties"])
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            return failed(
                f"unknown argument(s) {', '.join(unknown)}; "
                f"{name} takes {', '.join(sorted(allowed)) or 'no arguments'}"
            )
        try:
            result = handler(arguments)
        except ToolError as exc:
            return failed(str(exc))
        except Exception as exc:  # a bug in one call must not look like a protocol failure
            return failed(f"internal error in {name}: {type(exc).__name__}: {exc}")

        payload = {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
            "isError": False,
        }
        if self.protocol_version in _STRUCTURED:
            payload["structuredContent"] = result
        return payload

    def handle(self, message: Any) -> dict | None:
        """One JSON-RPC message in, at most one response out."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            # The id is answered when the message carried a usable one, even though the
            # envelope is wrong: a client that sent {"jsonrpc": "1.0", "id": 7} can match
            # the error to the call it made. A null id here is the spec's own answer for
            # a message too malformed to attribute.
            sent = message.get("id") if isinstance(message, dict) else None
            usable = sent if isinstance(sent, (str, int)) and not isinstance(sent, bool) else None
            return _error(usable, INVALID_REQUEST, "not a JSON-RPC 2.0 message")

        method = message.get("method")
        if "id" not in message:
            # Notifications (initialized, cancelled) and responses need no reply, and
            # this server sends no requests of its own.
            return None
        request_id = message["id"]
        if (
            request_id is None
            or isinstance(request_id, bool)
            or not isinstance(request_id, (str, int))
        ):
            # MCP forbids a null id; answering one makes a request look like a notification
            # that got a reply.
            return _error(None, INVALID_REQUEST, "id must be a string or an integer")
        if not isinstance(method, str):
            if "result" in message or "error" in message:
                return None
            return _error(request_id, INVALID_REQUEST, "request has no method")

        params = message.get("params") or {}
        try:
            if not isinstance(params, dict):
                raise ProtocolError(INVALID_PARAMS, "params must be an object")
            # The response shape is negotiated, so answering before the negotiation
            # is answering under an assumption the client never made. `protocol_version`
            # is None until `initialize`, and `None not in _STRUCTURED`, so a
            # `tools/call` arriving cold used to come back *without*
            # `structuredContent` and a `tools/list` without `outputSchema` - the
            # degraded shape meant for a 2024 client, handed silently to a client that
            # had said nothing at all. A client that skipped the handshake got a worse
            # answer and no way to tell why.
            #
            # MCP puts the MUST on the client and a SHOULD on the server, and being
            # lenient here looked harmless until the shapes were compared: see
            # `test_mcp_server.test_the_answer_is_the_same_before_and_after_the_handshake`,
            # which is what found this.
            if method not in BEFORE_INITIALIZE and self.protocol_version is None:
                raise ProtocolError(
                    NOT_INITIALIZED,
                    f"{method} before initialize: the response shape depends on the "
                    "protocol version, which is agreed by the initialize handshake. "
                    "Send initialize first.",
                )
            if method == "initialize":
                result = self._initialize(params)
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                # `outputSchema` arrived in 2025-06-18 alongside `structuredContent`.
                # Advertising it to an older client while correctly withholding the
                # payload is advertising a contract this server then declines to honour.
                if self.protocol_version in _STRUCTURED:
                    result = {"tools": TOOLS}
                else:
                    result = {
                        "tools": [
                            {k: v for k, v in tool.items() if k != "outputSchema"} for tool in TOOLS
                        ]
                    }
            elif method == "tools/call":
                result = self._call_tool(params)
            else:
                raise ProtocolError(METHOD_NOT_FOUND, f"method not found: {method}")
        except ProtocolError as exc:
            return _error(request_id, exc.code, exc.message)
        except Exception as exc:  # a bug must not take the server down with it
            # ...but it must leave a trace. stdout is the protocol and stderr is free,
            # so the traceback goes there rather than nowhere: a one-line message with
            # no frames is not enough to find the bug it is reporting.
            traceback.print_exc(file=sys.stderr)
            return _error(request_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

        return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


#: Text that is valid JSON and cannot be encoded as UTF-8. A lone surrogate is half of
#: a surrogate pair, which is what a client truncating a UTF-16 buffer emits, and
#: `json.loads` accepts it.
_LONE_SURROGATE = re.compile(r"[\ud800-\udfff]")


def _message_id(message: Any) -> Any:
    """The id to answer with when a reply cannot be written, or None if there is none."""
    if isinstance(message, dict):
        found = message.get("id")
        if isinstance(found, (str, int, type(None))) and not isinstance(found, bool):
            return found
    return None


def _encode(reply: Any, message_id: Any = None) -> bytes:
    """One reply as a line of bytes, or an error object if it cannot be encoded.

    `_echo` keeps lone surrogates out of what the server reflects; this is the backstop
    for anything else that reaches here, because a reply that cannot be serialised used
    to end the process - the one outcome worse than any error object, since the client
    is left waiting on a request that WAS answered and on every request after it.
    """
    try:
        return json.dumps(reply, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    except (UnicodeEncodeError, ValueError, TypeError) as exc:
        fallback = _error(message_id, INTERNAL_ERROR, f"result could not be encoded: {exc}")
        return json.dumps(fallback, ensure_ascii=True).encode() + b"\n"


def serve(server: LawServer, stdin: BinaryIO, stdout: BinaryIO) -> None:
    """Read newline-delimited JSON-RPC until stdin closes."""
    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        message: Any = None
        try:
            message = json.loads(line)
        # `Exception`, not `(JSONDecodeError, UnicodeDecodeError)`. Those are the two
        # ways a line is *invalid*; they are not the two ways parsing one can fail. A
        # line of 20,000 nested `[` raises `RecursionError` from inside `json.loads`,
        # which propagated out of this loop and exited the process - so the client got
        # no reply to that line and none to anything after it, against a README
        # sentence promising that a malformed line does not end the session.
        except Exception as exc:  # noqa: BLE001 - a hostile line may break a parser
            reply: Any = _error(None, PARSE_ERROR, f"parse error: {type(exc).__name__}: {exc}")
        else:
            if isinstance(message, list):
                # Batches exist in 2025-03-26 and were dropped after; answering one costs
                # nothing and refusing it strands an older client.
                replies = [r for r in map(server.handle, message) if r is not None]
                reply = replies or (
                    None if message else _error(None, INVALID_REQUEST, "empty batch")
                )
            else:
                reply = server.handle(message)

        if reply is not None:
            stdout.write(_encode(reply, _message_id(message)))
            stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="paklaw-mcp",
        description="MCP server (stdio) over a versioned corpus of Pakistani statutes.",
    )
    parser.add_argument(
        "--corpus",
        default=os.environ.get(CORPUS_ENV) or None,
        help=f"JSON array or JSON Lines of provisions (default: ${CORPUS_ENV}, else the "
        "three-provision sample). Build one with paklaw-corpus.",
    )
    parser.add_argument(
        "corpus_file",
        nargs="?",
        metavar="CORPUS",
        help="the corpus file, as an alternative to --corpus",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="load and validate the corpus, print what it covers, and exit",
    )
    parser.add_argument("--version", action="version", version=f"paklaw-mcp {__version__}")
    args = parser.parse_args(argv)
    if args.corpus_file:
        args.corpus = args.corpus_file
    # Windows defaults both text streams to the ANSI code page, and an em dash in a
    # diagnostic reaches the client's log as a stray byte.
    #
    # stdout was left out of this, and stdout is where the only non-ASCII output
    # actually goes: `--check` prints the corpus description with
    # `ensure_ascii=False`, and the sample corpus's warning carries an em dash. On a
    # cp1252 console that printed a replacement character; on an ASCII one it raised
    # `UnicodeEncodeError: 'ascii' codec can't encode character '\u2014'` and exited
    # 1, so the one subcommand whose job is to say whether the corpus loaded failed
    # with a traceback about punctuation.
    #
    # The protocol channel is unaffected either way: `serve` writes UTF-8 bytes to
    # `sys.stdout.buffer`, below the text layer being reconfigured here.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    try:
        corpus, source = load_corpus(args.corpus)
    except (OSError, json.JSONDecodeError, CorpusError) as exc:
        print(f"paklaw-mcp: cannot load corpus: {exc}", file=sys.stderr)
        return 2

    server = LawServer(corpus, source)
    info = server.corpus_info({})
    if args.check:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    print(
        f"paklaw-mcp {__version__}: {info['provisions']} provisions "
        f"({info['versions']} versions) from {source}",
        file=sys.stderr,
    )
    if server.is_sample:
        print(f"paklaw-mcp: {SAMPLE_WARNING}", file=sys.stderr)
    serve(server, sys.stdin.buffer, sys.stdout.buffer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
