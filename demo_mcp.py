"""The same three questions, asked through the MCP server instead of the library.

    python demo_mcp.py

Starts `paklaw.mcp_server` as a subprocess and speaks JSON-RPC to it over stdio, the
way Claude Desktop, Claude Code or any MCP client does. Standard library only.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
QUESTION = "What is the penalty for publicly transmitting false information about a person?"
HIDDEN = ("question", "text")  # long; the labels show the dates
DRAFT = (
    "The post is punishable under section 20(1) PECA with up to five years, and it offends "
    "Article 25 of the Constitution; see PLD 2015 SC 401 and section 21 PECA."
)

CALLS = [
    ("answer_question", {"question": QUESTION, "as_of": "2026-01-01"}, "asked about today"),
    ("answer_question", {"question": QUESTION, "as_of": "2020-06-01"}, "the same, about 2020"),
    ("answer_question", {"question": "section 20 PECA", "as_of": "2015-01-01"}, "before enactment"),
    ("provision_history", {"citation": "s. 20 of the Prevention of Electronic Crimes Act"}, ""),
    (
        "compare_versions",
        {"citation": "section 20 PECA", "before": "2020-01-01", "after": "2026-01-01"},
        "what the amendment changed",
    ),
    ("check_citations", {"text": DRAFT, "as_of": "2019-03-01"}, "a draft about 2019 conduct"),
    ("answer_question", {"question": QUESTION, "as_of": "last year"}, "a date the model got wrong"),
]


def _printable() -> None:
    """Make stdout and stderr able to carry the text these demos print.

    Windows defaults the console to the ANSI code page - cp437 on a US install, cp850
    in parts of Europe - and the server's refusal prose contains an em dash. So this
    demo exited 1 with `UnicodeEncodeError: 'charmap' codec can't encode character
    '\u2014'` on the default console of the platform it was written on, while
    `--check` and the protocol channel had both been fixed for exactly this.

    `errors="replace"` rather than a different encoding: the point of a demo is to
    run, and a replacement glyph in one dash is a better outcome than no output.
    """
    import contextlib

    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _printable()
    server = subprocess.Popen(
        [sys.executable, "-m", "paklaw.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )

    def send(message: dict) -> dict | None:
        server.stdin.write(json.dumps(message).encode() + b"\n")
        server.stdin.flush()
        if "id" not in message:
            return None
        return json.loads(server.stdout.readline())

    hello = send(
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {}},
        }
    )["result"]
    send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    tools = send({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]

    print("INPUT")
    print(f"   server     {hello['serverInfo']['name']} {hello['serverInfo']['version']}")
    print(f"   protocol   {hello['protocolVersion']}")
    print(f"   tools      {', '.join(t['name'] for t in tools)}")
    print()
    print("OUTPUT")

    for n, (name, arguments, note) in enumerate(CALLS, 2):
        reply = send(
            {
                "jsonrpc": "2.0",
                "id": n,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )["result"]
        shown = ", ".join(f"{k}={v!r}" for k, v in arguments.items() if k not in HIDDEN)
        label = f"{name}({shown})"
        print(f"   [{n - 1}] {label}" + (f"   -- {note}" if note else ""))

        if reply["isError"]:
            print(f"       TOOL ERROR  {reply['content'][0]['text']}")
        else:
            result = json.loads(reply["content"][0]["text"])
            if result.get("refused"):
                print(f"       REFUSED     {result['refusal_reason']}")
            for p in result.get("passages", []):
                print(
                    f"       {p['citation']}   in force {p['in_force_from']} -> "
                    f"{p['in_force_to'] or 'present'}"
                )
                print(f"       ...{p['text'][p['text'].index('shall be punished') :]}")
            for version in result.get("history", []):
                print(
                    f"       {version['from']} -> {version['to'] or 'present':<10}  "
                    f"{version['manner']}  {version['amended_by']}".rstrip()
                )
            for change in result.get("changes", []):
                before, after = change["before"], change["after"]
                shown = f"{before!r} -> {after!r}" if before and after else repr(before or after)
                print(f"       {change['change']:<9} {shown}")
            for c in result.get("citations", []):
                print(f"       {c['status']:<17} {c['citation']:<17} {c.get('note', '')}".rstrip())
            if "verdict" in result:
                print(f"       verdict: {result['verdict']}")
        print()

    server.stdin.close()
    server.wait(timeout=10)


if __name__ == "__main__":
    main()
