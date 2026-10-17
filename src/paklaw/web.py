"""A local web page over the law server: ask a question as of a date, or check a draft.

    paklaw-web --port 8841 --no-browser
    python -m paklaw.web --corpus data/pakistan-code.jsonl.gz

Standard library only, like the package. The page calls the same `LawServer` methods
the MCP server exposes as tools, so an answer here is the answer a client gets:

    POST /api/ask      {question, as_of, statute?}  -> passages with the dates they were in force
    POST /api/check    {text, as_of}                -> each citation in a draft, checked on the date
    POST /api/history  {citation, statute?}         -> each version of one provision
    GET  /api/info                                  -> what corpus is loaded, and the statute list

The corpus is `--corpus`, else `PAKLAW_CORPUS`, else `data/pakistan-code.jsonl.gz` in a
clone, else the three-provision sample, and the page says which. Binds 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import audit
from .corpus import CorpusError
from .mcp_server import CORPUS_ENV, LawServer, ToolError, load_corpus

DEFAULT_PORT = 8841
HERE = Path(__file__).resolve().parent
PAGE = HERE / "webui" / "index.html"
# Present in a clone that has fetched the dataset (src/paklaw/web.py -> repo/data).
CLONE_CORPUS = HERE.parent.parent / "data" / "pakistan-code.jsonl.gz"
MAX_BODY = 2 * 1024 * 1024

#: Questions the page offers as one-click samples, each with the date it is asked about.
SAMPLES = [
    {"question": "What is the punishment for theft?", "as_of": "2024-01-01"},
    # Asked before PECA commenced (2016-08-18): refused, with the date it came into force.
    {"question": "Section 20 of PECA", "as_of": "2015-06-01"},
    {"question": "What is the punishment for murder?", "as_of": "2024-06-01"},
    {"question": "Is section 124A of the Penal Code in force?", "as_of": "2024-06-01"},
    {
        "question": "punishment for defamation through an information system",
        "as_of": "2020-01-01",
    },
]
SAMPLE_DRAFT = (
    "The accused was charged under section 302 PPC and section 124A of the Pakistan Penal "
    "Code, and separately under s.20 of PECA. The prosecution also relied on section 999 PPC."
)


def resolve_corpus(explicit: str | None) -> str | None:
    """The corpus path to load, or None for the bundled sample."""
    if explicit:
        return explicit
    if os.environ.get(CORPUS_ENV):
        return os.environ[CORPUS_ENV]
    if CLONE_CORPUS.exists():
        return str(CLONE_CORPUS)
    return None


#: How the page shows each citation status: (tone, words). Keyed by the audit's own
#: constants, and a test checks every status has an entry.
STATUS_LABELS = {
    audit.IN_FORCE: ("good", "in force"),
    audit.AMENDED_SINCE: ("warn", "in force, amended since"),
    audit.REPEALED: ("bad", "no longer in force"),
    audit.NOT_YET: ("bad", "not yet in force"),
    audit.UNKNOWN: ("bad", "no such provision"),
    audit.NO_ACT: ("warn", "names no Act"),
    audit.FOREIGN_ACT: ("bad", "Act not held here"),
    audit.NOT_LOADED: ("warn", "statute not loaded"),
    audit.NOT_CHECKED: ("plain", "not checkable here"),
    audit.BEFORE_RECORD: ("warn", "before this corpus's record"),
}


class InputError(ValueError):
    """The request is wrong; the message is shown on the page."""


class App:
    """The loaded server, the statute titles, and one lock: answers are not concurrent."""

    def __init__(self, corpus_path: str | None) -> None:
        corpus, source = load_corpus(corpus_path)
        self.server = LawServer(corpus, source)
        self.lock = threading.Lock()
        titles: dict[str, str] = {}
        for p in corpus:
            if p.statute not in titles and p.source:
                titles[p.statute] = p.source.removeprefix("Pakistan Code: ").strip()
        self.titles = titles
        self.provisions = len({p.key for p in corpus})

    def info(self) -> dict:
        return {
            "source": Path(self.server.source).name if not self.server.is_sample else "sample",
            "sample": self.server.is_sample,
            "statutes": len(self.server.loaded),
            "provisions": self.provisions,
            "statute_list": [
                {"key": key, "title": self.titles.get(key, "")} for key in self.server.loaded
            ],
            "samples": SAMPLES,
            "sample_draft": SAMPLE_DRAFT,
            "today": dt.date.today().isoformat(),
        }

    def _call(self, method: str, arguments: dict) -> dict:
        with self.lock:
            try:
                result = getattr(self.server, method)(arguments)
            except ToolError as exc:
                message = str(exc)
                if message.startswith("unknown statute"):
                    # The tool lists every loaded statute, which over the statute book is
                    # 970 keys: right for a model, a wall of text on a page that already
                    # offers the list as you type.
                    named = message.split(";", 1)[0].removeprefix("unknown statute ").strip()
                    message = (
                        f"No statute {named} in this corpus. Start typing in the statute box "
                        "to pick one from the list, or leave it empty to search all of them."
                    )
                raise InputError(message) from None
        for passage in result.get("passages", []):
            passage["act_title"] = self.titles.get(passage.get("statute", ""), "")
        return result

    @staticmethod
    def _args(body: dict, *names: str) -> dict:
        out = {}
        for name in names:
            value = body.get(name)
            if isinstance(value, str):
                value = value.strip()
            if value not in (None, ""):
                out[name] = value
        return out

    def ask(self, body: dict) -> dict:
        args = self._args(body, "question", "as_of", "statute")
        if "question" not in args:
            raise InputError("Type a question first.")
        args.setdefault("as_of", dt.date.today().isoformat())
        return self._call("answer_question", args)

    def check(self, body: dict) -> dict:
        args = self._args(body, "text", "as_of")
        if "text" not in args:
            raise InputError("Paste or upload the text to check first.")
        args.setdefault("as_of", dt.date.today().isoformat())
        result = self._call("check_citations", args)
        for citation in result.get("citations", []):
            tone, words = STATUS_LABELS.get(
                citation.get("status"), ("warn", citation.get("status"))
            )
            citation["tone"], citation["label"] = tone, words
        return result

    def history(self, body: dict) -> dict:
        args = self._args(body, "citation", "statute")
        args["with_text"] = True
        return self._call("provision_history", args)


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    routes = {"/api/ask": app.ask, "/api/check": app.check, "/api/history": app.history}

    class Handler(BaseHTTPRequestHandler):
        server_version = "paklaw-web"

        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, data, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/info":
                self._json(200, app.info())
            else:
                self._json(404, {"error": f"no such page: {path}"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            route = routes.get(path)
            if route is None:
                self._json(404, {"error": f"no such endpoint: {path}"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY:
                self._json(413, {"error": "request too large"})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise InputError("the request body must be a JSON object")
                self._json(200, route(body))
            except json.JSONDecodeError:
                self._json(400, {"error": "the request body is not valid JSON"})
            except (InputError, ValueError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - shown on the page, not swallowed
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def make_server(app: App, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="paklaw-web", description="Local web page: ask Pakistani statute law as of a date."
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--corpus", default=None, help=f"corpus file (default: ${CORPUS_ENV})")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")

    path = resolve_corpus(args.corpus)
    try:
        app = App(path)
    except (CorpusError, OSError, ValueError) as exc:
        print(f"error: cannot load the corpus: {exc}", file=sys.stderr)
        return 2
    try:
        server = make_server(app, args.port)
    except OSError as exc:
        print(f"error: cannot listen on 127.0.0.1:{args.port} ({exc})", file=sys.stderr)
        return 2
    url = f"http://127.0.0.1:{args.port}/"
    info = app.info()
    print(
        f"pak-law page at {url}  ({info['provisions']:,} provisions, {info['statutes']} statutes "
        f"from {info['source']}; Ctrl+C to stop)",
        flush=True,
    )
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
