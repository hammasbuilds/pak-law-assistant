"""The console: one page, one JSON endpoint, `http.server`, no dependencies.

    PAKLAW_TELEMETRY=calls.jsonl paklaw-mcp          # the server, now recording
    python observability/serve.py calls.jsonl        # this, reading what it wrote

It reads the file and serves what `collect.py` computed. It does not write to the log,
does not start the MCP server, and holds no arithmetic of its own.

Bound to 127.0.0.1 and not configurable. A call log is a record of what somebody asked a
legal corpus, and the host for that is this machine.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import collect  # noqa: E402

PAGE = HERE / "console.html"


class Console(BaseHTTPRequestHandler):
    log_file: Path = Path("calls.jsonl")

    def log_message(self, fmt: str, *args) -> None:
        """Quiet, except for what went wrong. A console that prints a line per poll
        buries the one line that matters."""
        line = fmt % args
        if " 4" in line or " 5" in line:
            sys.stderr.write(f"{line}\n")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Nothing here is cacheable: the whole point is the number as it is now.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            try:
                body = PAGE.read_bytes()
            except OSError:
                self._send(500, b"console.html is missing", "text/plain; charset=utf-8")
                return
            self._send(200, body, "text/html; charset=utf-8")
        elif path == "/api/summary":
            events = collect.read(self.log_file)
            payload = collect.summarise(events)
            # Said plainly rather than left as an empty dashboard: an empty file and a
            # file that is not there are different problems with different fixes.
            payload["log"] = str(self.log_file)
            payload["log_exists"] = self.log_file.exists()
            body = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        else:
            self._send(404, b'{"error":"no route"}', "application/json; charset=utf-8")


def free_port(preferred: int) -> int:
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log", nargs="?", default="calls.jsonl", help="the file PAKLAW_TELEMETRY named"
    )
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)

    Console.log_file = Path(args.log).resolve()
    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}/"

    if not Console.log_file.exists():
        # A warning, not a refusal: starting the console before the first call is the
        # ordinary way round, and the page says the same thing.
        print(f"  {Console.log_file} does not exist yet.")
        print(f"  Run the server with PAKLAW_TELEMETRY={Console.log_file} to create it.")
    print(f"\n  reading : {Console.log_file}\n  open    : {url}\n  stop    : Ctrl-C\n")

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Console)
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
