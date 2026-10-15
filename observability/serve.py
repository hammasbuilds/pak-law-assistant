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
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import collect  # noqa: E402

PAGE = HERE / "console.html"


# ---- the tool's own page ------------------------------------------------------------
#
# The console says what happened; the tool's page is where something goes in and an answer
# comes out. Opening one from the other saves going back to a desktop shortcut.
#
# The directory is a guess that can be overridden, never an assumption. These front ends
# live outside the package deliberately - this installs with zero dependencies and no web
# surface - so the default is a sibling directory, `PAKLAW_TOOL_DIR` overrides it, and when neither
# exists the console says so and gives the command rather than pretending.

TOOL_ENV = "PAKLAW_TOOL_DIR"
TOOL_DEFAULT = "../_demos/paklaw"
TOOL_PORT = 8765
TOOL_NAME = "Pakistan Law"


def tool_directory() -> Path | None:
    """Where the tool's page is, or None if nothing is there."""
    raw = (os.environ.get(TOOL_ENV) or "").strip()
    candidates = (
        [Path(raw)]
        if raw
        else [
            Path(__file__).resolve().parent.parent / TOOL_DEFAULT,
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "serve.py").is_file():
            return resolved
    return None


def tool_answering(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1.5) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def tool_state() -> dict:
    """Everything the button needs to decide what to do."""
    folder = tool_directory()
    running = tool_answering(TOOL_PORT)
    return {
        "name": TOOL_NAME,
        "url": f"http://127.0.0.1:{TOOL_PORT}/",
        "running": running,
        "found": folder is not None,
        "directory": str(folder) if folder else None,
        "variable": TOOL_ENV,
        "command": (
            f"python {folder / 'serve.py'}"
            if folder
            else f"set {TOOL_ENV} to the directory holding the tool's serve.py"
        ),
    }


def start_tool() -> dict:
    """Start it if it is not already up, and wait until it answers.

    Returns the same shape as `tool_state`, so the page has one thing to read whether it
    started something or found it already running.
    """
    if tool_answering(TOOL_PORT):
        return tool_state()
    folder = tool_directory()
    if folder is None:
        return tool_state()
    # Its own console window, so closing this one does not take the tool with it, and
    # its output is its own rather than interleaved with the console's.
    creation = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    subprocess.Popen(
        [sys.executable, "serve.py"],
        cwd=str(folder),
        env={**os.environ, "DEMO_NO_BROWSER": "1"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation,
    )
    # Waited for rather than assumed: a URL opened before the server answers shows a
    # connection error and the reader reloads by hand.
    for _ in range(40):
        time.sleep(0.5)
        if tool_answering(TOOL_PORT):
            break
    return tool_state()


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
        elif path == "/api/tool":
            body = json.dumps(tool_state()).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        else:
            self._send(404, b'{"error":"no route"}', "application/json; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        """Start the tool's page. The only thing this server does besides read."""
        if self.path.split("?", 1)[0] != "/api/tool/start":
            self._send(404, b'{"error":"no route"}', "application/json; charset=utf-8")
            return
        body = json.dumps(start_tool()).encode("utf-8")
        self._send(200, body, "application/json; charset=utf-8")


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
