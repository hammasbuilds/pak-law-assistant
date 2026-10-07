"""Three well-formed lines that used to end the session.

The README says "a malformed line does not end the session" and `test_mcp_server.py`
has a test of that name. It sends syntactically bad JSON, which is the one case that
was handled. An independent review found two inputs that are *valid* JSON and killed
the process - the client got no reply to that request and none to any request after it,
which is worse than any error object - and a third that quietly changed the shape of
every later result.

Each is driven through the real read loop, with a message after it: the point is not
that the bad line is answered, it is that the session is still there.
"""

from __future__ import annotations

import json

import pytest

from tests.test_mcp_server import init, run

#: The id of the message sent after the hostile one. A reply carrying it is the whole
#: assertion: the loop survived.
AFTER = 99


def _ping(request_id: int = AFTER) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "method": "ping"}


def test_a_lone_surrogate_in_a_question_does_not_kill_the_server():
    """`"\\ud800"` is legal JSON and cannot be encoded as UTF-8.

    It is what a client truncating a UTF-16 buffer emits. `answer_question` echoes the
    caller's question into its result, so it reached `json.dumps(...).encode()`, raised
    `UnicodeEncodeError`, and nothing caught it: the process exited 1 mid-session.
    """
    replies = run(
        [
            init(),
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "answer_question",
                    "arguments": {"question": "murder \ud800", "as_of": "2026-01-01"},
                },
            },
            _ping(),
        ]
    )
    answered = [r for r in replies if r.get("id") == 1]
    assert answered, replies
    assert "result" in answered[0], answered[0]
    # The surrogate is replaced, not passed through: the reply has to be encodable.
    echoed = json.loads(answered[0]["result"]["content"][0]["text"])["question"]
    assert "\ud800" not in echoed
    assert "murder" in echoed
    assert [r for r in replies if r.get("id") == AFTER], "the session ended"


@pytest.mark.parametrize("depth", [200, 20_000])
def test_a_deeply_nested_line_is_a_parse_error_and_not_an_exit(depth: int):
    """`json.loads` raises `RecursionError` on deep nesting, which is not an
    `ImportError`-style subclass of anything the except clause listed: it listed
    `(JSONDecodeError, UnicodeDecodeError)`, the two ways a line is *invalid*, which
    are not the two ways parsing one can fail."""
    replies = run([init(), b"[" * depth, _ping()])
    assert [r for r in replies if r.get("id") == AFTER], "the session ended"
    errors = [r for r in replies if "error" in r]
    if depth > 1_000:
        assert errors, replies
        assert errors[0]["error"]["code"] == -32700, errors[0]


def test_a_second_initialize_at_a_different_version_is_refused():
    """MCP initialises once. A second one used to be accepted and to reset the version.

    With an unknown version it reset to the newest supported, which decides whether a
    result carries `structuredContent` - so a client's results changed shape in the
    middle of a session because of a message the server should have refused.
    """
    replies = run(
        [
            init("2025-06-18"),
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "initialize",
                "params": {"protocolVersion": "not-a-version", "capabilities": {}},
            },
            _ping(),
        ]
    )
    second = [r for r in replies if r.get("id") == 7]
    assert second and "error" in second[0], second
    assert second[0]["error"]["code"] == -32600, second[0]
    assert "already initialized" in second[0]["error"]["message"]
    assert [r for r in replies if r.get("id") == AFTER], "the session ended"


def test_re_sending_the_same_initialize_is_allowed():
    """A client retrying a message it is not sure arrived is not an error, and the
    refusal above must not turn a retry into a dead session."""
    replies = run([init("2025-06-18"), {**init("2025-06-18"), "id": 7}, _ping()])
    second = [r for r in replies if r.get("id") == 7]
    assert second and "result" in second[0], second
    assert second[0]["result"]["protocolVersion"] == "2025-06-18"
    assert [r for r in replies if r.get("id") == AFTER]


def test_the_reply_to_every_hostile_line_is_still_on_one_line():
    """stdout is a protocol stream. An error object spread over two lines is not a
    message, and a client reading line by line would desynchronise on it."""
    import io

    from paklaw.mcp_server import LawServer, load_corpus, serve

    corpus, source = load_corpus(None)
    stdin = io.BytesIO(
        b"\n".join(
            [
                json.dumps(init()).encode(),
                b"[" * 20_000,
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "answer_question",
                            "arguments": {"question": "x \ud800", "as_of": "2026-01-01"},
                        },
                    }
                ).encode("utf-8", "surrogatepass"),
                json.dumps(_ping()).encode(),
            ]
        )
        + b"\n"
    )
    stdout = io.BytesIO()
    serve(LawServer(corpus, source), stdin, stdout)
    lines = stdout.getvalue().splitlines()
    assert len(lines) == 4, lines
    for line in lines:
        json.loads(line)
