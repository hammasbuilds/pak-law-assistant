"""The suite runs in a known environment, whatever the operator has set.

`PAKLAW_CORPUS` is the one environment variable this package reads, and it is a
legitimate thing to set — it is how the server is pointed at a corpus. With it set, the
suite did not pass:

    unset                                  534 passed
    empty dir / empty file / missing path  7 failed, 527 passed
    a valid corpus                         2 failed, 532 passed

Failures, not skips: three subprocess tests and five script-runner cases that spawn
`demo_mcp.py` or `paklaw-mcp` with `{**os.environ, ...}` and so handed the child whatever
the parent had. Anyone who actually runs this server could not run its tests.

Cleared once for the session, in `os.environ`, so that child processes inherit the
cleared value too — a function-scoped `monkeypatch` would not reach a module-scoped
fixture, and these spawn subprocesses. The tests that are ABOUT the variable set it
themselves, which is what makes them tests of it rather than tests under it.
"""

from __future__ import annotations

import os

import pytest

#: Environment this package reads. `src/paklaw/mcp_server.py` reads exactly one variable
#: and `tests/test_coverage.py` asserts that, so this list is one name by construction.
OWN_ENVIRONMENT = ("PAKLAW_CORPUS",)


@pytest.fixture(autouse=True, scope="session")
def _own_environment_is_not_the_operators():
    saved = {name: os.environ.pop(name, None) for name in OWN_ENVIRONMENT}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
