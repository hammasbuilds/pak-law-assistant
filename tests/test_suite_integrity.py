"""No test module can disappear from a run without the run saying so.

An independent review installed the package the way the README says to - `pip install
-e .` and pytest, without the dev group - and got

    333 passed, 2 skipped

against a suite of 534. `tests/test_mcp_server.py` had `pytest.importorskip(
"jsonschema")` at module level, 950 lines in. A module-level importorskip raises during
*collection*, so the whole module was dropped; `tests/test_user_review.py` imports
`body`, `call`, `init`, `run` and `tool` from it, so that module was dropped as well.
140 protocol tests and 22 review tests gone, announced as two skips. CI installs the
dev group, so CI never saw it.

Two tests use the validator. It is a fixture now, so the skip lands on those two
parametrised tests - 464 passed, 45 skipped with jsonschema absent, and nothing
vanishes.

This file is the guard. It is about the SHAPE of the suite rather than about any
behaviour of the package: a module-level skip is invisible in a pass count, and a count
is the only thing anyone reads.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent

#: Every test module. Built from the directory, so a new one is covered by being added.
MODULES = sorted(p for p in TESTS.glob("test_*.py"))

#: Calls that stop a whole module when they run at import time. `skip` needs
#: `allow_module_level=True` to be legal there at all; `importorskip` does not and is
#: the one that caught us.
MODULE_STOPPERS = {"importorskip", "skip", "exit", "fail"}


def _module_level_calls(tree: ast.Module) -> list[tuple[str, int]]:
    """Calls evaluated at import time, excluding anything inside a def or class.

    A call inside a function body runs when that function runs. Only the module's own
    top level - including the body of a top-level `if` or `try`, which is where this
    kind of thing usually hides - runs during collection.
    """
    found: list[tuple[str, int]] = []
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in MODULE_STOPPERS:
                found.append((name, node.lineno))
        stack.extend(ast.iter_child_nodes(node))
    return found


@pytest.mark.parametrize("module", MODULES, ids=lambda p: p.name)
def test_no_module_skips_itself_at_import_time(module: Path):
    """The defect, as a rule rather than as the one case it happened in.

    A dependency that only some tests need belongs in a fixture: the skip then lands on
    those tests, is counted as a skip, and the rest of the module runs. At module level
    the same call silently removes every test in the file - and, through imports, in
    other files.
    """
    tree = ast.parse(module.read_text(encoding="utf-8"))
    stoppers = _module_level_calls(tree)
    assert not stoppers, (
        f"{module.name} calls {stoppers} at import time, which drops the whole module "
        "during collection. Use a fixture so the skip is counted against the tests "
        "that need it."
    )


def test_the_suite_is_all_of_it():
    """Collection, which is the number a module-level skip changes.

    Asserted as an exact figure because the failure mode is a count that drops while
    the line above it still says "passed". `python -m pytest -q --collect-only` is the
    command.
    """
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=600,
    )
    assert done.returncode == 0, done.stdout[-2000:]
    import re

    stated = re.search(r"(\d+) tests? collected", done.stdout)
    assert stated, done.stdout[-2000:]
    assert int(stated.group(1)) == 534, (
        f"{stated.group(1)} tests collected, not 534. If that is deliberate, the figure "
        "here and in the README move together."
    )


def test_the_modules_that_import_from_another_test_module_are_named():
    """Why one module-level skip removed two files.

    `test_user_review.py` imports helpers from `test_mcp_server.py`. That is reasonable
    - the helpers run a real server over a pipe and there should be one copy - but it
    means a collection-time skip in the imported module takes the importer with it. The
    dependency is allowed and written down; what is not allowed is the skip, which the
    test above forbids.
    """
    importers = {}
    for module in MODULES:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                ("tests.test_", "test_")
            ):
                importers.setdefault(module.name, set()).add(node.module)
    assert importers == {
        "test_generated_questions.py": {"test_retrieval_quality"},
        "test_mcp_survives.py": {"tests.test_mcp_server"},
        "test_user_review.py": {"tests.test_mcp_server"},
    }, importers


def test_every_declared_dev_dependency_is_optional_in_the_suite():
    """A dev dependency the suite cannot run without is a runtime dependency.

    The package advertises zero dependencies. `jsonschema` is in a dependency group, so
    a reader who installs the package alone must still get a complete, honest run - and
    the two tests that need the validator must skip rather than error.
    """
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[dependency-groups]" in pyproject
    assert "jsonschema" in pyproject
    # And the fixture, not a module-level import, is what the suite uses.
    server_tests = (TESTS / "test_mcp_server.py").read_text(encoding="utf-8")
    assert "def jsonschema():" in server_tests, "the validator is not behind a fixture"
    assert 'pytest.importorskip("jsonschema"' in server_tests
    first_use = server_tests.index('pytest.importorskip("jsonschema"')
    assert (
        server_tests[:first_use].rstrip().endswith('"""')
        or "def jsonschema" in (server_tests[:first_use])
    ), "the importorskip is not inside the fixture"
