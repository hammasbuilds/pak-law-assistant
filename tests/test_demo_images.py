"""The README's two screenshots, against what `demo.py` prints.

Both were stale, and in the way a screenshot always goes stale: nobody can diff a PNG.

  * `output.png` showed `REFUSED  no provision in force on that date matches the
    question` - the message `src/paklaw/answer.py:407` documents as having been
    *replaced* for being "true and says nothing". The shipped code names the words the
    corpus does not hold.
  * `input.png` showed `corpus  4 provisions`, where the code prints `3 provisions, 4
    versions`: a screenshot asserting a corpus size this repository had corrected.

CI already ran `python demo.py`, and checked that it exited 0. An image cannot be
checked, so `scripts/render_demo_image.py` draws the images *from* the output and keeps
the text it drew in `docs/images/demo.txt`. This file compares that text to a fresh run.
The images are then only as stale as the transcript, and the transcript is checked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "docs" / "images"
TRANSCRIPT = IMAGES / "demo.txt"


def _demo_output() -> str:
    done = subprocess.run(
        [sys.executable, "demo.py"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.replace("\r\n", "\n")


def test_the_transcript_the_images_were_drawn_from_exists():
    assert TRANSCRIPT.is_file(), (
        "docs/images/demo.txt is missing; run `python scripts/render_demo_image.py`"
    )
    assert TRANSCRIPT.read_text(encoding="utf-8").strip(), "the transcript is empty"


def test_the_images_are_drawn_from_what_the_demo_prints():
    """The check the pictures cannot carry themselves.

    A difference here means the demo's output moved and the images did not, which is
    the state both of them were in: regenerate with
    `python scripts/render_demo_image.py`.
    """
    stored = TRANSCRIPT.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert stored == _demo_output(), (
        "docs/images/demo.txt is not what `python demo.py` prints now, so the README's "
        "screenshots show something the code no longer does. Regenerate them with "
        "`python scripts/render_demo_image.py`."
    )


@pytest.mark.parametrize("name", ["input.png", "output.png"])
def test_each_image_is_a_png_that_is_not_empty(name: str):
    """Enough to catch a file that failed to write, which is all a test can say about
    a picture without reading pixels."""
    path = IMAGES / name
    assert path.is_file(), path
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{name} is not a PNG"
    assert len(data) > 5_000, f"{name} is {len(data)} bytes, which is too small to be the output"


def test_the_readme_shows_both_images_and_names_the_command():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("input.png", "output.png"):
        assert name in readme, f"the README no longer shows {name}"
    assert "python demo.py" in readme


@pytest.mark.parametrize(
    "phrase",
    [
        "3 provisions, 4 versions",
        "no provision in this corpus contains a word the question turns on",
    ],
)
def test_the_corrected_sentences_are_the_ones_in_the_transcript(phrase: str):
    """Named, because a transcript regenerated from a broken demo would also match.

    These two are the ones the old images contradicted: the corpus size the repository
    corrected, and the refusal message it replaced for saying nothing.
    """
    assert phrase in TRANSCRIPT.read_text(encoding="utf-8"), phrase


def test_the_replaced_refusal_message_is_not_back():
    stored = TRANSCRIPT.read_text(encoding="utf-8")
    assert "no provision in force on that date matches the question" not in stored
