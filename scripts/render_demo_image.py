"""Render `demo.py`'s output to the PNGs the README shows.

The README's Input and Output sections were images dated 2026-09-16, and neither was
what the shipped code printed any more:

  * `output.png` showed `REFUSED  no provision in force on that date matches the
    question`, which is the exact message `src/paklaw/answer.py` documents as having
    been *replaced* for being "true and says nothing";
  * `input.png` showed `corpus  4 provisions`, where the code prints `3 provisions,
    4 versions` - the screenshot asserted a corpus size the repository had corrected.

CI ran `python demo.py` and checked only that it exited 0, so a stale image was
invisible. An image is not checkable, so this script makes one *from* the output:

    python scripts/render_demo_image.py

It writes `docs/images/demo.txt` - the command's output, byte for byte - and renders
`input.png` and `output.png` from it. `tests/test_demo_images.py` then compares
`demo.txt` to a fresh run, which is a check on the picture by proxy: the text the image
was drawn from is the text the command prints, or the test fails.

Pillow is not a dependency of this package and is not in any dependency group. It is
needed only to regenerate the images, which happens when the demo's output changes, so
this script asks for it and says so rather than adding a dependency to a package that
advertises none.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "docs" / "images"
TRANSCRIPT = IMAGES / "demo.txt"

#: Colours of a plain dark terminal, which is what these replace.
BACKGROUND = (19, 20, 24)
FOREGROUND = (221, 223, 228)
MUTED = (131, 137, 150)
HEADING = (126, 188, 255)

#: The sections of the output. `demo.py` prints INPUT then OUTPUT, and the README shows
#: them as two images, so the split is by those two headings rather than by line count.
SECTIONS = ("INPUT", "OUTPUT")


def demo_output() -> str:
    """What `python demo.py` prints, from running it."""
    done = subprocess.run(
        [sys.executable, "demo.py"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if done.returncode != 0:
        raise SystemExit(f"demo.py exited {done.returncode}:\n{done.stdout}\n{done.stderr}")
    return done.stdout.replace("\r\n", "\n")


def split_sections(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.split("\n"):
        if line.strip() in SECTIONS:
            current = line.strip()
            out[current] = []
            continue
        if current:
            out[current].append(line.rstrip())
    return {name: lines for name, lines in out.items() if any(lines)}


def _wrap(lines: list[str], width: int) -> list[str]:
    """Hard-wrap at `width`, keeping each continuation under its line's indent.

    A terminal wraps; an image has to be told to. The long refusal message is 200
    characters, and drawn unwrapped it would run off the picture - which is how a
    screenshot ends up showing less than the command said.
    """
    out: list[str] = []
    for line in lines:
        if len(line) <= width:
            out.append(line)
            continue
        indent = " " * (len(line) - len(line.lstrip()) + 4)
        rest = line
        first = True
        while rest:
            room = width - (0 if first else len(indent))
            cut = rest[:room]
            if len(rest) > room and " " in cut:
                cut = cut[: cut.rfind(" ")]
            out.append(cut if first else indent + cut.lstrip())
            rest = rest[len(cut) :].lstrip()
            first = False
    return out


def render(lines: list[str], path: Path, title: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    size = 15
    font = None
    for candidate in (
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, size)
            break
    if font is None:  # pragma: no cover - no monospace font on this machine
        font = ImageFont.load_default()

    width = 104
    body = _wrap(lines, width)
    advance = font.getlength("M") or size * 0.6
    line_height = size + 7
    margin = 22
    image = Image.new(
        "RGB",
        (int(advance * width) + 2 * margin, line_height * (len(body) + 3) + 2 * margin),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(image)

    y = margin
    draw.text((margin, y), "$ python demo.py", font=font, fill=MUTED)
    y += line_height * 2
    draw.text((margin, y), title, font=font, fill=HEADING)
    y += line_height
    for line in body:
        colour = MUTED if line.strip().startswith(("in force", "...")) else FOREGROUND
        draw.text((margin, y), line, font=font, fill=colour)
        y += line_height
    image.save(path)


def main() -> int:
    text = demo_output()
    IMAGES.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT.write_text(text, encoding="utf-8")
    sections = split_sections(text)
    missing = [name for name in SECTIONS if name not in sections]
    if missing:
        raise SystemExit(f"demo.py no longer prints {missing}; its output shape changed")

    try:
        import PIL  # noqa: F401
    except ImportError:
        print(
            "wrote docs/images/demo.txt. Pillow is not installed, so the PNGs were not\n"
            "redrawn: `pip install pillow` and run this again.",
            file=sys.stderr,
        )
        return 1

    render(sections["INPUT"], IMAGES / "input.png", "INPUT")
    render(sections["OUTPUT"], IMAGES / "output.png", "OUTPUT")
    print(f"wrote {TRANSCRIPT.relative_to(ROOT)}, input.png and output.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
