r"""Build a temporal corpus from the Pakistan Code.

    python scripts/build_corpus.py catalogue   # index  -> catalogue.json
    python scripts/build_corpus.py fetch       # 1,034 Acts -> pages/ and pdf/
    python scripts/build_corpus.py build       # pdf/   -> corpus.jsonl

Three stages, each resumable, because the middle one takes the best part of an hour over
a government site that is the only copy of this text. Both fetch stages skip what is
already on disk, so an interrupted run is restarted by running the same command again -
nothing is re-downloaded to prove it is still there.

`--work` is where the raw download lives; it defaults to a directory beside the
repository and belongs outside it. The PDFs are 400 MB and are not the artefact: the
artefact is `corpus.jsonl`, which `PAKLAW_CORPUS` points the server at.

Requires `pdftotext` (poppler) on PATH, and nothing else.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paklaw import act_index, sources, split  # noqa: E402
from paklaw.ingest import ARTICLE_STATUTES  # noqa: E402

BASE = "https://pakistancode.gov.pk/english/"
INDEX = BASE + "LGu0xAD?alp={letter}&page={page}"
ROW = re.compile(r'href="(UY2F[^"]+)"[^>]*>([^<]{3,200})<')
PDF_HREF = re.compile(r'href="(https://pakistancode\.gov\.pk/pdffiles/[^"]+\.pdf)"', re.I)
AGENT = {"User-Agent": "Mozilla/5.0 (corpus build for an MCP legal-retrieval server)"}

#: Six, to be polite to the publisher rather than to be fast. Serially this ran at 1.2
#: Acts a minute - fourteen hours - and the time went on waiting, not on moving bytes.
WORKERS = 6
#: A provision longer than this is the splitter having missed the next heading, not a
#: long section: the longest real provision in the Penal Code is about 12,000 characters.
#: One 200,000-character "section 1" matches every query and answers none of them.
RUNAWAY = 60_000


def _text(body: bytes) -> str:
    """Decode a page, trying the encoding it probably is before giving up on a byte.

    The index is mostly UTF-8 and carries the odd Windows-1252 byte - a curly apostrophe
    in "Public Accountants' Default Act" is 0x92 there and not valid UTF-8 - so decoding
    with errors="replace" put a replacement character in the Act's own name, and from
    there into the provenance line of every provision of it.
    """
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return body.decode("cp1252", "replace")


def slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower()[:110] or "untitled"


def get(url: str, tries: int = 3) -> bytes | None:
    """The bytes, or None after `tries` attempts. One Act must not stop the run."""
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers=AGENT)
            with urllib.request.urlopen(request, timeout=180) as answer:
                return answer.read()
        except (urllib.error.URLError, OSError, TimeoutError):
            if attempt == tries - 1:
                return None
            time.sleep(3 + 5 * attempt)
    return None


# ---- stage 1: the index ----------------------------------------------------------------


def catalogue(work: Path) -> int:
    """Every Act the alphabetical index lists, as title -> relative URL."""
    found: dict[str, str] = {}
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        first_seen = None
        for page in range(1, 40):
            body = get(INDEX.format(letter=letter, page=page))
            if body is None:
                break
            rows = [(url, html.unescape(title).strip()) for url, title in ROW.findall(_text(body))]
            # Asked for a page past the end, the index serves the last page again, so
            # the walk stops on a repeat rather than on a guessed page count.
            if not rows or rows[0][1] == first_seen:
                break
            first_seen = rows[0][1]
            for url, title in rows:
                found.setdefault(title, url)
            time.sleep(0.4)
        print(f"  {letter}  {len(found):>5} cumulative", flush=True)

    out = work / "catalogue.json"
    out.write_text(json.dumps(found, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n  {len(found)} Act(s) -> {out}")
    return 0


# ---- stage 2: the PDFs -------------------------------------------------------------------


def fetch(work: Path) -> int:
    pages, pdfs = work / "pages", work / "pdf"
    for folder in (pages, pdfs):
        folder.mkdir(parents=True, exist_ok=True)
    listed = json.loads((work / "catalogue.json").read_text(encoding="utf-8"))

    lock = threading.Lock()
    failures: list[dict] = []
    done = 0

    def one(item: tuple[str, str]) -> None:
        nonlocal done
        title, href = item
        name = slug(title)
        page_file, pdf_file = pages / f"{name}.html", pdfs / f"{name}.pdf"

        if not (pdf_file.is_file() and pdf_file.stat().st_size > 1000):
            if not page_file.is_file():
                body = get(BASE + href)
                if body is None:
                    with lock:
                        failures.append({"title": title, "stage": "page"})
                    return
                page_file.write_bytes(body)

            links = PDF_HREF.findall(page_file.read_text(encoding="utf-8", errors="replace"))
            if not links:
                with lock:
                    failures.append({"title": title, "stage": "no-pdf-link"})
                return
            blob = get(links[0])
            if blob is None or not blob.startswith(b"%PDF"):
                with lock:
                    failures.append({"title": title, "stage": "pdf"})
                return
            pdf_file.write_bytes(blob)

        with lock:
            done += 1
            if done % 50 == 0:
                print(f"  {done:>5}/{len(listed)}  {len(failures)} failure(s)", flush=True)

    started = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(one, sorted(listed.items())))

    (work / "fetch_failures.json").write_text(
        json.dumps(failures, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"\n  {done} Act(s) on disk, {len(failures)} failure(s), "
        f"{(time.time() - started) / 60:.1f} min",
        flush=True,
    )
    return 0


# ---- stage 3: the corpus ------------------------------------------------------------------


def pdf_text(pdf: Path) -> str:
    handle, name = tempfile.mkstemp(suffix=".txt")
    os.close(handle)
    try:
        # `-enc UTF-8`, because pdftotext writes Latin-1 by default and the Code's text
        # is full of soft hyphens: the byte 0xAD is U+00AD there and not valid UTF-8 at
        # all, so decoding with errors="replace" turned "Pro<shy>Chancellor" into
        # "Pro�Chancellor". 1,023 provisions - 3.4% of the corpus - carried a
        # replacement character where a real one belonged, and `corpus.normalise_source`
        # already strips U+00AD, so the character it was meant to handle never reached
        # it. Decoded strictly now: a byte this cannot read is a fault to see, not one
        # to paper over.
        done = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), name],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if done.returncode != 0:
            return ""
        return Path(name).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""
    except (subprocess.TimeoutExpired, OSError):
        return ""
    finally:
        Path(name).unlink(missing_ok=True)


def one_act(job: tuple[str, str, str]) -> dict:
    """One Act's rows, or the reason there are none. Never raises: a single unusual
    layout must not end a run over a thousand Acts."""
    title, key, work = job
    name = act_index.strip_annotations(title)
    out: dict = {"title": title, "name": name, "statute": key, "rows": [], "why": ""}

    pdf = Path(work) / "pdf" / f"{slug(title)}.pdf"
    if not pdf.is_file():
        out["why"] = "not downloaded"
        return out
    raw = pdf_text(pdf)
    if len(raw) < 400:
        out["why"] = "pdftotext produced nothing; a scanned image, most likely"
        return out
    try:
        clean = sources.pakistan_code(raw)
    except Exception as err:  # noqa: BLE001
        out["why"] = f"source clean-up failed: {type(err).__name__}"
        return out

    date = act_index.assent_date(raw)
    year = act_index.title_year(title)
    if date is None and year is None:
        out["why"] = "no date in the header and no year in the title"
        return out
    start_known = date is not None
    if date is None:
        date = dt.date(year, 1, 1)

    # The package already knows which instruments are numbered in articles rather than
    # sections, and refuses a corpus that gets it wrong - stored as sections, "Article 6
    # QSO" finds nothing. Reading its table beats testing the title for "constitution",
    # which is how the Qanun-e-Shahadat Order came to be imported as sections.
    unit = ARTICLE_STATUTES.get(key, "section")
    try:
        rows, report = split.split_act(
            clean.text, statute=key, in_force_from=date.isoformat(), unit=unit
        )
    except Exception as err:  # noqa: BLE001
        out["why"] = f"split failed: {type(err).__name__}: {err}"[:200]
        return out
    if not rows:
        out["why"] = "split into no provisions"
        return out

    # A provision past RUNAWAY is a span the splitter never found the end of - the Code
    # of Civil Procedure's sections run to 158 and are then followed by a Schedule of
    # Orders and Rules whose numbering starts again, so one "section" swallowed 337,654
    # characters of it. Dropping that span keeps the Act's other 150 sections, which
    # dropping the Act loses entirely; dropping the Act is still right when most of it
    # ran away, because then the split genuinely did not take.
    runaway = [row for row in rows if len(row["text"]) > RUNAWAY]
    if len(runaway) * 2 >= len(rows):
        out["why"] = (
            f"{len(runaway)} of {len(rows)} provisions over {RUNAWAY:,} characters; "
            "the split did not take"
        )
        return out
    if runaway:
        rows = [row for row in rows if len(row["text"]) <= RUNAWAY]
        out["dropped"] = [
            {"number": row["number"], "characters": len(row["text"])} for row in runaway
        ]

    number = act_index.act_number(raw)
    found = act_index.repeal(title)
    end, end_exact = act_index.ended(date, found) if found else (None, True)
    out["repealed"] = bool(found)

    for row in rows:
        row["source"] = f"Pakistan Code: {name}" + (f" ({number})" if number else "")
        if found:
            row["in_force_to"] = end.isoformat()
            row["manner"] = "repealed"
            row["end_known"] = end_exact
            if found["by"]:
                row["amended_by"] = found["by"]
        if not start_known:
            row["start_known"] = False
            row["note"] = (
                "the Code's text of this Act carries no date in its header, so this is "
                "where this corpus's record begins, not when the provision came into force"
            )
    out["rows"] = rows
    out["report"] = {k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}
    out["start_known"] = start_known
    return out


def build(work: Path) -> int:
    listed = json.loads((work / "catalogue.json").read_text(encoding="utf-8"))
    # Keyed on the Act's own name: "Agricultural Census Act, 1958 (Repealed by Act XiV of
    # 2011 ...)" initialises to ACRXSWEF, an abbreviation of the index's annotation.
    names = {title: act_index.strip_annotations(title) for title in listed}
    by_name = act_index.assign_keys(set(names.values()))
    keys = {title: by_name[name] for title, name in names.items()}
    if len(set(by_name.values())) != len(set(names.values())):
        raise SystemExit("two Acts share a statute key; every lookup of either would return both")

    # The index lists some Acts twice - once plain and once annotated as repealed - and
    # both entries strip to one name, so both would be built under one key. The corpus
    # then holds two conflicting versions of every provision and refuses to load at all.
    # The annotated entry wins: an Act served as repealed when it is live is a refusal
    # somebody checks, and the other way round is a wrong answer nobody does.
    chosen: dict[str, str] = {}
    duplicates: list[dict] = []
    for title in sorted(listed):
        name = names[title]
        if name not in chosen:
            chosen[name] = title
            continue
        kept, other = chosen[name], title
        if act_index.repeal(other) and not act_index.repeal(kept):
            chosen[name], kept, other = other, other, kept
        duplicates.append({"title": other, "why": f"the index lists it again as {kept!r}"})

    jobs = [(title, keys[title], str(work)) for title in sorted(chosen.values())]
    started = time.time()
    built: list[dict] = []
    skipped: list[dict] = list(duplicates)
    provisions = 0

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for index, out in enumerate(pool.map(one_act, jobs, chunksize=4), 1):
            if out["rows"]:
                built.append(out)
                provisions += len(out["rows"])
            else:
                skipped.append({"title": out["title"], "why": out["why"]})
            if index % 100 == 0:
                print(
                    f"  {index:>5}/{len(jobs)}  {provisions:>7,} provisions  "
                    f"{len(skipped)} skipped",
                    flush=True,
                )

    today = dt.date.today().isoformat()
    meta = {
        "_corpus": {
            "as_at": today,
            "sources": [
                {
                    "name": "Pakistan Code",
                    "publisher": "Ministry of Law and Justice, Government of Pakistan",
                    "url": "https://pakistancode.gov.pk/",
                    "retrieved": today,
                }
            ],
            "acts": len(built),
            "note": (
                "Consolidated federal statute text as the Ministry publishes it. A provision "
                "with start_known=false carries the year from its Act's title because the "
                "Code's text of it gives no date; it was in force on that date and may have "
                "been earlier. One with end_known=false belongs to an Act the index records "
                "as repealed without naming the day."
            ),
        }
    }

    out_file = work / "corpus.jsonl"
    with out_file.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for act in built:
            for row in act["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (work / "build_report.json").write_text(
        json.dumps(
            {
                "acts_built": len(built),
                "acts_skipped": len(skipped),
                "provisions": provisions,
                "dates_known": sum(1 for act in built if act["start_known"]),
                "acts_repealed": sum(1 for act in built if act.get("repealed")),
                "skipped": skipped,
                "per_act": [
                    {
                        "name": act["name"],
                        "statute": act["statute"],
                        "provisions": len(act["rows"]),
                        "dropped": act.get("dropped", []),
                        **act.get("report", {}),
                    }
                    for act in built
                ],
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\n  {provisions:,} provisions from {len(built)} Act(s) in "
        f"{(time.time() - started) / 60:.1f} min"
    )
    print(f"  {len(skipped)} Act(s) left out; see {work / 'build_report.json'}")
    print(f"  -> {out_file}  ({out_file.stat().st_size / 1e6:.1f} MB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("catalogue", "fetch", "build"))
    parser.add_argument(
        "--work",
        default=str(Path(__file__).resolve().parent.parent.parent / "pak-law-corpus-build"),
        help="where the raw download lives; outside the repository by default",
    )
    args = parser.parse_args(argv)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    return {"catalogue": catalogue, "fetch": fetch, "build": build}[args.stage](work)


if __name__ == "__main__":
    raise SystemExit(main())
