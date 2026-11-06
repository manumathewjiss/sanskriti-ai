"""
Bhagavad-gita As It Is PDF parser (structured verse records).

This script turns the PDF into JSON verse blocks. It does not create
embeddings, FAISS, or answers.

How the pipeline works (plain English)
--------------------------------------
1. Read every PDF page as text, remembering the page number.
2. Remove the repeating copyright line so it does not pollute verse text.
3. Cut the book into 18 chapter slices using "- CHAPTER N -" banners.
   Chapter 18 has no next banner, so we stop at the publisher back matter.
4. Split each chapter on headings like "TEXT  1" or "TEXTS  16–18".
5. Inside each block, split on SYNONYMS / TRANSLATION / PURPORT.
6. The leftover text before SYNONYMS is Devanagari + transliteration.
7. Save one JSON object per TEXT/TEXTS block (not per PDF page).

Grouped verses (TEXTS 16–18) stay ONE record with verse_start/verse_end.
A missing PURPORT becomes has_purport=false and purport="".
Page breaks are ignored as split points; they only become page_start/page_end.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover - environment hint
    raise SystemExit(
        "This script needs the pypdf package. Install it with:\n"
        "  python3 -m pip install pypdf\n"
        "or use the small environment in gita/requirements.txt."
    ) from exc


PDF_DEFAULT = Path(__file__).resolve().parents[1] / "data" / "Bhagavad-gita_As_It_Is.pdf"
OUTPUT_DEFAULT = Path(__file__).resolve().parent / "gita_structured.json"
VALIDATION_DEFAULT = Path(__file__).resolve().parent / "gita_validation.json"

# This edition's chapter lengths as seen in Step 1 inspection (BBT 1972-style).
# Used only as a comparison, never to rewrite extracted data.
BBT_VERSE_COUNTS = {
    1: 46, 2: 72, 3: 43, 4: 42, 5: 29, 6: 47, 7: 30, 8: 28, 9: 34,
    10: 42, 11: 55, 12: 20, 13: 35, 14: 27, 15: 20, 16: 24, 17: 28, 18: 78,
}

COPYRIGHT_RE = re.compile(
    r"Copyright © 1998 The Bhaktivedanta Book Trust Int'l\. All Rights Reserved\."
)
PAGE_MARK_RE = re.compile(r"<<<PAGE (\d+)>>>")
CHAPTER_BANNER_RE = re.compile(r"- CHAPTER\s+(\d+)\s*-")
# TEXT  1   or   TEXTS  16–18  (en-dash or hyphen, extra spaces OK)
TEXT_HEADING_RE = re.compile(
    r"^(TEXTS?)\s+(\d+)(?:\s*[–—-]\s*(\d+))?\s*$",
    re.MULTILINE,
)
SECTION_HEADING_RE = re.compile(
    r"^(SYNONYMS|TRANSLATION|PURPORT)\s*$",
    re.MULTILINE,
)
THUS_END_RE = re.compile(
    r"Thus end the Bhaktivedanta Purports to the \w+ Chapter.*$",
    re.DOTALL | re.IGNORECASE,
)
# Publisher catalog after Chapter 18 (no "- CHAPTER 19 -" exists).
BACK_MATTER_RE = re.compile(
    r"Bhagavad-gita As It Is, translated with purports by"
)

# Characters that usually appear in the *garbled* Devanagari extraction,
# not in the Roman transliteration lines.
DEVANAGARI_EXTRACT_PUNCT = set(r"*\/+=<>()[]{}%&|\"")


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return (page_number, text) for every page. Page numbers start at 1."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = COPYRIGHT_RE.sub("", text)
        pages.append((i + 1, text))
    return pages


def join_pages_with_markers(pages: list[tuple[int, str]]) -> str:
    """
    One long string, with <<<PAGE 43>>> markers.

    Why: a verse often starts on one page and continues on the next.
    We parse the whole chapter as a stream, then use the markers only
    to remember page_start and page_end.
    """
    chunks = []
    for page_number, text in pages:
        chunks.append(f"<<<PAGE {page_number}>>>\n{text}")
    return "\n".join(chunks)


def find_chapter_banners(full_text: str) -> list[re.Match[str]]:
    banners = list(CHAPTER_BANNER_RE.finditer(full_text))
    numbers = [int(m.group(1)) for m in banners]
    if numbers != list(range(1, 19)):
        raise ValueError(
            f"Expected chapter banners 1–18 in order, found {numbers}."
        )
    return banners


def slice_chapter(full_text: str, chapter: int, banners: list[re.Match[str]] | None = None) -> str:
    """Keep one chapter: after its banner, before the next banner or back matter."""
    if banners is None:
        banners = find_chapter_banners(full_text)
    start = banners[chapter - 1]
    if chapter < 18:
        end_pos = banners[chapter].start()
    else:
        back = BACK_MATTER_RE.search(full_text, start.end())
        end_pos = back.start() if back else len(full_text)

    # Do not rewind into the previous chapter's text. Page 90 contains both
    # the end of Chapter 1 ("Thus end...") and the Chapter 2 banner. Including
    # that closer would let THUS_END_RE delete the whole next chapter.
    chapter_text = full_text[start.start() : end_pos]
    marker = None
    for mark in PAGE_MARK_RE.finditer(full_text):
        if mark.start() <= start.start():
            marker = mark
        else:
            break
    if marker and not chapter_text.lstrip().startswith("<<<PAGE"):
        chapter_text = f"<<<PAGE {marker.group(1)}>>>\n" + chapter_text
    chapter_text = THUS_END_RE.sub("", chapter_text)
    return chapter_text


def parse_verse_range(kind: str, start: str, end: str | None) -> tuple[str, int, int]:
    """TEXT 1 -> (TEXT 1, 1, 1).  TEXTS 16–18 -> (TEXTS 16–18, 16, 18)."""
    verse_start = int(start)
    verse_end = int(end) if end else verse_start
    if verse_end < verse_start:
        raise ValueError(f"Backwards verse range: {start}-{end}")
    if kind == "TEXTS":
        label = f"TEXTS {verse_start}–{verse_end}"
    else:
        label = f"TEXT {verse_start}"
    return label, verse_start, verse_end


def looks_like_devanagari_extract(line: str) -> bool:
    """Heuristic: garbled Devanagari lines are punctuation-heavy / mixed-case blobs."""
    stripped = line.strip()
    if not stripped:
        return False
    if re.search(r"\)\s*\)", stripped):
        return True
    punct = sum(ch in DEVANAGARI_EXTRACT_PUNCT for ch in stripped)
    if punct >= 2:
        return True
    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return False
    has_upper = any(ch.isupper() for ch in letters)
    has_lower = any(ch.islower() for ch in letters)
    # Extracted Devanagari often looks like: DaMaR+ae}ae  (mixed case, few spaces)
    if has_upper and has_lower and stripped.count(" ") <= 1:
        return True
    return False


def looks_like_transliteration(line: str) -> bool:
    """Heuristic: transliteration is mostly lowercase Roman letters."""
    stripped = line.strip()
    if not stripped or looks_like_devanagari_extract(stripped):
        return False
    letters = [ch for ch in stripped if ch.isalpha()]
    if len(letters) < 3:
        return False
    lower_ratio = sum(ch.islower() for ch in letters) / len(letters)
    return lower_ratio >= 0.85


def split_sanskrit_and_transliteration(pre_synonyms: str) -> tuple[str, str]:
    """
    Text before SYNONYMS is: Devanagari lines, then transliteration lines.
    There is no TRANSLITERATION heading in this PDF, so we detect the switch.
    """
    sanskrit_lines: list[str] = []
    translit_lines: list[str] = []
    seen_translit = False
    for raw_line in pre_synonyms.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not seen_translit and looks_like_transliteration(line):
            seen_translit = True
        if seen_translit:
            translit_lines.append(line)
        else:
            sanskrit_lines.append(line)
    return "\n".join(sanskrit_lines).strip(), "\n".join(translit_lines).strip()


def clean_section(text: str) -> str:
    """Turn wrapped PDF lines into readable paragraphs. Keep blank-line breaks."""
    text = PAGE_MARK_RE.sub("", text)
    paragraphs: list[str] = []
    buf: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        paragraphs.append(" ".join(buf))
    return "\n\n".join(paragraphs).strip()


def page_span(text_with_markers: str, fallback_page: int) -> tuple[int, int]:
    """Smallest and largest <<<PAGE n>>> seen in this block."""
    pages = [int(n) for n in PAGE_MARK_RE.findall(text_with_markers)]
    if not pages:
        return fallback_page, fallback_page
    return min(pages), max(pages)


def parse_block_body(body: str) -> dict[str, str]:
    """Split one TEXT/TEXTS body into named sections."""
    parts = SECTION_HEADING_RE.split(body)
    # parts[0] = text before first section heading (sanskrit + transliteration)
    # then repeating: heading, content, heading, content, ...
    pre = parts[0]
    sections = {"SYNONYMS": "", "TRANSLATION": "", "PURPORT": ""}
    for i in range(1, len(parts), 2):
        name = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        sections[name] = content

    sanskrit_raw, transliteration_raw = split_sanskrit_and_transliteration(pre)
    return {
        "sanskrit_raw": clean_section(sanskrit_raw),
        "transliteration_raw": clean_section(transliteration_raw),
        "synonyms": clean_section(sections["SYNONYMS"]),
        "translation": clean_section(sections["TRANSLATION"]),
        "purport": clean_section(sections["PURPORT"]),
    }


def parse_chapter(chapter_text: str, chapter: int) -> list[dict]:
    """
    Walk TEXT/TEXTS headings in order.

    The heading regex is used as a splitter: odd chunks after the first
    heading are (kind, start, end, body_until_next_heading).
    """
    matches = list(TEXT_HEADING_RE.finditer(chapter_text))
    if not matches:
        raise ValueError(f"No TEXT/TEXTS headings found in Chapter {chapter}.")

    records = []
    for i, match in enumerate(matches):
        kind, start, end = match.group(1), match.group(2), match.group(3)
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(chapter_text)
        body = chapter_text[body_start:body_end]
        # Page of the heading itself (search backwards for the nearest marker).
        prefix = chapter_text[: match.start()]
        prefix_pages = PAGE_MARK_RE.findall(prefix)
        heading_page = int(prefix_pages[-1]) if prefix_pages else 1

        label, verse_start, verse_end = parse_verse_range(kind, start, end)
        sections = parse_block_body(body)
        # Markers sit at the *top* of each page, often before the TEXT heading,
        # so the heading page must be included explicitly.
        marked_start, marked_end = page_span(
            chapter_text[match.start() : body_end], heading_page
        )
        page_start = min(heading_page, marked_start)
        page_end = max(heading_page, marked_end)
        purport = sections["purport"]
        records.append(
            {
                "chapter": chapter,
                "verse_start": verse_start,
                "verse_end": verse_end,
                "text_label": label,
                "sanskrit_raw": sections["sanskrit_raw"],
                "transliteration_raw": sections["transliteration_raw"],
                "synonyms": sections["synonyms"],
                "translation": sections["translation"],
                "purport": purport,
                "has_purport": bool(purport),
                "page_start": page_start,
                "page_end": page_end,
            }
        )
    return records


def validate_records(records: list[dict], chapter: int) -> dict:
    """
    Mechanical checks. These do not prove every translation is semantically
    perfect; they catch missing verses, empty fields, and overlap.
    """
    issues: list[str] = []
    covered: list[int] = []
    empty_translation = [rec["text_label"] for rec in records if not rec["translation"]]
    empty_synonyms = [rec["text_label"] for rec in records if not rec["synonyms"]]
    empty_transliteration = [rec["text_label"] for rec in records if not rec["transliteration_raw"]]
    empty_sanskrit = [rec["text_label"] for rec in records if not rec["sanskrit_raw"]]
    for rec in records:
        rng = list(range(rec["verse_start"], rec["verse_end"] + 1))
        covered.extend(rng)
        if not rec["translation"]:
            issues.append(f"{rec['text_label']}: empty TRANSLATION")
        if not rec["synonyms"]:
            issues.append(f"{rec['text_label']}: empty SYNONYMS")
        if not rec["transliteration_raw"]:
            issues.append(f"{rec['text_label']}: empty transliteration")
        if not rec["sanskrit_raw"]:
            issues.append(f"{rec['text_label']}: empty sanskrit_raw")
        if rec["verse_start"] > rec["verse_end"]:
            issues.append(f"{rec['text_label']}: verse_start > verse_end")
        if rec["page_start"] > rec["page_end"]:
            issues.append(f"{rec['text_label']}: page_start > page_end")
        if rec["has_purport"] != bool(rec["purport"]):
            issues.append(f"{rec['text_label']}: has_purport flag mismatch")
        if rec["chapter"] != chapter:
            issues.append(f"{rec['text_label']}: wrong chapter field")

    unique = sorted(set(covered))
    expected_max = unique[-1] if unique else 0
    missing = [n for n in range(1, expected_max + 1) if n not in unique]
    duplicates = sorted({n for n in covered if covered.count(n) > 1})
    if missing:
        issues.append(f"missing verse numbers: {missing}")
    if duplicates:
        issues.append(f"duplicate verse numbers: {duplicates}")

    return {
        "n_records": len(records),
        "n_text": sum(1 for r in records if r["text_label"].split()[0] == "TEXT"),
        "n_texts": sum(1 for r in records if r["text_label"].split()[0] == "TEXTS"),
        "n_with_purport": sum(1 for r in records if r["has_purport"]),
        "n_without_purport": sum(1 for r in records if not r["has_purport"]),
        "verse_min": unique[0] if unique else None,
        "verse_max": unique[-1] if unique else None,
        "n_traditional_verses": len(unique),
        "missing_verses": missing,
        "duplicate_verses": duplicates,
        "empty_translation": empty_translation,
        "empty_synonyms": empty_synonyms,
        "empty_transliteration": empty_transliteration,
        "empty_sanskrit": empty_sanskrit,
        "issues": issues,
    }


def extract_chapter(
    pdf_path: Path,
    chapter: int = 1,
    full_text: str | None = None,
    banners: list[re.Match[str]] | None = None,
) -> tuple[list[dict], dict]:
    if full_text is None:
        pages = extract_pages(pdf_path)
        full_text = join_pages_with_markers(pages)
    if banners is None:
        banners = find_chapter_banners(full_text)
    chapter_text = slice_chapter(full_text, chapter, banners=banners)
    records = parse_chapter(chapter_text, chapter)
    report = validate_records(records, chapter)
    expected = BBT_VERSE_COUNTS.get(chapter)
    report["expected_traditional_verses"] = expected
    if expected is not None and report["n_traditional_verses"] != expected:
        report["issues"].append(
            f"verse count {report['n_traditional_verses']} != edition comparison {expected}"
        )
    return records, report


def extract_book(pdf_path: Path) -> tuple[list[dict], dict]:
    """Extract chapters 1–18. PDF is read once."""
    pages = extract_pages(pdf_path)
    full_text = join_pages_with_markers(pages)
    banners = find_chapter_banners(full_text)

    all_records: list[dict] = []
    chapter_reports: list[dict] = []
    for chapter in range(1, 19):
        records, report = extract_chapter(
            pdf_path, chapter=chapter, full_text=full_text, banners=banners
        )
        report = {"chapter": chapter, **report}
        chapter_reports.append(report)
        all_records.extend(records)

    grouped = [r for r in all_records if r["text_label"].split()[0] == "TEXTS"]
    totals = {
        "n_records": len(all_records),
        "n_text": sum(1 for r in all_records if r["text_label"].split()[0] == "TEXT"),
        "n_texts": len(grouped),
        "n_with_purport": sum(1 for r in all_records if r["has_purport"]),
        "n_without_purport": sum(1 for r in all_records if not r["has_purport"]),
        "n_traditional_verses": sum(ch["n_traditional_verses"] for ch in chapter_reports),
        "expected_traditional_verses": sum(BBT_VERSE_COUNTS.values()),
        "chapters_with_issues": [ch["chapter"] for ch in chapter_reports if ch["issues"]],
        "empty_translation_total": sum(len(ch["empty_translation"]) for ch in chapter_reports),
        "empty_synonyms_total": sum(len(ch["empty_synonyms"]) for ch in chapter_reports),
        "missing_or_duplicate_chapters": [
            ch["chapter"]
            for ch in chapter_reports
            if ch["missing_verses"] or ch["duplicate_verses"]
        ],
    }
    return all_records, {"chapters": chapter_reports, "totals": totals}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Bhagavad Gita chapters from the PDF into JSON."
    )
    parser.add_argument("--pdf", type=Path, default=PDF_DEFAULT)
    parser.add_argument(
        "--chapter",
        type=int,
        default=None,
        help="Extract only this chapter (1–18). Default is all 18 chapters.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=VALIDATION_DEFAULT)
    args = parser.parse_args()

    if args.chapter is not None:
        if args.chapter < 1 or args.chapter > 18:
            raise SystemExit("--chapter must be between 1 and 18.")
        records, report = extract_chapter(args.pdf, chapter=args.chapter)
        output = args.output or Path(__file__).resolve().parent / f"chapter{args.chapter}_sample.json"
        report_payload = {"chapters": [{"chapter": args.chapter, **report}], "totals": report}
    else:
        records, report_payload = extract_book(args.pdf)
        output = args.output or OUTPUT_DEFAULT
        report = report_payload["totals"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report_payload, indent=2))
    print(f"Wrote {len(records)} records to {output}")
    print(f"Wrote validation report to {args.report}")


if __name__ == "__main__":
    main()
