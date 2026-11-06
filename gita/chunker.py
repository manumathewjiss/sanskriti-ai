"""
Turn structured Gita verse records into English retrieval chunks.

This does NOT create embeddings. It only decides what text belongs in
one searchable unit, while keeping chapter/verse identity for citations.

Why we do not use the 657 book records as-is
--------------------------------------------
Translations are short (median ~31 words). Purports vary from ~17 words
to ~2700 words. multilingual-e5-large (the model from the earlier project)
only accepts 512 tokens, roughly 350–400 English words. A 2700-word purport
would be truncated, so most of that commentary would never be searchable.

Recommended design (from this corpus, not a generic RAG default)
----------------------------------------------------------------
- TRANSLATION: always one intact chunk. Never split (max in this book is 131 words).
- PURPORT: one chunk if <= 350 words; otherwise pack paragraphs to ~350 words
  with ~40-word overlap. Split a paragraph by sentences only if it is itself
  longer than 350 words.
- SYNONYMS: do not embed. Keep on the source record as glossary metadata.
- GROUPED TEXTS: stay one identity (e.g. 1.16–18), not fake per-verse rows.
- NO PURPORT: translation chunk only.

Each chunk text starts with a short citation line so a retrieved passage
can still be named "Bhagavad Gita 2.47 — Purport" even if it is part 3.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CORPUS_DEFAULT = Path(__file__).resolve().parent / "gita_structured.json"
SAMPLE_OUTPUT_DEFAULT = Path(__file__).resolve().parent / "chunk_samples.json"
CHUNKS_OUTPUT_DEFAULT = Path(__file__).resolve().parent / "gita_chunks.json"
VALIDATION_DEFAULT = Path(__file__).resolve().parent / "gita_chunk_validation.json"

# Split purports only when they exceed this. Chosen because:
# - median purport is already 220 words (no split)
# - ~90th percentile is 556 words (usually 2 parts)
# - E5's 512-token window is about 350–400 English words once a citation line is added
TARGET_PURPORT_WORDS = 350
# Allow a part to go a little over the target so we do not leave a stub
# paragraph (e.g. 120 words) as its own chunk.
MAX_PURPORT_WORDS = 420
# Repeat the tail of the previous part so a page-break / mid-argument cut is still findable.
OVERLAP_WORDS = 40

SAMPLE_KEYS = [
    (1, 4),  # short purport
    (1, 5),  # no purport
    (1, 16),  # grouped TEXTS, modest purport
    (2, 47),  # medium purport, famous verse
    (9, 2),  # very long single-verse purport
    (11, 16),  # tiny purport
    (16, 1),  # grouped TEXTS + extremely long purport
]


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def source_record_id(record: dict) -> str:
    if record["verse_start"] == record["verse_end"]:
        return f"{record['chapter']}.{record['verse_start']}"
    return f"{record['chapter']}.{record['verse_start']}-{record['verse_end']}"


def citation_label(record: dict) -> str:
    rid = source_record_id(record)
    return f"Bhagavad-gita {rid}"


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z“"‘])', text)
    return [p.strip() for p in parts if p.strip()]


def pack_units(
    units: list[str],
    target: int,
    overlap_words: int,
    max_words: int = MAX_PURPORT_WORDS,
) -> list[str]:
    """Greedy pack of paragraphs/sentences into parts of about `target` words."""
    parts: list[str] = []
    buf: list[str] = []
    buf_words = 0

    def flush() -> None:
        nonlocal buf, buf_words
        if buf:
            parts.append(" ".join(buf).strip())
            buf = []
            buf_words = 0

    for unit in units:
        w = word_count(unit)
        would_exceed_target = buf and buf_words + w > target
        can_stretch = buf_words + w <= max_words
        if would_exceed_target and not can_stretch:
            flush()
            if parts and overlap_words > 0:
                tail = parts[-1].split()[-overlap_words:]
                overlap = " ".join(tail)
                buf = [overlap]
                buf_words = word_count(overlap)
        buf.append(unit)
        buf_words += w
    flush()
    return parts


def split_purport(purport: str, target: int = TARGET_PURPORT_WORDS, overlap_words: int = OVERLAP_WORDS) -> list[str]:
    if word_count(purport) <= target:
        return [purport.strip()]

    units: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", purport) if p.strip()]
    for para in paragraphs:
        if word_count(para) <= target:
            units.append(para)
        else:
            units.extend(split_sentences(para))
    return pack_units(units, target=target, overlap_words=overlap_words)


def record_to_chunks(record: dict) -> list[dict]:
    rid = source_record_id(record)
    cite = citation_label(record)
    chunks: list[dict] = []

    trans_text = f"{cite} — Translation\n\n{record['translation'].strip()}"
    chunks.append(
        {
            "chunk_id": f"{rid}#translation",
            "source_record_id": rid,
            "chapter": record["chapter"],
            "verse_start": record["verse_start"],
            "verse_end": record["verse_end"],
            "text_label": record["text_label"],
            "section": "translation",
            "part": 1,
            "n_parts": 1,
            "text": trans_text,
            "page_start": record["page_start"],
            "page_end": record["page_end"],
        }
    )

    if not record.get("has_purport") or not record.get("purport", "").strip():
        return chunks

    parts = split_purport(record["purport"])
    n_parts = len(parts)
    for i, part_text in enumerate(parts, start=1):
        if n_parts == 1:
            heading = f"{cite} — Purport"
        else:
            heading = f"{cite} — Purport (part {i} of {n_parts})"
        chunks.append(
            {
                "chunk_id": f"{rid}#purport.{i}",
                "source_record_id": rid,
                "chapter": record["chapter"],
                "verse_start": record["verse_start"],
                "verse_end": record["verse_end"],
                "text_label": record["text_label"],
                "section": "purport",
                "part": i,
                "n_parts": n_parts,
                "text": f"{heading}\n\n{part_text.strip()}",
                "page_start": record["page_start"],
                "page_end": record["page_end"],
            }
        )
    return chunks


def corpus_to_chunks(records: list[dict]) -> list[dict]:
    """Apply the approved rules to every source record, in book order."""
    chunks: list[dict] = []
    for record in records:
        chunks.extend(record_to_chunks(record))
    return chunks


def chunk_body_text(chunk: dict) -> str:
    """Purport/translation text without the citation heading line."""
    text = chunk["text"]
    if "\n\n" in text:
        return text.split("\n\n", 1)[1].strip()
    return text.strip()


def purport_parts_for_source(chunks: list[dict], source_id: str) -> list[str]:
    parts = [
        chunk_body_text(c)
        for c in chunks
        if c["source_record_id"] == source_id and c["section"] == "purport"
    ]
    return parts


def reconstruct_purport(parts: list[str], overlap_words: int = OVERLAP_WORDS) -> str:
    """Undo overlap so we can compare split purports with the source text."""
    if not parts:
        return ""
    words_out = parts[0].split()
    for part in parts[1:]:
        part_words = part.split()
        if (
            overlap_words
            and len(part_words) >= overlap_words
            and len(words_out) >= overlap_words
            and part_words[:overlap_words] == words_out[-overlap_words:]
        ):
            words_out.extend(part_words[overlap_words:])
        else:
            words_out.extend(part_words)
    return " ".join(words_out)


def normalize_words(text: str) -> list[str]:
    return text.split()


REQUIRED_FIELDS = [
    "chunk_id",
    "source_record_id",
    "chapter",
    "verse_start",
    "verse_end",
    "text_label",
    "section",
    "part",
    "n_parts",
    "text",
    "page_start",
    "page_end",
]


def validate_chunks(records: list[dict], chunks: list[dict]) -> dict:
    issues: list[str] = []
    trans = [c for c in chunks if c["section"] == "translation"]
    purps = [c for c in chunks if c["section"] == "purport"]
    other = [c for c in chunks if c["section"] not in ("translation", "purport")]
    if other:
        issues.append(f"unexpected sections: {sorted({c['section'] for c in other})}")

    ids = [c["chunk_id"] for c in chunks]
    dup_ids = sorted({i for i in ids if ids.count(i) > 1})
    if dup_ids:
        issues.append(f"duplicate chunk_id: {dup_ids[:10]}")

    empty = [c["chunk_id"] for c in chunks if not str(c.get("text", "")).strip()]
    if empty:
        issues.append(f"empty text: {empty[:10]}")

    missing_fields = []
    for c in chunks:
        for field in REQUIRED_FIELDS:
            if field not in c or c[field] in (None, ""):
                missing_fields.append(f"{c.get('chunk_id')} missing {field}")
        if c["verse_start"] > c["verse_end"]:
            issues.append(f"{c['chunk_id']}: verse_start > verse_end")
        if c["part"] < 1 or c["part"] > c["n_parts"]:
            issues.append(f"{c['chunk_id']}: part {c['part']} not in 1..{c['n_parts']}")
        if c["chapter"] < 1 or c["chapter"] > 18:
            issues.append(f"{c['chunk_id']}: bad chapter {c['chapter']}")
    if missing_fields:
        issues.extend(missing_fields[:20])

    source_ids = [source_record_id(r) for r in records]
    if len(source_ids) != len(set(source_ids)):
        issues.append("duplicate source_record_id in corpus")
    chunk_sources = {c["source_record_id"] for c in chunks}
    missing_sources = [sid for sid in source_ids if sid not in chunk_sources]
    if missing_sources:
        issues.append(f"sources with no chunks: {missing_sources[:10]}")

    trans_by_source = {c["source_record_id"] for c in trans}
    missing_trans = [sid for sid in source_ids if sid not in trans_by_source]
    if missing_trans:
        issues.append(f"sources missing translation chunk: {missing_trans[:10]}")
    extra_trans = len(trans) - len(records)

    purport_sources = {c["source_record_id"] for c in purps}
    with_purport = [r for r in records if r.get("has_purport") and r.get("purport", "").strip()]
    without_purport = [r for r in records if not (r.get("has_purport") and r.get("purport", "").strip())]
    missing_purp = [
        source_record_id(r) for r in with_purport if source_record_id(r) not in purport_sources
    ]
    fake_purp = [
        source_record_id(r) for r in without_purport if source_record_id(r) in purport_sources
    ]
    if missing_purp:
        issues.append(f"has_purport records missing purport chunks: {missing_purp[:10]}")
    if fake_purp:
        issues.append(f"no-purport records got purport chunks: {fake_purp[:10]}")

    covered: list[tuple[int, int]] = []
    for r in records:
        for v in range(r["verse_start"], r["verse_end"] + 1):
            covered.append((r["chapter"], v))
    chunk_covered: list[tuple[int, int]] = []
    for c in trans:
        for v in range(c["verse_start"], c["verse_end"] + 1):
            chunk_covered.append((c["chapter"], v))
    missing_verses = sorted(set(covered) - set(chunk_covered))
    extra_verses = sorted(set(chunk_covered) - set(covered))
    if missing_verses:
        issues.append(f"verse numbers missing from chunks: {missing_verses[:10]}")
    if extra_verses:
        issues.append(f"unexpected verse numbers in chunks: {extra_verses[:10]}")

    grouped_bad = []
    for r in records:
        if r["verse_start"] == r["verse_end"]:
            continue
        sid = source_record_id(r)
        for c in chunks:
            if c["source_record_id"] != sid:
                continue
            if c["verse_start"] != r["verse_start"] or c["verse_end"] != r["verse_end"]:
                grouped_bad.append(c["chunk_id"])
            if c["verse_start"] == c["verse_end"]:
                grouped_bad.append(f"{c['chunk_id']} flattened to single verse")
    if grouped_bad:
        issues.append(f"grouped TEXTS range broken: {grouped_bad[:10]}")

    # per-source purport split stats
    parts_per_source: dict[str, int] = defaultdict(int)
    for c in purps:
        parts_per_source[c["source_record_id"]] = max(parts_per_source[c["source_record_id"]], c["n_parts"])
    n_unsplit = sum(1 for n in parts_per_source.values() if n == 1)
    n_split = sum(1 for n in parts_per_source.values() if n > 1)
    max_parts = max(parts_per_source.values()) if parts_per_source else 0

    body_words = [word_count(chunk_body_text(c)) for c in purps]
    full_words = [word_count(c["text"]) for c in purps]

    def bucket(w: int) -> str:
        if w < 80:
            return "<80"
        if w < 200:
            return "80-199"
        if w <= 350:
            return "200-350"
        if w <= 420:
            return "351-420"
        return ">420"

    body_buckets = dict(Counter(bucket(w) for w in body_words))
    over_body = [
        {"chunk_id": c["chunk_id"], "body_words": word_count(chunk_body_text(c)), "full_words": word_count(c["text"])}
        for c in purps
        if word_count(chunk_body_text(c)) > 420
    ]

    # overlap check: consecutive purport parts should share the configured tail
    overlap_ok = 0
    overlap_fail = 0
    overlap_fail_ids = []
    by_source: dict[str, list[dict]] = defaultdict(list)
    for c in purps:
        by_source[c["source_record_id"]].append(c)
    for sid, group in by_source.items():
        group = sorted(group, key=lambda c: c["part"])
        if len(group) < 2:
            continue
        for a, b in zip(group, group[1:]):
            tail = chunk_body_text(a).split()[-OVERLAP_WORDS:]
            head = chunk_body_text(b).split()[:OVERLAP_WORDS]
            if tail == head:
                overlap_ok += 1
            else:
                overlap_fail += 1
                overlap_fail_ids.append(f"{a['chunk_id']} -> {b['chunk_id']}")

    # content preservation on split purports
    preservation_fail = []
    preservation_checked = 0
    for r in with_purport:
        sid = source_record_id(r)
        parts = purport_parts_for_source(chunks, sid)
        if len(parts) <= 1:
            # single chunk: body should match source words
            if normalize_words(parts[0] if parts else "") != normalize_words(r["purport"]):
                preservation_fail.append(sid)
            continue
        preservation_checked += 1
        rebuilt = reconstruct_purport(parts)
        if normalize_words(rebuilt) != normalize_words(r["purport"]):
            preservation_fail.append(sid)
    if preservation_fail:
        issues.append(f"purport reconstruct mismatch: {preservation_fail[:10]}")

    def stats(xs: list[int]) -> dict:
        if not xs:
            return {}
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        return {
            "n": n,
            "min": xs_sorted[0],
            "median": xs_sorted[n // 2],
            "mean": round(sum(xs_sorted) / n, 1),
            "max": xs_sorted[-1],
        }

    return {
        "n_source_records": len(records),
        "n_chunks": len(chunks),
        "n_translation_chunks": len(trans),
        "n_purport_chunks": len(purps),
        "n_sources_with_purport": len(with_purport),
        "n_sources_without_purport": len(without_purport),
        "n_purports_unsplit": n_unsplit,
        "n_purports_split": n_split,
        "max_purport_parts": max_parts,
        "n_empty_text": len(empty),
        "n_duplicate_chunk_ids": len(dup_ids),
        "n_missing_source_record_id": sum(1 for c in chunks if not c.get("source_record_id")),
        "n_traditional_verses_in_sources": len(set(covered)),
        "n_traditional_verses_in_chunks": len(set(chunk_covered)),
        "translation_chunks_minus_sources": extra_trans,
        "purport_body_word_stats": stats(body_words),
        "purport_full_text_word_stats": stats(full_words),
        "purport_body_word_buckets": body_buckets,
        "n_purport_body_over_420": len(over_body),
        "purport_body_over_420": over_body[:20],
        "overlap_pairs_ok": overlap_ok,
        "overlap_pairs_fail": overlap_fail,
        "overlap_fail_examples": overlap_fail_ids[:10],
        "n_split_purports_checked_for_preservation": preservation_checked,
        "n_purport_preservation_fail": len(preservation_fail),
        "issues": issues,
    }
    for rec in records:
        if rec["chapter"] == chapter and rec["verse_start"] <= verse <= rec["verse_end"]:
            return rec
    raise KeyError(f"No record covers {chapter}.{verse}")


def find_record(records: list[dict], chapter: int, verse: int) -> dict:
    for rec in records:
        if rec["chapter"] == chapter and rec["verse_start"] <= verse <= rec["verse_end"]:
            return rec
    raise KeyError(f"No record covers {chapter}.{verse}")


def sample_transformations(records: list[dict]) -> list[dict]:
    samples = []
    for chapter, verse in SAMPLE_KEYS:
        rec = find_record(records, chapter, verse)
        chunks = record_to_chunks(rec)
        samples.append(
            {
                "source_record_id": source_record_id(rec),
                "source_summary": {
                    "text_label": rec["text_label"],
                    "chapter": rec["chapter"],
                    "verse_start": rec["verse_start"],
                    "verse_end": rec["verse_end"],
                    "has_purport": rec["has_purport"],
                    "translation_words": word_count(rec["translation"]),
                    "purport_words": word_count(rec["purport"]),
                    "page_start": rec["page_start"],
                    "page_end": rec["page_end"],
                },
                "n_chunks": len(chunks),
                "chunk_ids": [c["chunk_id"] for c in chunks],
                "chunks": chunks,
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval chunks from structured Gita JSON.")
    parser.add_argument("--corpus", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Transform only the representative records (Step 4 sample).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=VALIDATION_DEFAULT)
    args = parser.parse_args()

    records = json.loads(args.corpus.read_text(encoding="utf-8"))

    if args.sample_only:
        samples = sample_transformations(records)
        output = args.output or SAMPLE_OUTPUT_DEFAULT
        payload = {
            "design": {
                "target_purport_words": TARGET_PURPORT_WORDS,
                "max_purport_words": MAX_PURPORT_WORDS,
                "overlap_words": OVERLAP_WORDS,
                "translation": "one intact chunk per source record",
                "synonyms": "not embedded; remain on the source record",
                "grouped_texts": "kept as one source identity",
                "no_purport": "translation chunk only",
            },
            "n_source_records": len(samples),
            "n_chunks": sum(s["n_chunks"] for s in samples),
            "samples": samples,
        }
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({k: payload[k] for k in ("design", "n_source_records", "n_chunks")}, indent=2))
        print(f"Wrote {output}")
        return

    chunks = corpus_to_chunks(records)
    output = args.output or CHUNKS_OUTPUT_DEFAULT
    output.write_text(json.dumps(chunks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = validate_chunks(records, chunks)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {len(chunks)} chunks to {output}")
    print(f"Wrote validation report to {args.report}")


if __name__ == "__main__":
    main()
