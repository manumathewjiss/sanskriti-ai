"""
BM25-only retrieval baseline on the SAME 1,443 Gita chunks and frozen questions.

This does not use E5, FAISS, hybrid fusion, or gold labels in the query.

Tokenizer (documented, not a verse router):
  - lowercase
  - treat "bhagavad-gita" like "bhagavad gita"
  - keep chapter.verse tokens such as 11.32 and 1.16-18 as ONE token
  - split other text on non-letters/digits

Naive punctuation splitting would turn "11.32" into "11" and "32", which would
make citation lookup almost meaningless. We keep those dotted numbers intact.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from evaluate_e5 import (
    TOP_K,
    first_gold_rank,
    hit_at,
    metrics_from_ranks,
    preview,
    sha256_file,
)

ROOT = Path(__file__).resolve().parent
CHUNKS_DEFAULT = ROOT / "gita_chunks.json"
EVAL_DEFAULT = ROOT / "gita_eval_frozen.json"
E5_RESULTS_DEFAULT = ROOT / "gita_e5_eval_results.json"
E5_SUMMARY_DEFAULT = ROOT / "gita_e5_eval_summary.json"
RESULTS_DEFAULT = ROOT / "gita_bm25_eval_results.json"
SUMMARY_DEFAULT = ROOT / "gita_bm25_eval_summary.json"

# 2.47  or  grouped 1.16-18
VERSE_TOKEN_RE = re.compile(r"\d+\.\d+(?:-\d+)?")
PLACEHOLDER = " citverse{} "


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("bhagavad-gita", "bhagavad gita")
    text = text.replace("—", " ").replace("–", " ")
    verses: list[str] = []

    def stash(match: re.Match[str]) -> str:
        verses.append(match.group(0))
        return PLACEHOLDER.format(len(verses) - 1)

    held = VERSE_TOKEN_RE.sub(stash, text)
    out: list[str] = []
    for raw in re.findall(r"[a-z0-9]+", held):
        if raw.startswith("citverse"):
            out.append(verses[int(raw[len("citverse") :])])
        else:
            out.append(raw)
    return out


def tokenization_examples() -> dict[str, list[str]]:
    samples = [
        "Bhagavad Gita 11.32",
        "Bhagavad-gita 11.32 — Translation",
        "Bhagavad-gita 2.47 — Purport",
        "Bhagavad-gita 1.16-18 — Translation",
        "In Bhagavad Gita 11.32, how does Krishna describe Himself?",
        "What does Bhagavad Gita 2.47 say?",
    ]
    naive = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    return {
        "naive_split_on_non_alphanumeric": {s: naive(s) for s in samples},
        "this_baseline": {s: tokenize(s) for s in samples},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25-only Gita retrieval evaluation.")
    parser.add_argument("--chunks", type=Path, default=CHUNKS_DEFAULT)
    parser.add_argument("--eval", type=Path, default=EVAL_DEFAULT)
    parser.add_argument("--e5-results", type=Path, default=E5_RESULTS_DEFAULT)
    parser.add_argument("--e5-summary", type=Path, default=E5_SUMMARY_DEFAULT)
    parser.add_argument("--results", type=Path, default=RESULTS_DEFAULT)
    parser.add_argument("--summary", type=Path, default=SUMMARY_DEFAULT)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    eval_hash_before = sha256_file(args.eval)
    payload = json.loads(args.eval.read_text(encoding="utf-8"))
    questions = payload["questions"]
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    if len(chunks) != 1443:
        raise SystemExit(f"Expected 1443 chunks, found {len(chunks)}")
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate chunk_id in BM25 corpus.")

    token_examples = tokenization_examples()
    print("TOKENIZATION EXAMPLES")
    print(json.dumps(token_examples["this_baseline"], indent=2))

    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"BM25 indexed {len(tokenized)} documents")

    per_question = []
    source_ranks: list[int | None] = []
    chunk_ranks: list[int | None] = []

    for q in questions:
        qtext = q["question"]
        for g in q["gold_sources"]:
            for cid in g["gold_chunk_ids"]:
                if cid in qtext:
                    raise SystemExit(f"{q['id']} question contains gold chunk id")
        query_tokens = tokenize(qtext)
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.top_k]
        gold_sources = {g["source_record_id"] for g in q["gold_sources"]}
        gold_chunks = {cid for g in q["gold_sources"] for cid in g["gold_chunk_ids"]}
        top = []
        retrieved_chunk_ids = []
        retrieved_source_ids = []
        for rank, idx in enumerate(ranked, start=1):
            ch = chunks[idx]
            retrieved_chunk_ids.append(ch["chunk_id"])
            retrieved_source_ids.append(ch["source_record_id"])
            top.append(
                {
                    "rank": rank,
                    "score": float(scores[idx]),
                    "chunk_id": ch["chunk_id"],
                    "source_record_id": ch["source_record_id"],
                    "chapter": ch["chapter"],
                    "verse_start": ch["verse_start"],
                    "verse_end": ch["verse_end"],
                    "section": ch["section"],
                    "text_preview": preview(ch["text"]),
                }
            )
        if len(top) != args.top_k:
            raise SystemExit(f"{q['id']} did not return Top {args.top_k}")
        s_rank = first_gold_rank(retrieved_source_ids, gold_sources)
        c_rank = first_gold_rank(retrieved_chunk_ids, gold_chunks)
        source_ranks.append(s_rank)
        chunk_ranks.append(c_rank)
        per_question.append(
            {
                "id": q["id"],
                "category": q["category"],
                "difficulty": q["difficulty"],
                "question": q["question"],
                "query_tokens": query_tokens,
                "gold_source_record_ids": sorted(gold_sources),
                "gold_chunk_ids": sorted(gold_chunks),
                "source_gold_rank": s_rank,
                "chunk_gold_rank": c_rank,
                "source_hit_at_1": hit_at(s_rank, 1),
                "source_hit_at_5": hit_at(s_rank, 5),
                "source_hit_at_10": hit_at(s_rank, 10),
                "chunk_hit_at_1": hit_at(c_rank, 1),
                "chunk_hit_at_5": hit_at(c_rank, 5),
                "chunk_hit_at_10": hit_at(c_rank, 10),
                "top_results": top,
            }
        )

    def slice_ranks(key: str) -> dict[str, dict]:
        grouped: dict[str, list[int | None]] = defaultdict(list)
        for item, rank in zip(per_question, source_ranks):
            grouped[item[key]].append(rank)
        return {name: metrics_from_ranks(rs) for name, rs in sorted(grouped.items())}

    e5_results = json.loads(args.e5_results.read_text(encoding="utf-8"))
    e5_summary = json.loads(args.e5_summary.read_text(encoding="utf-8"))
    e5_by_id = {row["id"]: row for row in e5_results}

    comparison_rows = []
    for item in per_question:
        e5 = e5_by_id[item["id"]]
        comparison_rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "e5_source_gold_rank": e5["source_gold_rank"],
                "bm25_source_gold_rank": item["source_gold_rank"],
                "e5_chunk_gold_rank": e5["chunk_gold_rank"],
                "bm25_chunk_gold_rank": item["chunk_gold_rank"],
            }
        )

    def better(a, b) -> str:
        if a is None and b is None:
            return "both_miss"
        if a is None:
            return "bm25_only"
        if b is None:
            return "e5_only"
        if a < b:
            return "e5_better"
        if b < a:
            return "bm25_better"
        return "tie"

    source_outcomes = Counter(
        better(r["e5_source_gold_rank"], r["bm25_source_gold_rank"]) for r in comparison_rows
    )

    summary = {
        "method": "BM25Okapi",
        "n_documents": len(chunks),
        "top_k": args.top_k,
        "n_questions": len(questions),
        "frozen_eval_sha256": eval_hash_before,
        "tokenization": {
            "description": (
                "lowercase; bhagavad-gita -> bhagavad gita; keep dotted verse "
                "tokens (11.32, 1.16-18); split remaining on non-alphanumerics"
            ),
            "examples": token_examples,
        },
        "scoring_rule": (
            "Any valid gold source/chunk counts as the first hit. "
            "Not all golds need to appear. Missing from top-10 => 0 for that question."
        ),
        "source_level": metrics_from_ranks(source_ranks),
        "chunk_level": metrics_from_ranks(chunk_ranks),
        "source_level_by_category": slice_ranks("category"),
        "source_level_by_difficulty": slice_ranks("difficulty"),
        "e5_source_level": e5_summary["source_level"],
        "e5_chunk_level": e5_summary["chunk_level"],
        "e5_source_level_by_category": e5_summary["source_level_by_category"],
        "source_rank_outcomes_vs_e5": dict(source_outcomes),
        "per_question_comparison": comparison_rows,
    }
    args.results.write_text(json.dumps(per_question, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    eval_hash_after = sha256_file(args.eval)
    if eval_hash_after != eval_hash_before:
        raise SystemExit("FROZEN EVAL FILE CHANGED DURING BM25 EVALUATION.")
    if eval_hash_after != e5_summary["frozen_eval_sha256"]:
        raise SystemExit("Frozen eval hash does not match the Step 7 E5 summary hash.")

    print(json.dumps({
        "source_level": summary["source_level"],
        "chunk_level": summary["chunk_level"],
        "source_level_by_category": summary["source_level_by_category"],
        "source_level_by_difficulty": summary["source_level_by_difficulty"],
        "source_rank_outcomes_vs_e5": summary["source_rank_outcomes_vs_e5"],
    }, indent=2))
    print(f"Frozen eval unchanged: {eval_hash_after}")
    print(f"Wrote {args.results}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
