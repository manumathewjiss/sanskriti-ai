"""
Citation-gated retrieval: a routing rule over EXISTING rankings.

If the question text contains an explicit dotted verse token (e.g. 11.32),
use the saved Step 9 RRF hybrid Top 10.

Otherwise use the saved Step 7 E5-only Top 10.

This does not:
  - rebuild E5 embeddings or FAISS
  - change BM25 or RRF
  - look up a verse by metadata
  - use gold / category / difficulty for routing
  - expand the detector after seeing scores

The detector is a regex over question text only:

    \\b\\d{1,2}\\.\\d{1,3}\\b

That matches 1.1, 2.47, 11.32, 18.66.
It does not parse "Chapter 1" or "verse forty-seven".

Step 10 is exploratory: the rule was motivated by the same frozen 30 questions.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from evaluate_e5 import (
    first_gold_rank,
    hit_at,
    metrics_from_ranks,
    sha256_file,
)

ROOT = Path(__file__).resolve().parent
EVAL_DEFAULT = ROOT / "gita_eval_frozen.json"
E5_RESULTS_DEFAULT = ROOT / "gita_e5_eval_results.json"
E5_SUMMARY_DEFAULT = ROOT / "gita_e5_eval_summary.json"
BM25_SUMMARY_DEFAULT = ROOT / "gita_bm25_eval_summary.json"
HYBRID_RESULTS_DEFAULT = ROOT / "gita_hybrid_rrf_eval_results.json"
HYBRID_SUMMARY_DEFAULT = ROOT / "gita_hybrid_rrf_eval_summary.json"
RESULTS_DEFAULT = ROOT / "gita_citation_gated_eval_results.json"
SUMMARY_DEFAULT = ROOT / "gita_citation_gated_eval_summary.json"

EXPECTED_FROZEN_HASH = "8da7a3ed170cdf995401e17c7481a61174b61e4e5cc5583c4282c745deaf6c27"

# Fixed before scoring. Not tuned against gold labels.
CITATION_RE = re.compile(r"\b\d{1,2}\.\d{1,3}\b")


def detect_citations(question: str) -> list[str]:
    return CITATION_RE.findall(question)


def main() -> None:
    parser = argparse.ArgumentParser(description="Citation-gated mix of E5 and RRF hybrid rankings.")
    parser.add_argument("--eval", type=Path, default=EVAL_DEFAULT)
    parser.add_argument("--e5-results", type=Path, default=E5_RESULTS_DEFAULT)
    parser.add_argument("--e5-summary", type=Path, default=E5_SUMMARY_DEFAULT)
    parser.add_argument("--bm25-summary", type=Path, default=BM25_SUMMARY_DEFAULT)
    parser.add_argument("--hybrid-results", type=Path, default=HYBRID_RESULTS_DEFAULT)
    parser.add_argument("--hybrid-summary", type=Path, default=HYBRID_SUMMARY_DEFAULT)
    parser.add_argument("--results", type=Path, default=RESULTS_DEFAULT)
    parser.add_argument("--summary", type=Path, default=SUMMARY_DEFAULT)
    args = parser.parse_args()

    eval_hash_before = sha256_file(args.eval)
    if eval_hash_before != EXPECTED_FROZEN_HASH:
        raise SystemExit(
            f"Frozen eval hash mismatch.\n expected {EXPECTED_FROZEN_HASH}\n got      {eval_hash_before}"
        )
    e5_results_hash = sha256_file(args.e5_results)
    hybrid_results_hash = sha256_file(args.hybrid_results)

    payload = json.loads(args.eval.read_text(encoding="utf-8"))
    questions = payload["questions"]
    e5_rows = {r["id"]: r for r in json.loads(args.e5_results.read_text(encoding="utf-8"))}
    hy_rows = {r["id"]: r for r in json.loads(args.hybrid_results.read_text(encoding="utf-8"))}
    e5_summary = json.loads(args.e5_summary.read_text(encoding="utf-8"))
    bm25_summary = json.loads(args.bm25_summary.read_text(encoding="utf-8"))
    hy_summary = json.loads(args.hybrid_summary.read_text(encoding="utf-8"))

    print("Citation detector examples (rule fixed before scoring):")
    for sample in (
        "What does Bhagavad Gita 11.32 say about Krishna?",
        "How can the mind become a person's friend or enemy?",
        "What does King Dhritarashtra ask at the beginning of Chapter 1?",
        "What does Bhagavad Gita 2.47 say?",
        "commentary on 15.7",
    ):
        print(f"  {sample!r} -> {detect_citations(sample)}")

    per_question = []
    source_ranks: list[int | None] = []
    chunk_ranks: list[int | None] = []
    citation_bearing = []

    for q in questions:
        qid = q["id"]
        qtext = q["question"]
        for g in q["gold_sources"]:
            for cid in g["gold_chunk_ids"]:
                if cid in qtext:
                    raise SystemExit(f"{qid} question accidentally contains gold chunk id")

        tokens = detect_citations(qtext)
        citation_detected = len(tokens) > 0
        selected = "hybrid_rrf" if citation_detected else "e5"
        chosen = hy_rows[qid] if selected == "hybrid_rrf" else e5_rows[qid]
        top = chosen["top_results"]
        if len(top) != 10:
            raise SystemExit(f"{qid} selected ranking does not have Top 10")

        gold_sources = {g["source_record_id"] for g in q["gold_sources"]}
        gold_chunks = {cid for g in q["gold_sources"] for cid in g["gold_chunk_ids"]}
        retrieved_chunk_ids = [t["chunk_id"] for t in top]
        retrieved_source_ids = [t["source_record_id"] for t in top]
        s_rank = first_gold_rank(retrieved_source_ids, gold_sources)
        c_rank = first_gold_rank(retrieved_chunk_ids, gold_chunks)
        source_ranks.append(s_rank)
        chunk_ranks.append(c_rank)

        if citation_detected:
            citation_bearing.append(
                {
                    "id": qid,
                    "question": qtext,
                    "detected_citation_tokens": tokens,
                    "selected_retriever": selected,
                }
            )

        per_question.append(
            {
                "id": qid,
                "category": q["category"],
                "difficulty": q["difficulty"],
                "question": qtext,
                "citation_detected": citation_detected,
                "detected_citation_tokens": tokens,
                "selected_retriever": selected,
                "gold_source_record_ids": sorted(gold_sources),
                "gold_chunk_ids": sorted(gold_chunks),
                "e5_source_gold_rank": e5_rows[qid]["source_gold_rank"],
                "bm25_source_gold_rank": hy_rows[qid]["bm25_source_gold_rank"],
                "rrf_source_gold_rank": hy_rows[qid]["hybrid_source_gold_rank"],
                "gated_source_gold_rank": s_rank,
                "e5_chunk_gold_rank": e5_rows[qid]["chunk_gold_rank"],
                "bm25_chunk_gold_rank": hy_rows[qid]["bm25_chunk_gold_rank"],
                "rrf_chunk_gold_rank": hy_rows[qid]["hybrid_chunk_gold_rank"],
                "gated_chunk_gold_rank": c_rank,
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

    gated_source = metrics_from_ranks(source_ranks)
    gated_chunk = metrics_from_ranks(chunk_ranks)

    def four(level: str, metric: str) -> dict:
        src_or_chunk = "source_level" if level == "source" else "chunk_level"
        gated = gated_source if level == "source" else gated_chunk
        return {
            "e5": e5_summary[src_or_chunk][metric],
            "bm25": bm25_summary[src_or_chunk][metric],
            "rrf_hybrid": hy_summary[src_or_chunk][metric],
            "citation_gated": gated[metric],
        }

    def category_four(cat: str, metric: str) -> dict:
        return {
            "e5": e5_summary["source_level_by_category"][cat][metric],
            "bm25": bm25_summary["source_level_by_category"][cat][metric],
            "rrf_hybrid": hy_summary["source_level_by_category"][cat][metric],
            "citation_gated": slice_ranks("category")[cat][metric],
        }

    n_hybrid = sum(1 for r in per_question if r["selected_retriever"] == "hybrid_rrf")
    n_e5 = sum(1 for r in per_question if r["selected_retriever"] == "e5")

    summary = {
        "method": "citation_gated",
        "exploratory_note": (
            "The routing rule was motivated by Step 7–9 results on THIS same frozen 30-question set. "
            "Step 10 is a diagnostic architecture experiment, not an unbiased held-out evaluation."
        ),
        "citation_regex": CITATION_RE.pattern,
        "routing_rule": (
            "If question text matches \\b\\d{1,2}\\.\\d{1,3}\\b, use saved Step 9 RRF hybrid Top 10; "
            "else use saved Step 7 E5 Top 10. No metadata verse lookup."
        ),
        "n_questions": len(questions),
        "n_routed_to_hybrid_rrf": n_hybrid,
        "n_routed_to_e5": n_e5,
        "citation_bearing_questions": citation_bearing,
        "reused_rankings": {
            "e5_results": str(args.e5_results.name),
            "hybrid_results": str(args.hybrid_results.name),
            "e5_results_sha256": e5_results_hash,
            "hybrid_results_sha256": hybrid_results_hash,
        },
        "frozen_eval_sha256": eval_hash_before,
        "scoring_rule": (
            "Any valid gold source/chunk counts as the first hit. "
            "Not all golds need to appear. Missing from top-10 => 0 for that question."
        ),
        "source_level": gated_source,
        "chunk_level": gated_chunk,
        "source_level_by_category": slice_ranks("category"),
        "source_level_by_difficulty": slice_ranks("difficulty"),
        "four_way_source_level": {
            "recall@1": four("source", "recall@1"),
            "recall@5": four("source", "recall@5"),
            "recall@10": four("source", "recall@10"),
            "mrr@10": four("source", "mrr@10"),
        },
        "four_way_chunk_level": {
            "recall@1": four("chunk", "recall@1"),
            "recall@5": four("chunk", "recall@5"),
            "recall@10": four("chunk", "recall@10"),
            "mrr@10": four("chunk", "mrr@10"),
        },
        "four_way_source_by_category": {
            cat: {
                "recall@1": category_four(cat, "recall@1"),
                "recall@5": category_four(cat, "recall@5"),
                "recall@10": category_four(cat, "recall@10"),
                "mrr@10": category_four(cat, "mrr@10"),
            }
            for cat in sorted(slice_ranks("category"))
        },
        "per_question_comparison": [
            {
                "id": r["id"],
                "category": r["category"],
                "citation_detected": r["citation_detected"],
                "detected_citation_tokens": r["detected_citation_tokens"],
                "selected_retriever": r["selected_retriever"],
                "e5_source_gold_rank": r["e5_source_gold_rank"],
                "bm25_source_gold_rank": r["bm25_source_gold_rank"],
                "rrf_source_gold_rank": r["rrf_source_gold_rank"],
                "gated_source_gold_rank": r["gated_source_gold_rank"],
                "e5_chunk_gold_rank": r["e5_chunk_gold_rank"],
                "rrf_chunk_gold_rank": r["rrf_chunk_gold_rank"],
                "gated_chunk_gold_rank": r["gated_chunk_gold_rank"],
            }
            for r in per_question
        ],
    }

    args.results.write_text(json.dumps(per_question, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if sha256_file(args.eval) != eval_hash_before:
        raise SystemExit("FROZEN EVAL FILE CHANGED.")
    if sha256_file(args.e5_results) != e5_results_hash:
        raise SystemExit("E5 results were overwritten.")
    if sha256_file(args.hybrid_results) != hybrid_results_hash:
        raise SystemExit("Hybrid results were overwritten.")

    print(json.dumps({
        "n_routed_to_hybrid_rrf": n_hybrid,
        "n_routed_to_e5": n_e5,
        "citation_bearing_ids": [c["id"] for c in citation_bearing],
        "source_level": gated_source,
        "chunk_level": gated_chunk,
        "four_way_source_level": summary["four_way_source_level"],
        "four_way_chunk_level": summary["four_way_chunk_level"],
        "source_level_by_category": summary["source_level_by_category"],
    }, indent=2))
    print(f"Frozen eval unchanged: {eval_hash_before}")
    print(f"Wrote {args.results}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
