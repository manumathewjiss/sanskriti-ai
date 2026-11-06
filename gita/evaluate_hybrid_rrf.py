"""
Hybrid retrieval: Reciprocal Rank Fusion of existing E5+FAISS and Step-8 BM25.

This step does NOT:
  - re-embed corpus chunks
  - rebuild gita_e5.faiss
  - change BM25 tokenization
  - change frozen questions
  - use a reranker, LLM, or citation router

Saved Step 7 / Step 8 result files only store Top 10, so this script
reruns RETRIEVAL at depth 50:

  E5:  encode queries only, search the existing FAISS index
  BM25: rebuild the in-memory BM25Okapi object from the same 1,443 texts
        with the same tokenizer as evaluate_bm25.py

RRF (k=60, fixed before looking at hybrid scores):

  RRF(d) = 1/(k + rank_E5(d))  if d is in E5 Top 50, else 0
         + 1/(k + rank_BM25(d)) if d is in BM25 Top 50, else 0

Ranks are 1-based. Fusion identity is chunk_id.

Tie-break (fixed before evaluation), all descending preference:
  1. higher RRF score
  2. better (lower) E5 rank; missing E5 rank loses
  3. better (lower) BM25 rank; missing BM25 rank loses
  4. lexicographic chunk_id
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_bm25 import tokenize
from evaluate_e5 import (
    TOP_K,
    first_gold_rank,
    hit_at,
    metrics_from_ranks,
    preview,
    query_input,
    sha256_file,
)
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent
CHUNKS_DEFAULT = ROOT / "gita_chunks.json"
EVAL_DEFAULT = ROOT / "gita_eval_frozen.json"
INDEX_DEFAULT = ROOT / "gita_e5.faiss"
META_DEFAULT = ROOT / "gita_e5_metadata.json"
E5_RESULTS_DEFAULT = ROOT / "gita_e5_eval_results.json"
E5_SUMMARY_DEFAULT = ROOT / "gita_e5_eval_summary.json"
BM25_RESULTS_DEFAULT = ROOT / "gita_bm25_eval_results.json"
BM25_SUMMARY_DEFAULT = ROOT / "gita_bm25_eval_summary.json"
RESULTS_DEFAULT = ROOT / "gita_hybrid_rrf_eval_results.json"
SUMMARY_DEFAULT = ROOT / "gita_hybrid_rrf_eval_summary.json"

MODEL_NAME = "intfloat/multilingual-e5-large"
RRF_K = 60
CANDIDATE_DEPTH = 50
EXPECTED_FROZEN_HASH = "8da7a3ed170cdf995401e17c7481a61174b61e4e5cc5583c4282c745deaf6c27"


def rrf_contrib(rank: int | None, k: int = RRF_K) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (k + rank)


def fuse_ranks(
    e5_rank_by_id: dict[str, int],
    bm25_rank_by_id: dict[str, int],
) -> list[tuple[str, float, int | None, int | None, float, float]]:
    union = set(e5_rank_by_id) | set(bm25_rank_by_id)
    rows = []
    for cid in union:
        e5_r = e5_rank_by_id.get(cid)
        bm_r = bm25_rank_by_id.get(cid)
        e5_c = rrf_contrib(e5_r)
        bm_c = rrf_contrib(bm_r)
        score = e5_c + bm_c
        rows.append((cid, score, e5_r, bm_r, e5_c, bm_c))

    def sort_key(row: tuple[str, float, int | None, int | None, float, float]):
        cid, score, e5_r, bm_r, _, _ = row
        e5_key = e5_r if e5_r is not None else 10**9
        bm_key = bm_r if bm_r is not None else 10**9
        return (-score, e5_key, bm_key, cid)

    rows.sort(key=sort_key)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid RRF of E5+FAISS and BM25.")
    parser.add_argument("--chunks", type=Path, default=CHUNKS_DEFAULT)
    parser.add_argument("--eval", type=Path, default=EVAL_DEFAULT)
    parser.add_argument("--index", type=Path, default=INDEX_DEFAULT)
    parser.add_argument("--metadata", type=Path, default=META_DEFAULT)
    parser.add_argument("--e5-results", type=Path, default=E5_RESULTS_DEFAULT)
    parser.add_argument("--e5-summary", type=Path, default=E5_SUMMARY_DEFAULT)
    parser.add_argument("--bm25-results", type=Path, default=BM25_RESULTS_DEFAULT)
    parser.add_argument("--bm25-summary", type=Path, default=BM25_SUMMARY_DEFAULT)
    parser.add_argument("--results", type=Path, default=RESULTS_DEFAULT)
    parser.add_argument("--summary", type=Path, default=SUMMARY_DEFAULT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--candidate-depth", type=int, default=CANDIDATE_DEPTH)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    if args.rrf_k != 60:
        raise SystemExit("Step 9 primary experiment requires k=60.")
    if args.candidate_depth != 50:
        raise SystemExit("Step 9 primary experiment requires candidate depth 50.")

    eval_hash_before = sha256_file(args.eval)
    if eval_hash_before != EXPECTED_FROZEN_HASH:
        raise SystemExit(
            f"Frozen eval hash mismatch.\n expected {EXPECTED_FROZEN_HASH}\n got      {eval_hash_before}"
        )
    faiss_hash_before = sha256_file(args.index)
    e5_results_hash_before = sha256_file(args.e5_results)
    bm25_results_hash_before = sha256_file(args.bm25_results)

    payload = json.loads(args.eval.read_text(encoding="utf-8"))
    questions = payload["questions"]
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    if len(chunks) != 1443:
        raise SystemExit(f"Expected 1443 chunks, found {len(chunks)}")
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    if len(chunk_by_id) != 1443:
        raise SystemExit("Duplicate chunk_id in corpus.")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    e5_saved = {r["id"]: r for r in json.loads(args.e5_results.read_text(encoding="utf-8"))}
    bm25_saved = {r["id"]: r for r in json.loads(args.bm25_results.read_text(encoding="utf-8"))}
    e5_summary = json.loads(args.e5_summary.read_text(encoding="utf-8"))
    bm25_summary = json.loads(args.bm25_summary.read_text(encoding="utf-8"))

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(str(args.index))
    if index.ntotal != 1443 or len(metadata) != 1443:
        raise SystemExit("E5 index/metadata size is not 1443.")
    if faiss_hash_before != sha256_file(args.index):
        raise SystemExit("FAISS file changed while loading.")

    print(f"Frozen eval sha256: {eval_hash_before}")
    print(f"Reusing FAISS {args.index.name} sha256={faiss_hash_before}")
    print("Rerunning query-only E5 search at depth 50 (corpus embeddings NOT regenerated).")
    print("Rebuilding in-memory BM25 from the same chunk texts / tokenizer (index file not used).")

    model = SentenceTransformer(args.model)
    model.eval()
    query_texts = []
    for q in questions:
        qtext = q["question"]
        for g in q["gold_sources"]:
            for cid in g["gold_chunk_ids"]:
                if cid in qtext:
                    raise SystemExit(f"{q['id']} question accidentally contains gold chunk id")
        query_texts.append(query_input(qtext))

    with torch.inference_mode():
        q_emb = model.encode(
            query_texts,
            batch_size=8,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device="cpu",
        )
    q_emb = np.asarray(q_emb, dtype=np.float32)
    e5_scores, e5_rows = index.search(q_emb, args.candidate_depth)
    if e5_scores.shape != (len(questions), args.candidate_depth):
        raise SystemExit("E5 search did not return Top-50 for every question.")

    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"BM25 indexed {len(tokenized)} documents (must be 1443)")

    per_question = []
    source_ranks: list[int | None] = []
    chunk_ranks: list[int | None] = []

    for i, q in enumerate(questions):
        gold_sources = {g["source_record_id"] for g in q["gold_sources"]}
        gold_chunks = {cid for g in q["gold_sources"] for cid in g["gold_chunk_ids"]}

        e5_rank_by_id: dict[str, int] = {}
        e5_score_by_id: dict[str, float] = {}
        for rank in range(args.candidate_depth):
            meta = metadata[int(e5_rows[i, rank])]
            cid = meta["chunk_id"]
            if cid not in e5_rank_by_id:
                e5_rank_by_id[cid] = rank + 1
                e5_score_by_id[cid] = float(e5_scores[i, rank])

        q_tokens = tokenize(q["question"])
        bm_scores = bm25.get_scores(q_tokens)
        bm_order = sorted(range(len(bm_scores)), key=lambda j: bm_scores[j], reverse=True)
        bm25_rank_by_id: dict[str, int] = {}
        bm25_score_by_id: dict[str, float] = {}
        for rank, idx in enumerate(bm_order[: args.candidate_depth], start=1):
            cid = chunks[idx]["chunk_id"]
            bm25_rank_by_id[cid] = rank
            bm25_score_by_id[cid] = float(bm_scores[idx])

        fused = fuse_ranks(e5_rank_by_id, bm25_rank_by_id)
        hybrid_top = fused[: args.top_k]
        if len(hybrid_top) != args.top_k:
            raise SystemExit(f"{q['id']} did not produce Hybrid Top {args.top_k}")

        retrieved_chunk_ids = []
        retrieved_source_ids = []
        top = []
        for hy_rank, (cid, rrf, e5_r, bm_r, e5_c, bm_c) in enumerate(hybrid_top, start=1):
            ch = chunk_by_id[cid]
            retrieved_chunk_ids.append(cid)
            retrieved_source_ids.append(ch["source_record_id"])
            top.append(
                {
                    "rank": hy_rank,
                    "chunk_id": cid,
                    "source_record_id": ch["source_record_id"],
                    "chapter": ch["chapter"],
                    "verse_start": ch["verse_start"],
                    "verse_end": ch["verse_end"],
                    "section": ch["section"],
                    "rrf_score": rrf,
                    "e5_rank": e5_r,
                    "bm25_rank": bm_r,
                    "e5_contribution": e5_c,
                    "bm25_contribution": bm_c,
                    "e5_raw_score": e5_score_by_id.get(cid),
                    "bm25_raw_score": bm25_score_by_id.get(cid),
                    "text_preview": preview(ch["text"]),
                }
            )

        s_rank = first_gold_rank(retrieved_source_ids, gold_sources)
        c_rank = first_gold_rank(retrieved_chunk_ids, gold_chunks)
        source_ranks.append(s_rank)
        chunk_ranks.append(c_rank)

        e5_row = e5_saved[q["id"]]
        bm_row = bm25_saved[q["id"]]
        per_question.append(
            {
                "id": q["id"],
                "category": q["category"],
                "difficulty": q["difficulty"],
                "question": q["question"],
                "gold_source_record_ids": sorted(gold_sources),
                "gold_chunk_ids": sorted(gold_chunks),
                "e5_source_gold_rank": e5_row["source_gold_rank"],
                "bm25_source_gold_rank": bm_row["source_gold_rank"],
                "hybrid_source_gold_rank": s_rank,
                "e5_chunk_gold_rank": e5_row["chunk_gold_rank"],
                "bm25_chunk_gold_rank": bm_row["chunk_gold_rank"],
                "hybrid_chunk_gold_rank": c_rank,
                "source_hit_at_1": hit_at(s_rank, 1),
                "source_hit_at_5": hit_at(s_rank, 5),
                "source_hit_at_10": hit_at(s_rank, 10),
                "chunk_hit_at_1": hit_at(c_rank, 1),
                "chunk_hit_at_5": hit_at(c_rank, 5),
                "chunk_hit_at_10": hit_at(c_rank, 10),
                "n_e5_candidates": len(e5_rank_by_id),
                "n_bm25_candidates": len(bm25_rank_by_id),
                "n_union_candidates": len(set(e5_rank_by_id) | set(bm25_rank_by_id)),
                "top_results": top,
            }
        )

    def slice_ranks(key: str) -> dict[str, dict]:
        grouped: dict[str, list[int | None]] = defaultdict(list)
        for item, rank in zip(per_question, source_ranks):
            grouped[item[key]].append(rank)
        return {name: metrics_from_ranks(rs) for name, rs in sorted(grouped.items())}

    def three_way(metric_key: str, level: str) -> dict:
        return {
            "e5": e5_summary[level][metric_key],
            "bm25": bm25_summary[level][metric_key],
            "hybrid_rrf": metrics_from_ranks(source_ranks if level == "source_level" else chunk_ranks)[metric_key],
        }

    # Outcomes vs E5 on source ranks (Top-10 evaluation ranks)
    def classify(e5_r, bm_r, hy_r):
        def better(a, b):
            a_val = a if a is not None else 999
            b_val = b if b is not None else 999
            if a_val < b_val:
                return "first"
            if b_val < a_val:
                return "second"
            return "tie"

        vs_e5 = better(hy_r, e5_r)
        vs_bm = better(hy_r, bm_r)
        notes = []
        if hy_r is not None and e5_r is None:
            notes.append("hybrid_recovered_e5_miss")
        if hy_r is None and e5_r is not None:
            notes.append("hybrid_lost_e5_hit")
        if vs_e5 == "first" and bm_r is not None and (e5_r is None or bm_r < e5_r):
            notes.append("bm25_helped_vs_e5")
        if vs_e5 == "second":
            notes.append("worse_than_e5")
        if vs_e5 == "tie":
            notes.append("tie_e5")
        return {
            "vs_e5": vs_e5,
            "vs_bm25": vs_bm,
            "notes": notes,
        }

    comparison_rows = []
    for item in per_question:
        comparison_rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "e5_source_gold_rank": item["e5_source_gold_rank"],
                "bm25_source_gold_rank": item["bm25_source_gold_rank"],
                "hybrid_source_gold_rank": item["hybrid_source_gold_rank"],
                "e5_chunk_gold_rank": item["e5_chunk_gold_rank"],
                "bm25_chunk_gold_rank": item["bm25_chunk_gold_rank"],
                "hybrid_chunk_gold_rank": item["hybrid_chunk_gold_rank"],
                **classify(
                    item["e5_source_gold_rank"],
                    item["bm25_source_gold_rank"],
                    item["hybrid_source_gold_rank"],
                ),
            }
        )

    hybrid_source = metrics_from_ranks(source_ranks)
    hybrid_chunk = metrics_from_ranks(chunk_ranks)

    summary = {
        "method": "hybrid_rrf",
        "rrf_k": args.rrf_k,
        "candidate_depth_per_retriever": args.candidate_depth,
        "fusion_identity": "chunk_id",
        "tie_break": (
            "1) higher RRF score; 2) better E5 rank (missing loses); "
            "3) better BM25 rank (missing loses); 4) lexicographic chunk_id"
        ),
        "what_was_rerun": (
            "Query-only E5 encode + FAISS search at depth 50 on the existing index. "
            "In-memory BM25Okapi over the same 1,443 chunk texts and Step-8 tokenizer. "
            "Corpus E5 embeddings were not regenerated. gita_e5.faiss was not rebuilt."
        ),
        "e5_model": MODEL_NAME,
        "e5_query_prefix": "query: ",
        "e5_faiss": "IndexFlatIP reused",
        "e5_faiss_sha256": faiss_hash_before,
        "bm25": "BM25Okapi + Step-8 tokenize (dotted verse IDs preserved)",
        "n_chunks": 1443,
        "top_k": args.top_k,
        "n_questions": len(questions),
        "frozen_eval_sha256": eval_hash_before,
        "scoring_rule": (
            "Any valid gold source/chunk counts as the first hit. "
            "Not all golds need to appear. Missing from top-10 => 0 for that question."
        ),
        "source_level": hybrid_source,
        "chunk_level": hybrid_chunk,
        "source_level_by_category": slice_ranks("category"),
        "source_level_by_difficulty": slice_ranks("difficulty"),
        "three_way_source_level": {
            "recall@1": three_way("recall@1", "source_level"),
            "recall@5": three_way("recall@5", "source_level"),
            "recall@10": three_way("recall@10", "source_level"),
            "mrr@10": three_way("mrr@10", "source_level"),
        },
        "three_way_chunk_level": {
            "recall@1": three_way("recall@1", "chunk_level"),
            "recall@5": three_way("recall@5", "chunk_level"),
            "recall@10": three_way("recall@10", "chunk_level"),
            "mrr@10": three_way("mrr@10", "chunk_level"),
        },
        "e5_baselines": e5_summary["source_level"],
        "bm25_baselines": bm25_summary["source_level"],
        "per_question_comparison": comparison_rows,
    }

    args.results.write_text(json.dumps(per_question, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    eval_hash_after = sha256_file(args.eval)
    if eval_hash_after != eval_hash_before:
        raise SystemExit("FROZEN EVAL FILE CHANGED DURING EVALUATION.")
    if sha256_file(args.index) != faiss_hash_before:
        raise SystemExit("FAISS INDEX WAS MODIFIED.")
    if sha256_file(args.e5_results) != e5_results_hash_before:
        raise SystemExit("E5 eval results were overwritten.")
    if sha256_file(args.bm25_results) != bm25_results_hash_before:
        raise SystemExit("BM25 eval results were overwritten.")

    print(json.dumps({
        "source_level": hybrid_source,
        "chunk_level": hybrid_chunk,
        "three_way_source_level": summary["three_way_source_level"],
        "three_way_chunk_level": summary["three_way_chunk_level"],
        "source_level_by_category": summary["source_level_by_category"],
        "source_level_by_difficulty": summary["source_level_by_difficulty"],
    }, indent=2))
    print(f"Frozen eval unchanged: {eval_hash_after}")
    print(f"Wrote {args.results}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
