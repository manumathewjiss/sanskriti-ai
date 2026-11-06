"""
Held-out evaluation of four FROZEN retrieval systems.

Does not change E5, BM25, RRF, or the citation-gating regex.
Does not regenerate corpus embeddings.
Does not edit the held-out JSON after seeing scores.

Systems:
  A E5-only (Step 7)
  B BM25-only (Step 8)
  C RRF k=60, depth 50 (Step 9)
  D citation-gated: dotted citation -> C else A (Step 10)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from evaluate_bm25 import tokenize
from evaluate_citation_gated import detect_citations
from evaluate_e5 import (
    TOP_K,
    first_gold_rank,
    hit_at,
    metrics_from_ranks,
    preview,
    query_input,
    sha256_file,
)
from evaluate_hybrid_rrf import CANDIDATE_DEPTH, RRF_K, fuse_ranks

ROOT = Path(__file__).resolve().parent
HOLDOUT_DEFAULT = ROOT / "gita_holdout_frozen.json"
CHUNKS_DEFAULT = ROOT / "gita_chunks.json"
INDEX_DEFAULT = ROOT / "gita_e5.faiss"
META_DEFAULT = ROOT / "gita_e5_metadata.json"
DEV_E5_SUMMARY = ROOT / "gita_e5_eval_summary.json"
DEV_BM25_SUMMARY = ROOT / "gita_bm25_eval_summary.json"
DEV_HYBRID_SUMMARY = ROOT / "gita_hybrid_rrf_eval_summary.json"
DEV_GATED_SUMMARY = ROOT / "gita_citation_gated_eval_summary.json"
RESULTS_DEFAULT = ROOT / "gita_holdout_eval_results.json"
SUMMARY_DEFAULT = ROOT / "gita_holdout_eval_summary.json"

MODEL_NAME = "intfloat/multilingual-e5-large"
EXPECTED_HOLDOUT_HASH = "6fc70801e5863fb5661a727725847fd95380f7113b1230ac341aff5568f07e25"


def gold_sets(q: dict) -> tuple[set[str], set[str]]:
    sources = {g["source_record_id"] for g in q["gold_sources"]}
    chunks = {cid for g in q["gold_sources"] for cid in g["gold_chunk_ids"]}
    return sources, chunks


def ranks_from_top(top: list[dict], gold_sources: set[str], gold_chunks: set[str]) -> tuple[int | None, int | None]:
    return (
        first_gold_rank([t["source_record_id"] for t in top], gold_sources),
        first_gold_rank([t["chunk_id"] for t in top], gold_chunks),
    )


def slice_source_ranks(rows: list[dict], key: str, rank_field: str) -> dict[str, dict]:
    grouped: dict[str, list[int | None]] = defaultdict(list)
    for item in rows:
        grouped[item[key]].append(item[rank_field])
    return {name: metrics_from_ranks(rs) for name, rs in sorted(grouped.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate four frozen retrievers on the held-out set.")
    parser.add_argument("--holdout", type=Path, default=HOLDOUT_DEFAULT)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_DEFAULT)
    parser.add_argument("--index", type=Path, default=INDEX_DEFAULT)
    parser.add_argument("--metadata", type=Path, default=META_DEFAULT)
    parser.add_argument("--results", type=Path, default=RESULTS_DEFAULT)
    parser.add_argument("--summary", type=Path, default=SUMMARY_DEFAULT)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    holdout_hash_before = sha256_file(args.holdout)
    if holdout_hash_before != EXPECTED_HOLDOUT_HASH:
        raise SystemExit(
            f"Held-out hash mismatch.\n expected {EXPECTED_HOLDOUT_HASH}\n got      {holdout_hash_before}"
        )
    faiss_hash_before = sha256_file(args.index)

    payload = json.loads(args.holdout.read_text(encoding="utf-8"))
    questions = payload["questions"]
    if len(questions) != 40:
        raise SystemExit(f"Expected 40 held-out questions, found {len(questions)}")

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    if len(chunks) != 1443:
        raise SystemExit(f"Expected 1443 chunks, found {len(chunks)}")
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))

    print("THE HELD-OUT SET IS FROZEN. Retrieval begins now.")
    print(f"held-out sha256: {holdout_hash_before}")
    print(f"reusing FAISS sha256: {faiss_hash_before}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(str(args.index))
    if index.ntotal != 1443:
        raise SystemExit("FAISS size changed.")

    model = SentenceTransformer(args.model)
    model.eval()
    query_texts = []
    for q in questions:
        qtext = q["question"]
        for g in q["gold_sources"]:
            for cid in g["gold_chunk_ids"]:
                if cid in qtext:
                    raise SystemExit(f"{q['id']} question contains gold chunk id")
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
    e5_scores, e5_rows = index.search(q_emb, CANDIDATE_DEPTH)
    if e5_scores.shape != (len(questions), CANDIDATE_DEPTH):
        raise SystemExit("E5 search did not return depth 50.")

    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    per_question = []
    for i, q in enumerate(questions):
        gold_sources, gold_chunks = gold_sets(q)
        cite_tokens = detect_citations(q["question"])
        citation_detected = len(cite_tokens) > 0

        e5_rank_by_id: dict[str, int] = {}
        e5_top50 = []
        for rank in range(CANDIDATE_DEPTH):
            meta = metadata[int(e5_rows[i, rank])]
            cid = meta["chunk_id"]
            if cid not in e5_rank_by_id:
                e5_rank_by_id[cid] = rank + 1
            rec = {
                "rank": rank + 1,
                "chunk_id": cid,
                "source_record_id": meta["source_record_id"],
                "chapter": meta["chapter"],
                "verse_start": meta["verse_start"],
                "verse_end": meta["verse_end"],
                "section": meta["section"],
                "score": float(e5_scores[i, rank]),
                "text_preview": preview(meta["text"]),
            }
            e5_top50.append(rec)
        e5_top10 = [{k: v for k, v in t.items()} for t in e5_top50[:TOP_K]]

        q_tokens = tokenize(q["question"])
        bm_scores = bm25.get_scores(q_tokens)
        bm_order = sorted(range(len(bm_scores)), key=lambda j: bm_scores[j], reverse=True)
        bm25_rank_by_id: dict[str, int] = {}
        bm_top50 = []
        for rank, idx in enumerate(bm_order[:CANDIDATE_DEPTH], start=1):
            ch = chunks[idx]
            bm25_rank_by_id[ch["chunk_id"]] = rank
            bm_top50.append(
                {
                    "rank": rank,
                    "chunk_id": ch["chunk_id"],
                    "source_record_id": ch["source_record_id"],
                    "chapter": ch["chapter"],
                    "verse_start": ch["verse_start"],
                    "verse_end": ch["verse_end"],
                    "section": ch["section"],
                    "score": float(bm_scores[idx]),
                    "text_preview": preview(ch["text"]),
                }
            )
        bm_top10 = bm_top50[:TOP_K]

        fused = fuse_ranks(e5_rank_by_id, bm25_rank_by_id)
        rrf_top10 = []
        for hy_rank, (cid, rrf, e5_r, bm_r, e5_c, bm_c) in enumerate(fused[:TOP_K], start=1):
            ch = chunk_by_id[cid]
            rrf_top10.append(
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
                    "text_preview": preview(ch["text"]),
                }
            )

        selected = "hybrid_rrf" if citation_detected else "e5"
        gated_top10 = rrf_top10 if selected == "hybrid_rrf" else e5_top10

        e5_s, e5_c = ranks_from_top(e5_top10, gold_sources, gold_chunks)
        bm_s, bm_c = ranks_from_top(bm_top10, gold_sources, gold_chunks)
        rrf_s, rrf_c = ranks_from_top(rrf_top10, gold_sources, gold_chunks)
        gat_s, gat_c = ranks_from_top(gated_top10, gold_sources, gold_chunks)

        per_question.append(
            {
                "id": q["id"],
                "category": q["category"],
                "difficulty": q["difficulty"],
                "question": q["question"],
                "citation_detected": citation_detected,
                "detected_citation_tokens": cite_tokens,
                "selected_retriever": selected,
                "gold_source_record_ids": sorted(gold_sources),
                "gold_chunk_ids": sorted(gold_chunks),
                "e5_source_gold_rank": e5_s,
                "bm25_source_gold_rank": bm_s,
                "rrf_source_gold_rank": rrf_s,
                "gated_source_gold_rank": gat_s,
                "e5_chunk_gold_rank": e5_c,
                "bm25_chunk_gold_rank": bm_c,
                "rrf_chunk_gold_rank": rrf_c,
                "gated_chunk_gold_rank": gat_c,
                "top10": {
                    "e5": e5_top10,
                    "bm25": bm_top10,
                    "rrf": rrf_top10,
                    "gated": gated_top10,
                },
            }
        )

    def pack(field_s: str, field_c: str) -> dict:
        s = [r[field_s] for r in per_question]
        c = [r[field_c] for r in per_question]
        return {"source_level": metrics_from_ranks(s), "chunk_level": metrics_from_ranks(c)}

    systems = {
        "e5": pack("e5_source_gold_rank", "e5_chunk_gold_rank"),
        "bm25": pack("bm25_source_gold_rank", "bm25_chunk_gold_rank"),
        "rrf_hybrid": pack("rrf_source_gold_rank", "rrf_chunk_gold_rank"),
        "citation_gated": pack("gated_source_gold_rank", "gated_chunk_gold_rank"),
    }

    def four(level: str, metric: str) -> dict:
        key = "source_level" if level == "source" else "chunk_level"
        return {name: systems[name][key][metric] for name in systems}

    def cat_table(system_rank_field: str) -> dict:
        return slice_source_ranks(per_question, "category", system_rank_field)

    by_cat = {
        "e5": cat_table("e5_source_gold_rank"),
        "bm25": cat_table("bm25_source_gold_rank"),
        "rrf_hybrid": cat_table("rrf_source_gold_rank"),
        "citation_gated": cat_table("gated_source_gold_rank"),
    }
    by_diff = {
        "e5": slice_source_ranks(per_question, "difficulty", "e5_source_gold_rank"),
        "bm25": slice_source_ranks(per_question, "difficulty", "bm25_source_gold_rank"),
        "rrf_hybrid": slice_source_ranks(per_question, "difficulty", "rrf_source_gold_rank"),
        "citation_gated": slice_source_ranks(per_question, "difficulty", "gated_source_gold_rank"),
    }

    def load_dev(path: Path, source_key: str) -> dict:
        s = json.loads(path.read_text(encoding="utf-8"))
        return s[source_key]

    old_vs_new = {
        "e5": {
            "old_source": load_dev(DEV_E5_SUMMARY, "source_level"),
            "holdout_source": systems["e5"]["source_level"],
        },
        "bm25": {
            "old_source": load_dev(DEV_BM25_SUMMARY, "source_level"),
            "holdout_source": systems["bm25"]["source_level"],
        },
        "rrf_hybrid": {
            "old_source": load_dev(DEV_HYBRID_SUMMARY, "source_level"),
            "holdout_source": systems["rrf_hybrid"]["source_level"],
        },
        "citation_gated": {
            "old_source": load_dev(DEV_GATED_SUMMARY, "source_level"),
            "holdout_source": systems["citation_gated"]["source_level"],
        },
    }

    all_miss_source = [
        r["id"]
        for r in per_question
        if r["e5_source_gold_rank"] is None
        and r["bm25_source_gold_rank"] is None
        and r["rrf_source_gold_rank"] is None
        and r["gated_source_gold_rank"] is None
    ]
    source_ok_chunk_miss = [
        r["id"]
        for r in per_question
        if r["gated_source_gold_rank"] is not None and r["gated_chunk_gold_rank"] is None
    ]

    summary = {
        "eval_set": "held_out_40",
        "heldout_sha256_before": holdout_hash_before,
        "e5_faiss_sha256": faiss_hash_before,
        "rrf_k": RRF_K,
        "candidate_depth": CANDIDATE_DEPTH,
        "citation_regex": r"\b\d{1,2}\.\d{1,3}\b",
        "n_questions": 40,
        "n_routed_to_hybrid": sum(1 for r in per_question if r["selected_retriever"] == "hybrid_rrf"),
        "n_routed_to_e5": sum(1 for r in per_question if r["selected_retriever"] == "e5"),
        "systems": systems,
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
        "source_level_by_category": by_cat,
        "source_level_by_difficulty": by_diff,
        "old_30_vs_holdout_40_source": old_vs_new,
        "all_systems_source_miss": all_miss_source,
        "gated_source_hit_exact_chunk_miss": source_ok_chunk_miss,
        "per_question_ranks": [
            {
                "id": r["id"],
                "category": r["category"],
                "difficulty": r["difficulty"],
                "citation_detected": r["citation_detected"],
                "selected_retriever": r["selected_retriever"],
                "e5_source_gold_rank": r["e5_source_gold_rank"],
                "bm25_source_gold_rank": r["bm25_source_gold_rank"],
                "rrf_source_gold_rank": r["rrf_source_gold_rank"],
                "gated_source_gold_rank": r["gated_source_gold_rank"],
                "e5_chunk_gold_rank": r["e5_chunk_gold_rank"],
                "bm25_chunk_gold_rank": r["bm25_chunk_gold_rank"],
                "rrf_chunk_gold_rank": r["rrf_chunk_gold_rank"],
                "gated_chunk_gold_rank": r["gated_chunk_gold_rank"],
            }
            for r in per_question
        ],
        "architecture_frozen": True,
        "no_tuning_after_holdout": True,
    }

    args.results.write_text(json.dumps(per_question, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    holdout_hash_after = sha256_file(args.holdout)
    if holdout_hash_after != holdout_hash_before:
        raise SystemExit("HELD-OUT FILE CHANGED DURING EVALUATION.")
    if sha256_file(args.index) != faiss_hash_before:
        raise SystemExit("FAISS INDEX WAS MODIFIED.")

    print(json.dumps({
        "four_way_source_level": summary["four_way_source_level"],
        "four_way_chunk_level": summary["four_way_chunk_level"],
        "source_level_by_category": {
            sys: {cat: {k: v[k] for k in ("recall@1", "recall@5", "recall@10", "mrr@10")} for cat, v in cats.items()}
            for sys, cats in by_cat.items()
        },
        "all_systems_source_miss": all_miss_source,
        "gated_source_hit_exact_chunk_miss": source_ok_chunk_miss,
    }, indent=2))
    print(f"Held-out unchanged: {holdout_hash_after}")
    print(f"Wrote {args.results}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
