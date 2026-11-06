"""
Evaluate the E5+FAISS Gita index against the FROZEN 30-question set.

The retriever receives ONLY the question text.
Gold labels are used AFTER search, never as query input.

Scoring rule (fixed before looking at scores):
  Multiple golds: ANY matching gold counts as the first hit.
  We do NOT require every gold source to appear.
  Rank is 1-based. Missing from top-10 => rank None, MRR contribution 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EVAL_DEFAULT = ROOT / "gita_eval_frozen.json"
INDEX_DEFAULT = ROOT / "gita_e5.faiss"
META_DEFAULT = ROOT / "gita_e5_metadata.json"
RESULTS_DEFAULT = ROOT / "gita_e5_eval_results.json"
SUMMARY_DEFAULT = ROOT / "gita_e5_eval_summary.json"
MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def query_input(question: str) -> str:
    return "query: " + question


def first_gold_rank(retrieved_ids: list[str], gold_ids: set[str]) -> int | None:
    for i, item in enumerate(retrieved_ids, start=1):
        if item in gold_ids:
            return i
    return None


def hit_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def mrr_from_rank(rank: int | None, k: int = TOP_K) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / rank


def metrics_from_ranks(ranks: list[int | None], k_values=(1, 5, 10)) -> dict:
    n = len(ranks)
    out = {f"recall@{k}": sum(hit_at(r, k) for r in ranks) / n for k in k_values}
    out["mrr@10"] = sum(mrr_from_rank(r) for r in ranks) / n
    out["n"] = n
    return out


def preview(text: str, n: int = 180) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen Gita questions on the E5 FAISS index.")
    parser.add_argument("--eval", type=Path, default=EVAL_DEFAULT)
    parser.add_argument("--index", type=Path, default=INDEX_DEFAULT)
    parser.add_argument("--metadata", type=Path, default=META_DEFAULT)
    parser.add_argument("--results", type=Path, default=RESULTS_DEFAULT)
    parser.add_argument("--summary", type=Path, default=SUMMARY_DEFAULT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    eval_hash_before = sha256_file(args.eval)
    payload = json.loads(args.eval.read_text(encoding="utf-8"))
    questions = payload["questions"]
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    index = faiss.read_index(str(args.index))

    if index.ntotal != len(metadata):
        raise SystemExit(f"FAISS ntotal={index.ntotal} != metadata rows={len(metadata)}")
    chunk_ids = [m["chunk_id"] for m in metadata]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise SystemExit("Duplicate chunk_id in metadata.")

    print(f"Frozen eval sha256: {eval_hash_before}")
    print(f"Questions: {len(questions)}  FAISS vectors: {index.ntotal}  dim: {index.d}")

    model = SentenceTransformer(args.model)
    model.eval()
    query_texts = []
    for q in questions:
        # Gold fields are read later for scoring only. Query string is the question.
        query_texts.append(query_input(q["question"]))
        qtext = q["question"]
        for g in q["gold_sources"]:
            for cid in g["gold_chunk_ids"]:
                if cid in qtext:
                    raise SystemExit(f"{q['id']} question accidentally contains gold chunk id")

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
    qnorms = np.linalg.norm(q_emb, axis=1)
    if not np.allclose(qnorms, 1.0, atol=1e-3):
        raise SystemExit(f"Query embeddings not unit-normalized: min={qnorms.min()} max={qnorms.max()}")
    if q_emb.shape[1] != index.d:
        raise SystemExit("Query dim does not match index dim.")

    scores, rows = index.search(q_emb, args.top_k)
    if scores.shape != (len(questions), args.top_k):
        raise SystemExit("Search did not return Top-K for every question.")

    per_question = []
    source_ranks: list[int | None] = []
    chunk_ranks: list[int | None] = []

    for i, q in enumerate(questions):
        gold_sources = {g["source_record_id"] for g in q["gold_sources"]}
        gold_chunks = {cid for g in q["gold_sources"] for cid in g["gold_chunk_ids"]}
        top = []
        retrieved_chunk_ids = []
        retrieved_source_ids = []
        for rank in range(args.top_k):
            row = int(rows[i, rank])
            meta = metadata[row]
            retrieved_chunk_ids.append(meta["chunk_id"])
            retrieved_source_ids.append(meta["source_record_id"])
            top.append(
                {
                    "rank": rank + 1,
                    "score": float(scores[i, rank]),
                    "chunk_id": meta["chunk_id"],
                    "source_record_id": meta["source_record_id"],
                    "chapter": meta["chapter"],
                    "verse_start": meta["verse_start"],
                    "verse_end": meta["verse_end"],
                    "section": meta["section"],
                    "text_preview": preview(meta["text"]),
                }
            )
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

    summary = {
        "model": args.model,
        "faiss_index": "IndexFlatIP",
        "n_vectors": index.ntotal,
        "embedding_dim": index.d,
        "top_k": args.top_k,
        "n_questions": len(questions),
        "frozen_eval_sha256": eval_hash_before,
        "scoring_rule": (
            "Any valid gold source/chunk counts as the first hit. "
            "Not all golds need to appear. Missing from top-10 => 0 for that question."
        ),
        "source_level": metrics_from_ranks(source_ranks),
        "chunk_level": metrics_from_ranks(chunk_ranks),
        "source_level_by_category": slice_ranks("category"),
        "source_level_by_difficulty": slice_ranks("difficulty"),
        "per_question_ranks": [
            {
                "id": item["id"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "source_gold_rank": item["source_gold_rank"],
                "chunk_gold_rank": item["chunk_gold_rank"],
            }
            for item in per_question
        ],
    }
    args.results.write_text(json.dumps(per_question, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    eval_hash_after = sha256_file(args.eval)
    if eval_hash_after != eval_hash_before:
        raise SystemExit("FROZEN EVAL FILE CHANGED DURING EVALUATION.")
    print(json.dumps({k: summary[k] for k in (
        "source_level", "chunk_level", "source_level_by_category", "source_level_by_difficulty"
    )}, indent=2))
    print(f"Frozen eval unchanged: {eval_hash_after}")
    print(f"Wrote {args.results}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
