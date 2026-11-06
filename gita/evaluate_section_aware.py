"""
Step 12 DEVELOPMENT experiment: E5 + same-source section preference.

This is NOT a new held-out test. Both the original 30 and the previous 40
are development/error-analysis sets for this new rule.

Base retriever: existing E5 FAISS (corpus embeddings not regenerated).
No BM25, RRF, citation gating, LLM, or metadata verse lookup.

Intent uses ONLY question text.

Rerank: keep E5 source order (first time a source_record_id appears).
Within a source, put the requested section first, other section second,
preserving E5 order inside each section.

Semantic guardrail: a translation from a source E5 never retrieved in Top 50
cannot jump to rank 1. Only the other section of an already-retrieved source
may move next to that source's first hit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_e5 import (
    TOP_K,
    first_gold_rank,
    hit_at,
    metrics_from_ranks,
    preview,
    query_input,
    sha256_file,
)

ROOT = Path(__file__).resolve().parent
INDEX_DEFAULT = ROOT / "gita_e5.faiss"
META_DEFAULT = ROOT / "gita_e5_metadata.json"
RULES_PATH = ROOT / "gita_section_aware_rules.json"
MODEL_NAME = "intfloat/multilingual-e5-large"
CANDIDATE_DEPTH = 50

EXPECTED_DEV30_HASH = "8da7a3ed170cdf995401e17c7481a61174b61e4e5cc5583c4282c745deaf6c27"
EXPECTED_DEV40_HASH = "6fc70801e5863fb5661a727725847fd95380f7113b1230ac341aff5568f07e25"

# ----- FROZEN BEFORE SCORING. Do not edit after seeing metrics. -----

CITATION_RE = re.compile(r"\b\d{1,2}\.\d{1,3}\b")

PURPORT_PHRASES = (
    "purport",
    "commentary",
    "prabhupada",
)

TRANSLATION_PHRASES = (
    "translation",
    "the verse",
    "this verse",
    "that verse",
)

RULES = {
    "name": "e5_same_source_section_preference",
    "status": "frozen_before_scoring",
    "candidate_depth": CANDIDATE_DEPTH,
    "purport_phrases": list(PURPORT_PHRASES),
    "translation_phrases": list(TRANSLATION_PHRASES),
    "translation_also_if": (
        "question contains a dotted chapter.verse token matching "
        r"\b\d{1,2}\.\d{1,3}\b"
    ),
    "not_used_as_translation_cues": [
        "what does ... say",
        "what does Krishna say",
        "what does the Gita say",
        "according to the Gita",
    ],
    "precedence": (
        "If any purport cue matches, intent=purport even if translation cues "
        "also match (commentary request is more specific)."
    ),
    "neutral": "If neither purport nor translation cues match, keep raw E5 order.",
    "rerank": (
        "Sources keep the order of their first E5 hit. Within a source, the "
        "requested section's chunks (still in E5 order) are placed before the "
        "other section's chunks. Neutral: no change."
    ),
    "safety": (
        "Section preference cannot promote a chunk whose source never appeared "
        "in E5 Top 50. Rank-40 unrelated translations cannot outrank E5 rank 1."
    ),
}


def detect_section_intent(question: str) -> dict:
    q = question.lower()
    purport_hits = [p for p in PURPORT_PHRASES if p in q]
    translation_hits = [p for p in TRANSLATION_PHRASES if p in q]
    citations = CITATION_RE.findall(question)
    if citations:
        translation_hits.append("dotted_citation:" + ",".join(citations))

    if purport_hits:
        intent = "purport"
        cue = purport_hits
    elif translation_hits:
        intent = "translation"
        cue = translation_hits
    else:
        intent = "neutral"
        cue = []
    return {
        "intent": intent,
        "matched_cues": cue,
        "purport_cues": purport_hits,
        "translation_cues": translation_hits,
        "dotted_citations": citations,
    }


def rerank_same_source(candidates: list[dict], intent: str) -> list[dict]:
    if intent == "neutral":
        return list(candidates)
    by_source: dict[str, list[dict]] = defaultdict(list)
    source_order: list[str] = []
    for c in candidates:
        sid = c["source_record_id"]
        if sid not in by_source:
            source_order.append(sid)
        by_source[sid].append(c)

    out: list[dict] = []
    for sid in source_order:
        group = by_source[sid]
        trans = [c for c in group if c["section"] == "translation"]
        purp = [c for c in group if c["section"] == "purport"]
        other = [c for c in group if c["section"] not in ("translation", "purport")]
        if intent == "translation":
            ordered = trans + purp + other
        else:
            ordered = purp + trans + other
        out.extend(ordered)
    return out


def gold_sets(q: dict) -> tuple[set[str], set[str]]:
    sources = {g["source_record_id"] for g in q["gold_sources"]}
    chunks = {cid for g in q["gold_sources"] for cid in g["gold_chunk_ids"]}
    return sources, chunks


def ranks_from_top(top: list[dict], gold_sources: set[str], gold_chunks: set[str]):
    return (
        first_gold_rank([t["source_record_id"] for t in top], gold_sources),
        first_gold_rank([t["chunk_id"] for t in top], gold_chunks),
    )


def pack_metrics(rows: list[dict], s_field: str, c_field: str) -> dict:
    return {
        "source_level": metrics_from_ranks([r[s_field] for r in rows]),
        "chunk_level": metrics_from_ranks([r[c_field] for r in rows]),
    }


def by_intent_metrics(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["detected_section_intent"]].append(r)
    out = {}
    for intent, items in sorted(grouped.items()):
        out[intent] = {
            "n": len(items),
            "e5": pack_metrics(items, "e5_source_gold_rank", "e5_chunk_gold_rank"),
            "section_aware": pack_metrics(
                items, "section_aware_source_gold_rank", "section_aware_chunk_gold_rank"
            ),
        }
    return out


def evaluate_set(
    label: str,
    questions: list[dict],
    e5_scores: np.ndarray,
    e5_rows: np.ndarray,
    metadata: list[dict],
    offset: int,
) -> list[dict]:
    per = []
    for i, q in enumerate(questions):
        row_i = offset + i
        gold_sources, gold_chunks = gold_sets(q)
        detected = detect_section_intent(q["question"])
        intent = detected["intent"]

        candidates = []
        for rank in range(CANDIDATE_DEPTH):
            meta = metadata[int(e5_rows[row_i, rank])]
            candidates.append(
                {
                    "e5_rank": rank + 1,
                    "chunk_id": meta["chunk_id"],
                    "source_record_id": meta["source_record_id"],
                    "chapter": meta["chapter"],
                    "verse_start": meta["verse_start"],
                    "verse_end": meta["verse_end"],
                    "section": meta["section"],
                    "score": float(e5_scores[row_i, rank]),
                    "text_preview": preview(meta["text"]),
                }
            )
        reranked = rerank_same_source(candidates, intent)
        e5_top = []
        for t in candidates[:TOP_K]:
            item = dict(t)
            item["rank"] = t["e5_rank"]
            e5_top.append(item)
        sa_top = []
        for new_rank, t in enumerate(reranked[:TOP_K], start=1):
            item = dict(t)
            item["rank"] = new_rank
            item["original_e5_rank"] = t["e5_rank"]
            sa_top.append(item)

        e5_s, e5_c = ranks_from_top(e5_top, gold_sources, gold_chunks)
        sa_s, sa_c = ranks_from_top(sa_top, gold_sources, gold_chunks)
        per.append(
            {
                "id": q["id"],
                "category": q["category"],
                "difficulty": q["difficulty"],
                "question": q["question"],
                "detected_section_intent": intent,
                "matched_cues": detected["matched_cues"],
                "gold_source_record_ids": sorted(gold_sources),
                "gold_chunk_ids": sorted(gold_chunks),
                "e5_source_gold_rank": e5_s,
                "section_aware_source_gold_rank": sa_s,
                "e5_chunk_gold_rank": e5_c,
                "section_aware_chunk_gold_rank": sa_c,
                "e5_top10": e5_top,
                "section_aware_top10": sa_top,
            }
        )
    return per


def summarize(label: str, rows: list[dict], eval_hash: str) -> dict:
    e5 = pack_metrics(rows, "e5_source_gold_rank", "e5_chunk_gold_rank")
    sa = pack_metrics(rows, "section_aware_source_gold_rank", "section_aware_chunk_gold_rank")
    intent_counts = defaultdict(int)
    for r in rows:
        intent_counts[r["detected_section_intent"]] += 1

    def rk(x):
        return 999 if x is None else x

    exact_improved = []
    exact_regressed = []
    source_regressed = []
    section_fixed = []
    for r in rows:
        if rk(r["section_aware_chunk_gold_rank"]) < rk(r["e5_chunk_gold_rank"]):
            exact_improved.append(r["id"])
        if rk(r["section_aware_chunk_gold_rank"]) > rk(r["e5_chunk_gold_rank"]):
            exact_regressed.append(r["id"])
        if rk(r["section_aware_source_gold_rank"]) > rk(r["e5_source_gold_rank"]):
            source_regressed.append(r["id"])
        e5_chunk = r["e5_chunk_gold_rank"]
        sa_chunk = r["section_aware_chunk_gold_rank"]
        if r["e5_source_gold_rank"] is not None and e5_chunk is None and sa_chunk is not None:
            section_fixed.append(r["id"])

    return {
        "set_label": label,
        "scientific_status": "development_or_error_analysis_not_untouched_holdout",
        "n_questions": len(rows),
        "eval_sha256": eval_hash,
        "intent_counts": dict(intent_counts),
        "e5": e5,
        "section_aware": sa,
        "exact_chunk_improved_ids": exact_improved,
        "exact_chunk_regressed_ids": exact_regressed,
        "source_regressed_ids": source_regressed,
        "source_ok_exact_miss_then_fixed_ids": section_fixed,
        "by_intent": by_intent_metrics(rows),
        "per_question_ranks": [
            {
                "id": r["id"],
                "intent": r["detected_section_intent"],
                "cues": r["matched_cues"],
                "e5_source": r["e5_source_gold_rank"],
                "sa_source": r["section_aware_source_gold_rank"],
                "e5_chunk": r["e5_chunk_gold_rank"],
                "sa_chunk": r["section_aware_chunk_gold_rank"],
            }
            for r in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E5 + frozen section-aware same-source rerank.")
    parser.add_argument("--index", type=Path, default=INDEX_DEFAULT)
    parser.add_argument("--metadata", type=Path, default=META_DEFAULT)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    RULES_PATH.write_text(json.dumps(RULES, indent=2) + "\n", encoding="utf-8")
    print("FROZEN SECTION-AWARE RULES")
    print(json.dumps(RULES, indent=2))

    sets = [
        {
            "label": "development_30",
            "path": ROOT / "gita_eval_frozen.json",
            "expected_hash": EXPECTED_DEV30_HASH,
            "results": ROOT / "gita_section_aware_dev30_results.json",
            "summary": ROOT / "gita_section_aware_dev30_summary.json",
        },
        {
            "label": "development_40_former_holdout",
            "path": ROOT / "gita_holdout_frozen.json",
            "expected_hash": EXPECTED_DEV40_HASH,
            "results": ROOT / "gita_section_aware_dev40_results.json",
            "summary": ROOT / "gita_section_aware_dev40_summary.json",
        },
    ]

    hashes_before = {}
    all_questions = []
    set_slices = []
    for spec in sets:
        h = sha256_file(spec["path"])
        if h != spec["expected_hash"]:
            raise SystemExit(f"{spec['path'].name} hash mismatch: {h}")
        hashes_before[spec["label"]] = h
        qs = json.loads(spec["path"].read_text(encoding="utf-8"))["questions"]
        set_slices.append((spec, qs, len(all_questions), len(qs)))
        all_questions.extend(qs)

    faiss_hash_before = sha256_file(args.index)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if len(metadata) != 1443:
        raise SystemExit(f"Expected 1443 metadata rows, found {len(metadata)}")

    print("Intent detector examples (question text only):")
    for sample in (
        "What does the commentary explain about controlling the mind?",
        "What does Bhagavad Gita 11.3 say?",
        "After hearing Krishna describe Himself, what does Arjuna say he still wants to see?",
        "Does the commentary on 15.7 say the living being is an individual only while trapped?",
        "How can the mind become a person's friend or enemy?",
    ):
        print(f"  {sample!r}")
        print(f"    -> {detect_section_intent(sample)}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(str(args.index))
    if index.ntotal != 1443:
        raise SystemExit("FAISS size is not 1443.")
    model = SentenceTransformer(args.model)
    model.eval()

    for q in all_questions:
        for g in q["gold_sources"]:
            for cid in g["gold_chunk_ids"]:
                if cid in q["question"]:
                    raise SystemExit(f"{q['id']} question contains gold chunk id")

    query_texts = [query_input(q["question"]) for q in all_questions]
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

    for spec, qs, offset, n in set_slices:
        rows = evaluate_set(spec["label"], qs, e5_scores, e5_rows, metadata, offset)
        summary = summarize(spec["label"], rows, hashes_before[spec["label"]])
        spec["results"].write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        spec["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n=== {spec['label']} ===")
        print(json.dumps({
            "intent_counts": summary["intent_counts"],
            "e5_source": summary["e5"]["source_level"],
            "sa_source": summary["section_aware"]["source_level"],
            "e5_exact": summary["e5"]["chunk_level"],
            "sa_exact": summary["section_aware"]["chunk_level"],
            "exact_improved": summary["exact_chunk_improved_ids"],
            "exact_regressed": summary["exact_chunk_regressed_ids"],
            "source_regressed": summary["source_regressed_ids"],
            "section_fixed": summary["source_ok_exact_miss_then_fixed_ids"],
        }, indent=2))
        print(f"Wrote {spec['results']}")
        print(f"Wrote {spec['summary']}")

    if sha256_file(args.index) != faiss_hash_before:
        raise SystemExit("FAISS modified.")
    for spec in sets:
        if sha256_file(spec["path"]) != hashes_before[spec["label"]]:
            raise SystemExit(f"{spec['path'].name} changed during evaluation.")
    print("Eval JSON hashes unchanged. FAISS unchanged.")
    print(f"Wrote {RULES_PATH}")


if __name__ == "__main__":
    main()
