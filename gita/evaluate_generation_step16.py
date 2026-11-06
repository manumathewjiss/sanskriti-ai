"""
Step 16: run frozen Step 14 RAG on the unseen generation eval set.

Does not modify the frozen questions. Does not use the Step 15 prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

OVERLAP_WORDS = 40
ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "gita_generation_eval_frozen.json"
VALID = ROOT / "gita_generation_eval_validation.json"
INDEX = ROOT / "gita_e5.faiss"
META = ROOT / "gita_e5_metadata.json"
PROMPT_PATH = ROOT / "gita_rag_prompt.txt"
OUT_RESULTS = ROOT / "gita_generation_step16_results.json"
OUT_REPORT = ROOT / "gita_generation_step16_report.json"
CHUNKS = ROOT / "gita_chunks.json"

E5_MODEL = "intfloat/multilingual-e5-large"
TOP_K = 5
MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
TEMPERATURE = 0
NUM_PREDICT = 512
NUM_CTX = 8192
SEED = 0

CITATION_RE = re.compile(
    r"Bhagavad[\s-]*Gita\s+(\d{1,2}\.\d{1,3}(?:\s*[–-]\s*\d{1,3})?)",
    re.I,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query_input(question: str) -> str:
    return "query: " + question


def first_gold_rank(retrieved_ids: list[str], gold_ids: set[str]) -> int | None:
    for i, item in enumerate(retrieved_ids, start=1):
        if item in gold_ids:
            return i
    return None


def load_prompt_parts() -> tuple[str, str]:
    raw = PROMPT_PATH.read_text(encoding="utf-8")
    marker = "\nQuestion:\n"
    system, rest = raw.split(marker, 1)
    return system.strip(), "Question:\n" + rest


SYSTEM_PROMPT, USER_TEMPLATE = load_prompt_parts()


def citation_label(hit: dict) -> str:
    vs, ve = hit["verse_start"], hit["verse_end"]
    if vs == ve:
        return f"Bhagavad Gita {hit['chapter']}.{vs}"
    return f"Bhagavad Gita {hit['chapter']}.{vs}-{ve}"


def section_label(hit: dict) -> str:
    if hit["section"] == "translation":
        return "Translation"
    if hit["section"] == "purport":
        n = hit.get("n_parts") or 1
        part = hit.get("part") or 1
        if n > 1:
            return f"Purport, part {part} of {n}"
        return "Purport"
    return str(hit["section"])


def body_text(hit: dict) -> str:
    text = hit["text"]
    if "\n\n" in text:
        return text.split("\n\n", 1)[1]
    return text


def strip_known_overlap(prev_body: str, next_body: str, overlap_words: int = OVERLAP_WORDS) -> str:
    prev_w = prev_body.split()
    next_w = next_body.split()
    if overlap_words <= 0 or len(prev_w) < overlap_words or len(next_w) < overlap_words:
        return next_body
    if prev_w[-overlap_words:] == next_w[:overlap_words]:
        return " ".join(next_w[overlap_words:])
    return next_body


def build_context(hits: list[dict]) -> tuple[str, list[dict]]:
    blocks = []
    last_purport: dict | None = None
    display_hits = []
    for i, hit in enumerate(hits, start=1):
        body = body_text(hit)
        overlap_removed = False
        if (
            last_purport
            and hit["section"] == "purport"
            and hit["source_record_id"] == last_purport["source_record_id"]
            and hit.get("part") == last_purport.get("part", 0) + 1
        ):
            new_body = strip_known_overlap(last_purport["body"], body)
            if new_body != body:
                overlap_removed = True
                body = new_body
        if hit["section"] == "purport":
            last_purport = {"source_record_id": hit["source_record_id"], "part": hit.get("part"), "body": body_text(hit)}
        else:
            last_purport = None
        block = (
            f"[EVIDENCE {i}]\n"
            f"Citation: {citation_label(hit)}\n"
            f"Section: {section_label(hit)}\n"
            f"Chunk ID: {hit['chunk_id']}\n"
            f"Text:\n{body.strip()}\n"
        )
        blocks.append(block)
        row = dict(hit)
        row["overlap_removed_from_context"] = overlap_removed
        display_hits.append(row)
    return "\n".join(blocks).strip(), display_hits


def normalize_cite(token: str) -> str:
    return token.replace("–", "-").replace(" ", "")


def extract_citations(answer: str) -> list[str]:
    return [normalize_cite(m) for m in CITATION_RE.findall(answer)]


def validate_citations(answer: str, hits: list[dict]) -> dict:
    retrieved = {h["source_record_id"] for h in hits}
    cited = extract_citations(answer)
    unsupported = [c for c in cited if c not in retrieved]
    if not cited:
        status = "MISSING"
    elif unsupported and any(c in retrieved for c in cited):
        status = "PARTIALLY CORRECT"
    elif unsupported:
        status = "UNSUPPORTED"
    else:
        status = "CORRECT"
    return {
        "cited_source_ids": cited,
        "retrieved_source_ids": sorted(retrieved),
        "unsupported_citations": unsupported,
        "cited_any": len(cited) > 0,
        "citation_status": status,
    }


def ollama_chat(question: str, context: str) -> dict:
    user = USER_TEMPLATE.replace("{question}", question).replace("{context}", context)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
            "seed": SEED,
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return {
        "answer": ((data.get("message") or {}).get("content") or "").strip(),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
    }


def hit_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    expected = json.loads(VALID.read_text(encoding="utf-8"))["sha256"]
    frozen_hash_before = sha256_file(EVAL)
    print(f"FROZEN EVAL SHA-256 (before retrieval): {frozen_hash_before}")
    if frozen_hash_before != expected:
        raise SystemExit("Frozen eval hash does not match gita_generation_eval_validation.json")

    payload = json.loads(EVAL.read_text(encoding="utf-8"))
    questions = payload["questions"]
    if len(questions) != 40:
        raise SystemExit("Expected 40 questions")
    metadata = json.loads(META.read_text(encoding="utf-8"))
    chunks_by_id = {c["chunk_id"]: c for c in json.loads(CHUNKS.read_text(encoding="utf-8"))}
    if len(metadata) != 1443:
        raise SystemExit(f"Expected 1443 metadata rows, found {len(metadata)}")

    import faiss
    import torch
    torch.set_num_threads(1)
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(str(INDEX))
    if index.ntotal != 1443:
        raise SystemExit(f"FAISS ntotal {index.ntotal} != 1443")

    model = SentenceTransformer(E5_MODEL)
    model.eval()
    query_texts = [query_input(q["question"]) for q in questions]
    vecs = []
    with torch.inference_mode():
        for qt in query_texts:
            v = model.encode(
                [qt],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device="cpu",
            )
            vecs.append(v[0])
    q_emb = np.asarray(vecs, dtype=np.float32)
    scores, rows = index.search(q_emb, TOP_K)

    results = []
    source_ranks = []
    chunk_ranks = []
    for i, q in enumerate(questions):
        gold_sources = set(q["gold_source_ids"])
        gold_chunks = set(q["gold_chunk_ids"])
        hits = []
        for rank in range(TOP_K):
            meta = metadata[int(rows[i, rank])]
            full = chunks_by_id[meta["chunk_id"]]
            hits.append(
                {
                    "rank": rank + 1,
                    "chunk_id": meta["chunk_id"],
                    "source_record_id": meta["source_record_id"],
                    "chapter": meta["chapter"],
                    "verse_start": meta["verse_start"],
                    "verse_end": meta["verse_end"],
                    "section": meta["section"],
                    "part": full.get("part"),
                    "n_parts": full.get("n_parts"),
                    "page_start": full.get("page_start"),
                    "page_end": full.get("page_end"),
                    "text": full["text"],
                    "score": float(scores[i, rank]),
                }
            )
        s_rank = first_gold_rank([h["source_record_id"] for h in hits], gold_sources)
        c_rank = first_gold_rank([h["chunk_id"] for h in hits], gold_chunks)
        source_ranks.append(s_rank)
        chunk_ranks.append(c_rank)
        context, hits_ctx = build_context(hits)
        n_blocks = context.count("[EVIDENCE ")
        gen = ollama_chat(q["question"], context)
        cites = validate_citations(gen["answer"], hits)
        print(q["id"], "src", s_rank, "chunk", c_rank, "tokens", gen["prompt_eval_count"], "blocks", n_blocks)
        results.append(
            {
                "id": q["id"],
                "question": q["question"],
                "category": q["category"],
                "difficulty": q["difficulty"],
                "chapter": q["chapter"],
                "gold_source_ids": q["gold_source_ids"],
                "gold_chunk_ids": q["gold_chunk_ids"],
                "gold_sections": q["gold_sections"],
                "reference_answer": q["reference_answer"],
                "top5_chunk_ids": [h["chunk_id"] for h in hits_ctx],
                "top5_source_ids": [h["source_record_id"] for h in hits_ctx],
                "retrieval_top5": hits_ctx,
                "gold_source_rank": s_rank,
                "gold_chunk_rank": c_rank,
                "gold_source_in_top5": s_rank is not None,
                "gold_chunk_in_top5": c_rank is not None,
                "n_evidence_blocks_supplied": n_blocks,
                "all_top5_supplied": n_blocks == len(hits),
                "input_truncated": n_blocks < len(hits),
                "prompt_eval_count": gen["prompt_eval_count"],
                "generated_answer": gen["answer"],
                "citation_validation": cites,
                "citation_status": cites["citation_status"],
                "context_sent_to_llm": context,
                "generation": {
                    "model": MODEL,
                    "prompt_file": PROMPT_PATH.name,
                    "temperature": TEMPERATURE,
                    "num_ctx": NUM_CTX,
                    "seed": SEED,
                    "num_predict": NUM_PREDICT,
                },
                "manual_review": {
                    "correctness": None,
                    "grounding": None,
                    "citation": None,
                    "section_attribution": None,
                    "retrieval_failure": None,
                    "generation_failure": None,
                    "context_limit_failure": None,
                    "notes": "Fill after generation. Not used as gold during retrieval/generation.",
                },
            }
        )

    frozen_hash_after = sha256_file(EVAL)
    if frozen_hash_after != frozen_hash_before:
        raise SystemExit("Frozen eval file changed during evaluation")

    n = len(results)
    retrieval = {
        "source_recall@1": sum(hit_at(r, 1) for r in source_ranks) / n,
        "source_recall@5": sum(hit_at(r, 5) for r in source_ranks) / n,
        "exact_chunk_recall@1": sum(hit_at(r, 1) for r in chunk_ranks) / n,
        "exact_chunk_recall@5": sum(hit_at(r, 5) for r in chunk_ranks) / n,
        "source_misses": [x["id"] for x in results if not x["gold_source_in_top5"]],
        "chunk_misses": [x["id"] for x in results if not x["gold_chunk_in_top5"]],
    }
    by_cat = defaultdict(list)
    by_diff = defaultdict(list)
    for x in results:
        by_cat[x["category"]].append(x)
        by_diff[x["difficulty"]].append(x)

    def rec5(items, key):
        return sum(1 for x in items if x[key]) / len(items)

    cat_retrieval = {
        c: {
            "n": len(items),
            "source_recall@5": rec5(items, "gold_source_in_top5"),
            "exact_chunk_recall@5": rec5(items, "gold_chunk_in_top5"),
        }
        for c, items in by_cat.items()
    }
    diff_retrieval = {
        d: {
            "n": len(items),
            "source_recall@5": rec5(items, "gold_source_in_top5"),
            "exact_chunk_recall@5": rec5(items, "gold_chunk_in_top5"),
        }
        for d, items in by_diff.items()
    }

    out = {
        "step": 16,
        "label": "FRESH FROZEN GENERATION EVALUATION",
        "not_development_smoke_data": True,
        "frozen_eval_sha256_before": frozen_hash_before,
        "frozen_eval_sha256_after": frozen_hash_after,
        "hash_unchanged": True,
        "retriever": "plain_e5",
        "e5_model": E5_MODEL,
        "n_chunks": 1443,
        "top_k": TOP_K,
        "generator": MODEL,
        "prompt": "step14_gita_rag_prompt.txt",
        "generation_settings": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "seed": SEED,
            "num_predict": NUM_PREDICT,
        },
        "all_top5_supplied": all(x["all_top5_supplied"] for x in results),
        "any_truncation": any(x["input_truncated"] for x in results),
        "retrieval": retrieval,
        "retrieval_by_category": cat_retrieval,
        "retrieval_by_difficulty": diff_retrieval,
        "citation_status_counts": {
            s: sum(1 for x in results if x["citation_status"] == s)
            for s in ("CORRECT", "PARTIALLY CORRECT", "MISSING", "UNSUPPORTED")
        },
        "results": results,
    }
    OUT_RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_REPORT.write_text(
        json.dumps(
            {
                "step": 16,
                "frozen_eval_sha256": frozen_hash_before,
                "hash_unchanged_after_evaluation": True,
                "retrieval": retrieval,
                "retrieval_by_category": cat_retrieval,
                "retrieval_by_difficulty": diff_retrieval,
                "all_top5_supplied": out["all_top5_supplied"],
                "any_truncation": out["any_truncation"],
                "citation_status_counts": out["citation_status_counts"],
                "manual_generation_labels": "pending",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(retrieval, indent=2))
    print("citation", out["citation_status_counts"])
    print(f"Wrote {OUT_RESULTS}")
    print(f"SHA-256 after: {frozen_hash_after}")


if __name__ == "__main__":
    main()
