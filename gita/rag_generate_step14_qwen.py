"""
Step 14 generation experiment: Ollama qwen2.5:3b on FROZEN Step 13 evidence.

Does not rerun E5. Does not change FAISS, BM25, RRF, gating, or the grounding prompt intent.
"""

from __future__ import annotations

import json
import urllib.request
import re
from pathlib import Path

# Copied from rag_generate.py so this script does not import E5/FAISS/FLAN.
OVERLAP_WORDS = 40
PROMPT_PATH = Path(__file__).resolve().parent / "gita_rag_prompt.txt"
CITATION_RE = re.compile(
    r"Bhagavad[\s-]*Gita\s+(\d{1,2}\.\d{1,3}(?:\s*[–-]\s*\d{1,3})?)",
    re.I,
)


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
    token = token.replace("–", "-").replace(" ", "")
    return token


def extract_citations(answer: str) -> list[str]:
    found = []
    for m in CITATION_RE.findall(answer):
        found.append(normalize_cite(m))
    return found


def validate_citations(answer: str, hits: list[dict]) -> dict:
    retrieved = {h["source_record_id"] for h in hits}
    cited = extract_citations(answer)
    unsupported = [c for c in cited if c not in retrieved]
    return {
        "cited_source_ids": cited,
        "retrieved_source_ids": sorted(retrieved),
        "unsupported_citations": unsupported,
        "all_citations_supported": len(unsupported) == 0,
        "cited_any": len(cited) > 0,
    }

ROOT = Path(__file__).resolve().parent
STEP13 = ROOT / "gita_generation_smoke_results.json"
OUT_RESULTS = ROOT / "gita_generation_step14_qwen_results.json"
OUT_REPORT = ROOT / "gita_generation_step14_qwen_report.json"
CONFIG = ROOT / "gita_generation_model_config.json"

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
TEMPERATURE = 0
NUM_PREDICT = 512
NUM_CTX = 8192
SEED = 0

SYSTEM_PROMPT = """You are a Bhagavad Gita question-answering assistant.

Answer the user's question using ONLY the supplied Bhagavad Gita evidence.
Do not use outside knowledge.
If the supplied evidence does not contain enough information to answer the question, say that the retrieved evidence is insufficient.
Do not invent verses, quotations, explanations, or citations.
Answer in clear English.
Cite the Bhagavad Gita verse/source supporting the answer.
When relevant, distinguish between the verse translation and the purport/commentary.
Do not claim that something appears in the verse itself if it appears only in the purport."""


def user_prompt(question: str, context: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        "Answer the question using only the evidence above."
    )


def citation_status(answer: str, hits: list[dict]) -> str:
    v = validate_citations(answer, hits)
    if not v["cited_any"]:
        return "MISSING"
    if v["unsupported_citations"] and v["cited_source_ids"] and not all(
        c in v["retrieved_source_ids"] for c in v["cited_source_ids"]
    ):
        supported = [c for c in v["cited_source_ids"] if c in v["retrieved_source_ids"]]
        if supported and v["unsupported_citations"]:
            return "PARTIALLY CORRECT"
        return "UNSUPPORTED"
    if v["unsupported_citations"]:
        return "UNSUPPORTED"
    return "CORRECT"


def ollama_chat(question: str, context: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(question, context)},
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
    answer = (data.get("message") or {}).get("content", "").strip()
    prompt_tokens = data.get("prompt_eval_count")
    return {
        "answer": answer,
        "prompt_eval_count": prompt_tokens,
        "eval_count": data.get("eval_count"),
        "done_reason": data.get("done_reason"),
        "raw_keys": sorted(data.keys()),
    }


def hits_for_context(saved_hits: list[dict]) -> list[dict]:
    """Keep fields needed by build_context; drop Step 13-only flags."""
    out = []
    for h in saved_hits:
        out.append(
            {
                "rank": h["rank"],
                "chunk_id": h["chunk_id"],
                "source_record_id": h["source_record_id"],
                "chapter": h["chapter"],
                "verse_start": h["verse_start"],
                "verse_end": h["verse_end"],
                "section": h["section"],
                "part": h.get("part"),
                "n_parts": h.get("n_parts"),
                "page_start": h.get("page_start"),
                "page_end": h.get("page_end"),
                "text": h["text"],
                "score": h.get("score"),
            }
        )
    return out


def generate_one(question: str, saved_hits: list[dict]) -> dict:
    hits = hits_for_context(saved_hits)
    context, hits_ctx = build_context(hits)
    n_blocks = context.count("[EVIDENCE ")
    truncated = n_blocks < len(hits)
    gen = ollama_chat(question, context)
    cites = validate_citations(gen["answer"], hits)
    return {
        "context": context,
        "hits_ctx": hits_ctx,
        "n_evidence_blocks_supplied": n_blocks,
        "all_top5_supplied": n_blocks == len(hits),
        "input_truncated": truncated,
        "prompt_eval_count": gen["prompt_eval_count"],
        "eval_count": gen["eval_count"],
        "answer": gen["answer"],
        "citation_validation": cites,
        "citation_status": citation_status(gen["answer"], hits),
        "generation": {
            "model": MODEL,
            "provider": "ollama",
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
            "seed": SEED,
            "top_p": None,
        },
    }


def main() -> None:
    prompt_on_disk = PROMPT_PATH.read_text(encoding="utf-8")
    step13 = json.loads(STEP13.read_text(encoding="utf-8"))
    normal_out = []
    for row in step13["normal"]:
        ids13 = [h["chunk_id"] for h in row["retrieval_top5"]]
        if len(ids13) != 5:
            raise SystemExit(f"{row['id']} Step 13 Top 5 length is {len(ids13)}")
        packed = generate_one(row["question"], row["retrieval_top5"])
        ids_now = [h["chunk_id"] for h in packed["hits_ctx"]]
        if ids_now != ids13:
            raise SystemExit(f"{row['id']} Top 5 mismatch\n step13={ids13}\n now={ids_now}")
        normal_out.append(
            {
                "id": row["id"],
                "source_eval_id": row["source_eval_id"],
                "category": row["category"],
                "question": row["question"],
                "case_type": "normal_smoke",
                "gold_chunk_ids": row["gold_chunk_ids"],
                "gold_source_ids": row["gold_source_ids"],
                "gold_chunk_in_top5": row["gold_chunk_in_top5"],
                "gold_source_in_top5": row["gold_source_in_top5"],
                "top5_chunk_ids": ids13,
                "retrieval_top5": packed["hits_ctx"],
                "all_top5_supplied_to_qwen": packed["all_top5_supplied"],
                "n_evidence_blocks_supplied": packed["n_evidence_blocks_supplied"],
                "input_truncated": packed["input_truncated"],
                "prompt_eval_count": packed["prompt_eval_count"],
                "flan_answer": row["generated_answer"],
                "qwen_answer": packed["answer"],
                "citation_validation": packed["citation_validation"],
                "citation_status": packed["citation_status"],
                "generation": packed["generation"],
                "context_sent_to_llm": packed["context"],
            }
        )
        print(row["id"], "tokens", packed["prompt_eval_count"], "blocks", packed["n_evidence_blocks_supplied"], "trunc", packed["input_truncated"])

    ctrl13 = step13["insufficient_context_control"]
    ctrl_hits = hits_for_context(ctrl13["retrieval_top5"])
    ctrl_ids = [h["chunk_id"] for h in ctrl_hits]
    expected = ["1.46#translation", "16.21#translation", "11.3#translation"]
    if ctrl_ids != expected:
        raise SystemExit(f"Control chunk IDs {ctrl_ids} != {expected}")
    if any(cid.startswith("2.47") for cid in ctrl_ids):
        raise SystemExit("Control accidentally includes 2.47")
    packed_ctrl = generate_one(ctrl13["question"], ctrl13["retrieval_top5"])
    control = {
        "id": ctrl13["id"],
        "label": ctrl13["label"],
        "question": ctrl13["question"],
        "case_type": "insufficient_context_control",
        "forced_chunk_ids": ctrl_ids,
        "flan_answer": ctrl13["generated_answer"],
        "qwen_answer": packed_ctrl["answer"],
        "citation_validation": packed_ctrl["citation_validation"],
        "citation_status": packed_ctrl["citation_status"],
        "prompt_eval_count": packed_ctrl["prompt_eval_count"],
        "input_truncated": packed_ctrl["input_truncated"],
        "generation": packed_ctrl["generation"],
        "context_sent_to_llm": packed_ctrl["context"],
    }

    payload = {
        "normal": normal_out,
        "insufficient_context_control": control,
        "prompt_file": str(PROMPT_PATH.name),
        "prompt_intent_unchanged": True,
        "prompt_on_disk_sha_prefix": __import__("hashlib").sha256(prompt_on_disk.encode()).hexdigest()[:16],
    }
    OUT_RESULTS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = {
        "model": MODEL,
        "ollama": "0.33.0",
        "quantization": "Q4_K_M",
        "parameters": "3.1B",
        "size_on_disk": "1.9 GB",
        "context_length_model": 32768,
        "num_ctx_used": NUM_CTX,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "num_predict": NUM_PREDICT,
        "processor_observed": "100% GPU (ollama ps after warmup generate)",
        "n_normal": len(normal_out),
        "all_questions_all_top5_supplied": all(r["all_top5_supplied_to_qwen"] for r in normal_out),
        "any_truncation": any(r["input_truncated"] for r in normal_out),
        "prompt_eval_counts": {r["id"]: r["prompt_eval_count"] for r in normal_out},
        "gold_chunk_in_top5_count": sum(1 for r in normal_out if r["gold_chunk_in_top5"]),
        "citation_status_counts": {
            s: sum(1 for r in normal_out if r["citation_status"] == s)
            for s in ("CORRECT", "PARTIALLY CORRECT", "MISSING", "UNSUPPORTED")
        },
        "control_qwen_answer": control["qwen_answer"],
        "retrieval_frozen": True,
        "used_saved_step13_top5": True,
        "did_not_rerun_flan": True,
        "did_not_rerun_e5": True,
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = {
        "step": 14,
        "status": "ran_qwen2.5_3b",
        "model": MODEL,
        "provider": "ollama",
        "temperature": TEMPERATURE,
        "num_ctx": NUM_CTX,
        "num_predict": NUM_PREDICT,
        "seed": SEED,
        "retrieval": "frozen_step13_saved_top5",
    }
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {OUT_RESULTS}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
