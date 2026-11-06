"""
Step 15: prompt-only generation experiment.

Frozen: E5 Top 5 (saved Step 13/14), context construction, qwen2.5:3b, Ollama settings.
Changed: grounding prompt only.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

OVERLAP_WORDS = 40

ROOT = Path(__file__).resolve().parent
PROMPT_PATH = ROOT / "gita_rag_prompt_step15.txt"
STEP13 = ROOT / "gita_generation_smoke_results.json"
STEP14 = ROOT / "gita_generation_step14_qwen_results.json"
OUT_RESULTS = ROOT / "gita_generation_step15_results.json"
OUT_REPORT = ROOT / "gita_generation_step15_report.json"

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
TEMPERATURE = 0
NUM_PREDICT = 512
NUM_CTX = 8192
SEED = 0

# Source plus optional Translation/Purport, with or without parentheses.
CITATION_RE = re.compile(
    r"Bhagavad[\s-]*Gita\s+"
    r"(\d{1,2}\.\d{1,3}(?:\s*[–-]\s*\d{1,3})?)"
    r"(?:\s*,\s*(Translation|Purport))?",
    re.I,
)


def load_prompt_parts() -> tuple[str, str]:
    raw = PROMPT_PATH.read_text(encoding="utf-8")
    marker = "\nQuestion:\n"
    if marker not in raw:
        raise SystemExit("Step 15 prompt file missing Question: marker")
    system, rest = raw.split(marker, 1)
    user_template = "Question:\n" + rest
    return system.strip(), user_template


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


def extract_citations(answer: str) -> list[dict]:
    found = []
    for m in CITATION_RE.finditer(answer):
        section = m.group(2)
        found.append(
            {
                "source_id": normalize_cite(m.group(1)),
                "section": section.lower() if section else None,
                "raw": m.group(0),
            }
        )
    return found


def validate_citations(answer: str, hits: list[dict]) -> dict:
    retrieved_sources = {h["source_record_id"] for h in hits}
    retrieved_pairs = {(h["source_record_id"], h["section"]) for h in hits}
    cited = extract_citations(answer)
    unsupported = []
    section_mismatch = []
    section_unspecified = []
    ok = []
    for c in cited:
        sid = c["source_id"]
        if sid not in retrieved_sources:
            unsupported.append(c)
            continue
        if c["section"] is None:
            section_unspecified.append(c)
            continue
        if (sid, c["section"]) not in retrieved_pairs:
            section_mismatch.append(c)
            continue
        ok.append(c)
    cited_any = len(cited) > 0
    if not cited_any:
        status = "MISSING"
    elif unsupported and not ok and not section_mismatch and not section_unspecified:
        status = "UNSUPPORTED"
    elif unsupported and (ok or section_mismatch or section_unspecified):
        status = "PARTIALLY CORRECT"
    elif section_mismatch and not ok and not section_unspecified:
        status = "PARTIALLY CORRECT"
    elif section_mismatch:
        status = "PARTIALLY CORRECT"
    elif section_unspecified and not ok:
        status = "PARTIALLY CORRECT"
    elif section_unspecified and ok:
        status = "PARTIALLY CORRECT"
    else:
        status = "CORRECT"
    return {
        "cited": cited,
        "cited_source_ids": [c["source_id"] for c in cited],
        "retrieved_source_ids": sorted(retrieved_sources),
        "unsupported_citations": unsupported,
        "section_mismatch": section_mismatch,
        "section_unspecified": section_unspecified,
        "section_ok": ok,
        "all_citations_supported": len(unsupported) == 0,
        "cited_any": cited_any,
        "citation_status": status,
        "has_section_mismatch": len(section_mismatch) > 0,
    }


def user_prompt(question: str, context: str) -> str:
    return USER_TEMPLATE.replace("{question}", question).replace("{context}", context)


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
    return {
        "answer": answer,
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "done_reason": data.get("done_reason"),
    }


def hits_for_context(saved_hits: list[dict]) -> list[dict]:
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
    gen = ollama_chat(question, context)
    cites = validate_citations(gen["answer"], hits)
    return {
        "context": context,
        "hits_ctx": hits_ctx,
        "n_evidence_blocks_supplied": n_blocks,
        "all_top5_supplied": n_blocks == len(hits),
        "input_truncated": n_blocks < len(hits),
        "prompt_eval_count": gen["prompt_eval_count"],
        "eval_count": gen["eval_count"],
        "answer": gen["answer"],
        "citation_validation": cites,
        "citation_status": cites["citation_status"],
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
    step14 = json.loads(STEP14.read_text(encoding="utf-8"))
    prompt_sha = hashlib.sha256(PROMPT_PATH.read_text(encoding="utf-8").encode()).hexdigest()
    normal_out = []
    for row in step14["normal"]:
        ids14 = list(row["top5_chunk_ids"])
        if ids14 != [h["chunk_id"] for h in row["retrieval_top5"]]:
            raise SystemExit(f"{row['id']} Step 14 top5 list mismatch vs retrieval_top5")
        packed = generate_one(row["question"], row["retrieval_top5"])
        ids_now = [h["chunk_id"] for h in packed["hits_ctx"]]
        if ids_now != ids14:
            raise SystemExit(f"{row['id']} Top 5 mismatch\n step14={ids14}\n now={ids_now}")
        ctx14 = row.get("context_sent_to_llm")
        if ctx14 is not None and packed["context"] != ctx14:
            raise SystemExit(f"{row['id']} rebuilt context does not match Step 14 context")
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
                "top5_chunk_ids": ids14,
                "retrieval_top5": packed["hits_ctx"],
                "all_top5_supplied_to_qwen": packed["all_top5_supplied"],
                "n_evidence_blocks_supplied": packed["n_evidence_blocks_supplied"],
                "input_truncated": packed["input_truncated"],
                "prompt_eval_count": packed["prompt_eval_count"],
                "step14_answer": row["qwen_answer"],
                "step15_answer": packed["answer"],
                "citation_validation": packed["citation_validation"],
                "citation_status": packed["citation_status"],
                "generation": packed["generation"],
                "context_sent_to_llm": packed["context"],
            }
        )
        print(
            row["id"],
            "tokens",
            packed["prompt_eval_count"],
            "blocks",
            packed["n_evidence_blocks_supplied"],
            "status",
            packed["citation_status"],
        )

    step13 = json.loads(STEP13.read_text(encoding="utf-8"))
    ctrl14 = step14["insufficient_context_control"]
    ctrl13 = step13["insufficient_context_control"]
    expected = ["1.46#translation", "16.21#translation", "11.3#translation"]
    ctrl_hits = ctrl13["retrieval_top5"]
    ctrl_ids = [h["chunk_id"] for h in ctrl_hits]
    if ctrl_ids != expected or ctrl14["forced_chunk_ids"] != expected:
        raise SystemExit(f"Control chunk IDs {ctrl_ids} != {expected}")
    if any(cid.startswith("2.47") for cid in ctrl_ids):
        raise SystemExit("Control accidentally includes 2.47")
    packed_ctrl = generate_one(ctrl14["question"], ctrl_hits)
    if packed_ctrl["context"] != ctrl14["context_sent_to_llm"]:
        raise SystemExit("Control rebuilt context does not match Step 14")
    control = {
        "id": ctrl14["id"],
        "label": ctrl14.get("label"),
        "question": ctrl14["question"],
        "case_type": "insufficient_context_control",
        "forced_chunk_ids": expected,
        "step14_answer": ctrl14["qwen_answer"],
        "step15_answer": packed_ctrl["answer"],
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
        "prompt_file": PROMPT_PATH.name,
        "prompt_sha256": prompt_sha,
        "only_change": "grounding_prompt",
        "used_saved_step14_top5": True,
        "context_matched_step14": True,
    }
    OUT_RESULTS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "step": 15,
        "model": MODEL,
        "num_ctx": NUM_CTX,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "num_predict": NUM_PREDICT,
        "n_normal": len(normal_out),
        "all_questions_all_top5_supplied": all(r["all_top5_supplied_to_qwen"] for r in normal_out),
        "any_truncation": any(r["input_truncated"] for r in normal_out),
        "prompt_eval_counts": {r["id"]: r["prompt_eval_count"] for r in normal_out},
        "citation_status_counts": {
            s: sum(1 for r in normal_out if r["citation_status"] == s)
            for s in ("CORRECT", "PARTIALLY CORRECT", "MISSING", "UNSUPPORTED")
        },
        "section_mismatch_count": sum(
            1 for r in normal_out if r["citation_validation"]["has_section_mismatch"]
        ),
        "control_step15_answer": control["step15_answer"],
        "prompt_file": PROMPT_PATH.name,
        "prompt_sha256": prompt_sha,
        "did_not_rerun_e5": True,
        "did_not_change_model": True,
        "did_not_change_generation_settings": True,
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {OUT_RESULTS}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
