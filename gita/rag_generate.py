"""
Step 13 RAG baseline: frozen E5 Top-5 retrieval + grounded generation.

Does not change FAISS, embeddings, BM25, RRF, gating, or section rerank.
Gold labels are never passed to the generator.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from chunker import OVERLAP_WORDS
from evaluate_e5 import query_input, sha256_file

ROOT = Path(__file__).resolve().parent
INDEX_DEFAULT = ROOT / "gita_e5.faiss"
META_DEFAULT = ROOT / "gita_e5_metadata.json"
PROMPT_PATH = ROOT / "gita_rag_prompt.txt"
SMOKE_Q = ROOT / "gita_generation_smoke_questions.json"
SMOKE_RESULTS = ROOT / "gita_generation_smoke_results.json"
SMOKE_REPORT = ROOT / "gita_generation_smoke_report.json"
DEV30 = ROOT / "gita_eval_frozen.json"
DEV40 = ROOT / "gita_holdout_frozen.json"

E5_MODEL = "intfloat/multilingual-e5-large"
GEN_MODEL = "google/flan-t5-small"
TOP_K = 5
MAX_NEW_TOKENS = 192
MAX_INPUT_TOKENS = 512

CITATION_RE = re.compile(
    r"Bhagavad[\s-]*Gita\s+(\d{1,2}\.\d{1,3}(?:\s*[–-]\s*\d{1,3})?)",
    re.I,
)


def load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


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
    """Format evidence. Strip exact 40-word overlap between consecutive same-source purport parts."""
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


class E5Retriever:
    def __init__(self, index_path: Path = INDEX_DEFAULT, meta_path: Path = META_DEFAULT, model_name: str = E5_MODEL):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if len(self.metadata) != 1443:
            raise SystemExit(f"Expected 1443 metadata rows, found {len(self.metadata)}")
        self.by_id = {m["chunk_id"]: m for m in self.metadata}
        self.index = faiss.read_index(str(index_path))
        if self.index.ntotal != 1443:
            raise SystemExit("FAISS ntotal is not 1443.")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.model.eval()
        self.index_path = index_path

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        import torch

        with torch.inference_mode():
            q = self.model.encode(
                [query_input(question)],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device="cpu",
            )
        q = np.asarray(q, dtype=np.float32)
        scores, rows = self.index.search(q, top_k)
        hits = []
        for rank in range(top_k):
            meta = self.metadata[int(rows[0, rank])]
            hits.append(
                {
                    "rank": rank + 1,
                    "chunk_id": meta["chunk_id"],
                    "source_record_id": meta["source_record_id"],
                    "chapter": meta["chapter"],
                    "verse_start": meta["verse_start"],
                    "verse_end": meta["verse_end"],
                    "section": meta["section"],
                    "part": meta.get("part"),
                    "n_parts": meta.get("n_parts"),
                    "page_start": meta.get("page_start"),
                    "page_end": meta.get("page_end"),
                    "text": meta["text"],
                    "score": float(scores[0, rank]),
                }
            )
        return hits

    def hits_from_ids(self, chunk_ids: list[str]) -> list[dict]:
        hits = []
        for rank, cid in enumerate(chunk_ids, start=1):
            meta = self.by_id[cid]
            hits.append(
                {
                    "rank": rank,
                    "chunk_id": cid,
                    "source_record_id": meta["source_record_id"],
                    "chapter": meta["chapter"],
                    "verse_start": meta["verse_start"],
                    "verse_end": meta["verse_end"],
                    "section": meta["section"],
                    "part": meta.get("part"),
                    "n_parts": meta.get("n_parts"),
                    "page_start": meta.get("page_start"),
                    "page_end": meta.get("page_end"),
                    "text": meta["text"],
                    "score": None,
                    "note": "forced_unrelated_context",
                }
            )
        return hits


class FlanT5Generator:
    def __init__(self, model_name: str = GEN_MODEL):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        self.model.eval()
        self.max_input_tokens = min(MAX_INPUT_TOKENS, int(getattr(self.tokenizer, "model_max_length", MAX_INPUT_TOKENS) or MAX_INPUT_TOKENS))
        if self.max_input_tokens > 4096:
            self.max_input_tokens = MAX_INPUT_TOKENS

    def generate(self, question: str, context: str) -> dict:
        import torch

        template = load_prompt_template()
        full_prompt = template.replace("{question}", question).replace("{context}", context)
        truncated_context, n_kept = fit_context(
            self.tokenizer, template, question, context, self.max_input_tokens
        )
        prompt = template.replace("{question}", question).replace("{context}", truncated_context)
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return {
            "prompt_full": full_prompt,
            "prompt_sent_to_model": prompt,
            "context_truncated_for_model": truncated_context != context,
            "n_evidence_blocks_in_model_prompt": n_kept,
            "answer": text,
            "generation": {
                "model": self.model_name,
                "local": True,
                "already_cached": True,
                "temperature": 0,
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_input_tokens": self.max_input_tokens,
            },
        }


def fit_context(tokenizer, template: str, question: str, context: str, max_tokens: int) -> tuple[str, int]:
    blocks = re.split(r"\n(?=\[EVIDENCE )", context)
    kept: list[str] = []
    for block in blocks:
        trial = "\n".join(kept + [block]).strip()
        prompt = template.replace("{question}", question).replace("{context}", trial)
        n = len(tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])
        if n > max_tokens and kept:
            break
        kept.append(block)
        if n > max_tokens:
            break
    return "\n".join(kept).strip(), len(kept)


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


def gold_map() -> dict[str, dict]:
    out = {}
    for path in (DEV30, DEV40):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for q in payload["questions"]:
            chunks = {cid for g in q["gold_sources"] for cid in g["gold_chunk_ids"]}
            sources = {g["source_record_id"] for g in q["gold_sources"]}
            out[q["id"]] = {"gold_chunk_ids": sorted(chunks), "gold_source_ids": sorted(sources)}
    return out


def run_smoke() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    spec = json.loads(SMOKE_Q.read_text(encoding="utf-8"))
    golds = gold_map()
    faiss_hash = sha256_file(INDEX_DEFAULT)
    retriever = E5Retriever()
    generator = FlanT5Generator()

    results = []
    for item in spec["questions"]:
        qtext = item["question"]
        hits = retriever.retrieve(qtext, top_k=TOP_K)
        context, hits_ctx = build_context(hits)
        gen = generator.generate(qtext, context)
        cites = validate_citations(gen["answer"], hits)
        g = golds.get(item["source_eval_id"], {"gold_chunk_ids": [], "gold_source_ids": []})
        retrieved_cids = {h["chunk_id"] for h in hits}
        retrieved_sids = {h["source_record_id"] for h in hits}
        results.append(
            {
                "id": item["id"],
                "source_eval_id": item["source_eval_id"],
                "category": item["category"],
                "question": qtext,
                "case_type": "normal_smoke",
                "retrieval_top5": hits_ctx,
                "gold_chunk_ids": g["gold_chunk_ids"],
                "gold_source_ids": g["gold_source_ids"],
                "gold_chunk_in_top5": any(cid in retrieved_cids for cid in g["gold_chunk_ids"]),
                "gold_source_in_top5": any(sid in retrieved_sids for sid in g["gold_source_ids"]),
                "context_sent_to_llm": gen["prompt_sent_to_model"],
                "context_full": context,
                "context_truncated_for_model": gen["context_truncated_for_model"],
                "n_evidence_blocks_in_model_prompt": gen["n_evidence_blocks_in_model_prompt"],
                "generated_answer": gen["answer"],
                "citation_validation": cites,
                "generation": gen["generation"],
                "manual_review": {
                    "correctness": None,
                    "grounding": None,
                    "citation": None,
                    "section_attribution": None,
                    "notes": "Fill by human review. Not scored by the generator.",
                },
            }
        )

    ctrl = spec["insufficient_context_control"]
    forced = retriever.hits_from_ids(ctrl["forced_chunk_ids"])
    context, hits_ctx = build_context(forced)
    gen = generator.generate(ctrl["question"], context)
    cites = validate_citations(gen["answer"], forced)
    control = {
        "id": ctrl["id"],
        "label": ctrl["label"],
        "question": ctrl["question"],
        "case_type": "insufficient_context_control",
        "forced_chunk_ids": ctrl["forced_chunk_ids"],
        "retrieval_top5": hits_ctx,
        "context_sent_to_llm": gen["prompt_sent_to_model"],
        "generated_answer": gen["answer"],
        "citation_validation": cites,
        "generation": gen["generation"],
        "refused_or_said_insufficient": None,
        "note": ctrl["note"],
    }

    report = {
                "pipeline": f"question -> frozen E5 Top 5 -> formatted context -> {GEN_MODEL} -> answer",
        "retriever": "plain_e5",
        "e5_faiss_sha256": faiss_hash,
        "n_chunks": 1443,
        "top_k": TOP_K,
        "generator": GEN_MODEL,
        "generator_source": "already_in_local_huggingface_cache",
        "not_used": ["bm25", "rrf", "citation_gating", "section_aware", "itihasa_lora"],
        "generation_settings": results[0]["generation"] if results else {},
        "n_smoke_questions": len(results),
        "smoke_ids": [r["id"] for r in results],
        "citation_support_counts": {
            "all_supported": sum(1 for r in results if r["citation_validation"]["all_citations_supported"]),
            "unsupported_any": sum(1 for r in results if r["citation_validation"]["unsupported_citations"]),
            "no_citation_extracted": sum(1 for r in results if not r["citation_validation"]["cited_any"]),
        },
        "gold_chunk_in_top5_count": sum(1 for r in results if r["gold_chunk_in_top5"]),
        "insufficient_context_control": {
            "id": control["id"],
            "answer": control["generated_answer"],
            "citations": control["citation_validation"],
        },
        "scientific_note": "Generation smoke test only. Not a retrieval re-evaluation. Not a full generation benchmark.",
    }
    SMOKE_RESULTS.write_text(
        json.dumps({"normal": results, "insufficient_context_control": control}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    SMOKE_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "generator": GEN_MODEL,
        "n": len(results),
        "gold_chunk_in_top5": report["gold_chunk_in_top5_count"],
        "citation_support_counts": report["citation_support_counts"],
        "control_answer": control["generated_answer"],
        "answers": [{"id": r["id"], "gold_in_top5": r["gold_chunk_in_top5"], "answer": r["generated_answer"], "cites": r["citation_validation"]} for r in results],
    }, indent=2, ensure_ascii=False))
    print(f"Wrote {SMOKE_RESULTS}")
    print(f"Wrote {SMOKE_REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gita RAG generate (frozen E5).")
    parser.add_argument("--smoke", action="store_true", help="Run the 10-question generation smoke test.")
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
        return
    parser.print_help()


if __name__ == "__main__":
    main()
