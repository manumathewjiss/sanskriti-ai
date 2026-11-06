"""
Build an exact E5 + FAISS dense index over gita/gita_chunks.json.

This does not evaluate questions and does not load gold labels into embeddings.

Pipeline:
  chunk["text"]  ->  "passage: " + text  ->  E5 vector  ->  L2-normalize  ->  FAISS IndexFlatIP

Normalized vectors + inner product  ≈  cosine similarity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CHUNKS_DEFAULT = ROOT / "gita_chunks.json"
INDEX_DEFAULT = ROOT / "gita_e5.faiss"
META_DEFAULT = ROOT / "gita_e5_metadata.json"
TRUNC_REPORT_DEFAULT = ROOT / "gita_e5_truncation_report.json"

MODEL_NAME = "intfloat/multilingual-e5-large"


def passage_input(text: str) -> str:
    return "passage: " + text


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed Gita chunks with E5 and build FAISS IndexFlatIP.")
    parser.add_argument("--chunks", type=Path, default=CHUNKS_DEFAULT)
    parser.add_argument("--index", type=Path, default=INDEX_DEFAULT)
    parser.add_argument("--metadata", type=Path, default=META_DEFAULT)
    parser.add_argument("--truncation-report", type=Path, default=TRUNC_REPORT_DEFAULT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise SystemExit("gita_chunks.json must be a list of chunk objects.")

    print(f"Loaded {len(chunks)} chunks from {args.chunks}")
    model = SentenceTransformer(args.model)
    model.eval()
    tokenizer = model.tokenizer
    max_seq = int(model.max_seq_length)
    print(f"Model {args.model}")
    print(f"max_seq_length={max_seq}  embedding_dim={model.get_embedding_dimension()}")

    inputs = [passage_input(c["text"]) for c in chunks]
    token_lens = [count_tokens(tokenizer, t) for t in inputs]
    n_over = sum(1 for n in token_lens if n > max_seq)
    report = {
        "model": args.model,
        "max_seq_length": max_seq,
        "n_chunks": len(chunks),
        "token_count_includes_passage_prefix_and_special_tokens": True,
        "min_tokens": min(token_lens),
        "median_tokens": sorted(token_lens)[len(token_lens) // 2],
        "max_tokens_before_truncation": max(token_lens),
        "n_chunks_exceeding_limit": n_over,
        "n_chunks_that_would_be_truncated": n_over,
        "over_limit_chunk_ids": [
            chunks[i]["chunk_id"] for i, n in enumerate(token_lens) if n > max_seq
        ],
    }
    args.truncation_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "over_limit_chunk_ids"}, indent=2))
    if n_over:
        print(
            f"WARNING: {n_over} chunks exceed {max_seq} tokens and WOULD be truncated "
            "by E5. Not changing the chunker in this step."
        )

    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    with torch.inference_mode():
        embeddings = model.encode(
            inputs,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            device="cpu",
        )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise SystemExit(f"Passage embeddings not unit-normalized: min={norms.min()} max={norms.max()}")

    dim = embeddings.shape[1]
    if embeddings.shape[0] != len(chunks):
        raise SystemExit("Embedding count does not match chunk count.")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    if index.ntotal != len(chunks):
        raise SystemExit("FAISS ntotal does not match chunk count.")
    faiss.write_index(index, str(args.index))

    metadata = []
    seen_ids = set()
    for i, chunk in enumerate(chunks):
        cid = chunk["chunk_id"]
        if cid in seen_ids:
            raise SystemExit(f"Duplicate chunk_id in metadata: {cid}")
        seen_ids.add(cid)
        metadata.append(
            {
                "faiss_row": i,
                "chunk_id": cid,
                "source_record_id": chunk["source_record_id"],
                "chapter": chunk["chapter"],
                "verse_start": chunk["verse_start"],
                "verse_end": chunk["verse_end"],
                "text_label": chunk["text_label"],
                "section": chunk["section"],
                "part": chunk["part"],
                "n_parts": chunk["n_parts"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "text": chunk["text"],
                "n_tokens_untruncated": token_lens[i],
            }
        )
    args.metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote FAISS index ({index.ntotal} x {dim}) to {args.index}")
    print(f"Wrote metadata to {args.metadata}")
    print(f"Wrote truncation report to {args.truncation_report}")


if __name__ == "__main__":
    main()
