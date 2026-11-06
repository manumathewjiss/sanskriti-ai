#!/usr/bin/env python3
"""
English-only FAISS experiment (ONE variable vs the finished Sanskrit baseline).

CHANGE: Document.page_content = English `en` instead of Sanskrit `sn`.

UNCHANGED: embedding model, no E5 prefixes, no normalization, FAISS L2,
query text, frozen 500-query sample, metrics, production Sanskrit index.

Does NOT write to sanskrit_faiss_index_itihasa/.
Does NOT modify src/ production retrieval.

Usage
-----
    ./venv/bin/python eval/evaluate_faiss_english_only.py --allow-rebuild
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import importlib.metadata as _importlib_metadata

_importlib_metadata.packages_distributions = lambda: {}  # type: ignore[method-assign]

print("english-eval: imports starting", flush=True)
import pandas as pd
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

try:
    import torch

    torch.backends.mps.is_available = lambda: False  # type: ignore[method-assign]
    torch.backends.mps.is_built = lambda: False  # type: ignore[method-assign]
    print("english-eval: torch ok", torch.__version__, flush=True)
except Exception as _torch_err:
    print("english-eval: torch skip", _torch_err, flush=True)

from config import Config  # noqa: E402
from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402

import evaluate_faiss_baseline as baseline  # noqa: E402

ENGLISH_INDEX_PATH = ROOT / "faiss_index_itihasa_english_only"
RESULTS_DIR = ROOT / "eval" / "results" / "english_only"
BASELINE_INDEX = Path(Config.FAISS_INDEX_PATH).resolve()
BATCH_SIZE = 512


def clean_text(text: str) -> str:
    """Same whitespace/control cleaning as src.utils.clean_text (copied to avoid importing utils)."""
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t\r")
    return text.strip()


def build_english_documents(csv_path: Path) -> Tuple[List[Document], int, int]:
    df = pd.read_csv(csv_path)
    documents: List[Document] = []
    skipped_empty_en = 0
    for index, row in df.iterrows():
        english_text = clean_text(str(row.get("en", "")))
        if not english_text:
            skipped_empty_en += 1
            continue
        metadata = {
            "book": str(row.get("book", "Unknown")),
            "chapter": str(row.get("chapter", "Unknown")),
            "shloka_num": str(row.get("shloka", "Unknown")),
            "english_text": english_text,
            "source_file": str(csv_path),
            "row_index": index,
        }
        documents.append(Document(page_content=english_text, metadata=metadata))
    return documents, skipped_empty_en, len(df)


def index_exists(path: Path) -> bool:
    return path.is_dir() and (path / "index.faiss").is_file() and (path / "index.pkl").is_file()


def build_or_load_english_index(allow_rebuild: bool, embeddings) -> Tuple[FAISS, bool, int]:
    if ENGLISH_INDEX_PATH.resolve() == BASELINE_INDEX:
        raise RuntimeError("Refusing to use the Sanskrit baseline index path.")
    if index_exists(ENGLISH_INDEX_PATH):
        print(f"Loading existing English-only index from {ENGLISH_INDEX_PATH}", flush=True)
        store = FAISS.load_local(
            folder_path=str(ENGLISH_INDEX_PATH),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        n = int(store.index.ntotal)
        print(f"Loaded ntotal={n}", flush=True)
        return store, False, n

    if not allow_rebuild:
        print(f"English-only index missing at {ENGLISH_INDEX_PATH}")
        print("Re-run with --allow-rebuild to embed English `en` texts into a NEW index.")
        print(f"Will not touch {BASELINE_INDEX}")
        sys.exit(2)

    csv_path = Path(Config.DATA_PATH)
    print(f"Building English-only documents from {csv_path}", flush=True)
    docs, skipped, nrows = build_english_documents(csv_path)
    print(f"CSV rows: {nrows}", flush=True)
    print(f"English documents (non-empty en): {len(docs)}", flush=True)
    print(f"Skipped empty en: {skipped}", flush=True)
    print(f"Index path: {ENGLISH_INDEX_PATH}", flush=True)
    print(f"Baseline Sanskrit index left untouched: {BASELINE_INDEX}", flush=True)
    if not docs:
        raise RuntimeError("No English documents to index")

    ENGLISH_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"FAISS.from_documents batch 1/{((len(docs) - 1) // BATCH_SIZE) + 1} (size={min(BATCH_SIZE, len(docs))})", flush=True)
    store = FAISS.from_documents(docs[:BATCH_SIZE], embeddings)
    store.save_local(str(ENGLISH_INDEX_PATH))
    for start in range(BATCH_SIZE, len(docs), BATCH_SIZE):
        batch = docs[start : start + BATCH_SIZE]
        batch_i = start // BATCH_SIZE + 1
        total_b = (len(docs) - 1) // BATCH_SIZE + 1
        print(
            f"add_documents batch {batch_i}/{total_b} "
            f"({start + 1}-{start + len(batch)} / {len(docs)}) "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        store.add_documents(batch)
        store.save_local(str(ENGLISH_INDEX_PATH))
    print(f"English-only index saved ntotal={store.index.ntotal} in {time.time() - t0:.1f}s", flush=True)
    return store, True, int(store.index.ntotal)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-rebuild", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-k", type=int, default=baseline.DEFAULT_EVAL_K)
    args = parser.parse_args()

    if Config.DEMO_MODE:
        print("ERROR: SANSKRITY_DEMO is set. Unset it.")
        return 2

    print("EXPERIMENT: English `en` as Document.page_content (Sanskrit baseline untouched).")
    print("Unchanged: e5-large, no prefixes, no normalization, IndexFlatL2, frozen sample.")
    print(f"  EMBEDDINGS_MODEL = {Config.EMBEDDINGS_MODEL}")
    print(f"  DATA_PATH        = {Config.DATA_PATH}")
    print(f"  ENGLISH_INDEX    = {ENGLISH_INDEX_PATH}")
    print(f"  BASELINE_INDEX   = {BASELINE_INDEX} (read-only, not used)")
    print(f"  SAMPLE           = {baseline.SAMPLE_PATH}")
    print("")

    sample = baseline.load_sample(baseline.SAMPLE_PATH)
    print(f"Reusing frozen sample {baseline.SAMPLE_PATH} (n={sample['n']}, seed={sample.get('seed')})")
    print(f"Book counts: {sample.get('book_counts')}")
    if args.limit and args.limit > 0:
        from collections import Counter

        sample = dict(sample)
        sample["examples"] = sample["examples"][: args.limit]
        sample["n"] = len(sample["examples"])
        sample["book_counts"] = dict(Counter(ex["book"] for ex in sample["examples"]))
        print(f"--limit {args.limit}: scoring {sample['n']} (frozen file unchanged)")

    print(f"Embedding constructor: HuggingFaceEmbeddings(model_name={Config.EMBEDDINGS_MODEL})", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDINGS_MODEL)
    print("english-eval: embeddings loaded", flush=True)

    store, rebuilt, ntotal = build_or_load_english_index(args.allow_rebuild, embeddings)
    print(f"Index rebuilt this run: {rebuilt}")
    print(f"Indexed documents: {ntotal}")
    print(f"Searching Top {args.eval_k} with similarity_search_with_score (L2).")
    print("No query: prefix. No normalization. No MMR. No rerank.")
    print("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = RESULTS_DIR / "faiss_english_only_partial.json"
    result = baseline.run_evaluation(store, sample, eval_k=args.eval_k, checkpoint_path=checkpoint_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    detailed_json = RESULTS_DIR / f"faiss_english_only_detailed_{stamp}.json"
    detailed_csv = RESULTS_DIR / "faiss_english_only_detailed.csv"
    summary_path = RESULTS_DIR / "faiss_english_only_summary.json"

    summary = {
        "experiment": "english_only_page_content",
        "changed_variable": "Document.page_content: sn -> en",
        "baseline_index_untouched": str(BASELINE_INDEX),
        "english_index_path": str(ENGLISH_INDEX_PATH),
        "indexed_documents": ntotal,
        "index_rebuilt_this_run": rebuilt,
        "limitation": baseline.PROXY_LIMITATION,
        "gold_id_field": "row_index",
        "embeddings_model": Config.EMBEDDINGS_MODEL,
        "production_retriever_k": Config.RETRIEVER_K,
        "eval_k": args.eval_k,
        "distance": "L2 / Euclidean (lower is closer). NOT cosine similarity.",
        "prefixes": "none",
        "normalization": False,
        "document_page_content": "English en only",
        "sample_path": str(baseline.SAMPLE_PATH),
        "sample_n": sample["n"],
        "book_counts": sample.get("book_counts"),
        "elapsed_sec": result["elapsed_sec"],
        "overall": result["overall"],
        "by_book": result["by_book"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    detailed_json.write_text(
        json.dumps({"summary": summary, "rows": result["rows"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([baseline.flatten_result_row(r) for r in result["rows"]]).to_csv(detailed_csv, index=False)

    overall = result["overall"]
    rama = result["by_book"].get("ramayana")
    maha = result["by_book"].get("mahabharata")
    book_counts = sample.get("book_counts") or {}

    print("")
    print("=" * 70)
    print("ENGLISH-ONLY FAISS EXPERIMENT")
    print("=" * 70)
    print(baseline.PROXY_LIMITATION)
    print("")
    print(f"Evaluation examples: {overall['n']}")
    print(f"Ramayana examples: {book_counts.get('ramayana', 'n/a')}")
    print(f"Mahabharata examples: {book_counts.get('mahabharata', 'n/a')}")
    print(f"Indexed documents: {ntotal}")
    print("")
    print(f"Recall@1: {baseline.pct(overall['recall_at_1'])}")
    print(f"Recall@5: {baseline.pct(overall['recall_at_5'])}")
    print(f"Recall@10: {baseline.pct(overall['recall_at_10'])}")
    print(f"MRR: {overall['mrr']:.4f}")
    mean_rank = overall["mean_rank_when_found"]
    print(
        "Average rank when found in Top "
        + str(args.eval_k)
        + ": "
        + (f"{mean_rank:.4f}" if mean_rank is not None else "n/a")
    )
    print(f"Correct shloka not in Top {args.eval_k}: {overall['missing_in_top_k']}")
    print("")
    if rama:
        baseline.print_group("RAMAYANA", rama, args.eval_k)
    if maha:
        baseline.print_group("MAHABHARATA", maha, args.eval_k)
    print(f"Detailed CSV: {detailed_csv}")
    print(f"Detailed JSON: {detailed_json}")
    print(f"Summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
