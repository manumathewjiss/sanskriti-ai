#!/usr/bin/env python3
"""
CURRENT FAISS baseline evaluation (translation-retrieval PROXY).

Measures the EXISTING production FAISS dense retriever.
Does NOT change production retrieval, embeddings, prefixes, normalization,
FAISS metric, document text, RETRIEVER_K, or generation.

Proxy task
----------
For each sampled CSV row:
  query  = English translation (`en`)
  gold   = the Sanskrit shloka (`sn`) from THE SAME row

This is NOT an evaluation of real user questions. A high score means the
index can recover a verse from its paired translation.

Gold identifier
---------------
Uses metadata ``row_index`` (pandas index of res/combined.csv).

``book + chapter + shloka_num`` is NOT unique in this corpus (the same
chapter/shloka numbers repeat across volumes/kandas). ``row_index`` is unique
and is already stored on every FAISS Document by load_and_chunk_itihasa_data().

Usage
-----
    ./venv/bin/python eval/evaluate_faiss_baseline.py --prepare-sample-only
    ./venv/bin/python eval/evaluate_faiss_baseline.py --allow-rebuild
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Evaluation-process only. Set thread/process limits BEFORE numpy/torch/pandas.
# Does not change production FastAPI/retrieval code.
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

# Eval-process only. importlib.metadata.packages_distributions() walks every
# site-packages top_level.txt. On this machine that call blocks for minutes or
# raises Errno 89 (iCloud Documents File Provider), so torch/transformers never
# finish importing. FastAPI/production retrieval is unchanged.
import importlib.metadata as _importlib_metadata

_importlib_metadata.packages_distributions = lambda: {}  # type: ignore[method-assign]

print("eval: imports starting", flush=True)
import pandas as pd
print("eval: pandas ok", flush=True)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# multilingual-e5-large hung on Apple MPS (Metal copy waitUntilCompleted) during
# the first baseline rebuild, so this process forces CPU for embedding.
try:
    import torch

    torch.backends.mps.is_available = lambda: False  # type: ignore[method-assign]
    torch.backends.mps.is_built = lambda: False  # type: ignore[method-assign]
    print("eval: torch ok", torch.__version__, flush=True)
except Exception as _torch_err:
    print("eval: torch skip", _torch_err, flush=True)

print("eval: importing config", flush=True)
from config import Config  # noqa: E402
print("eval: config ok", flush=True)

# Do NOT import data_loader / rag_chain / models at module load.
# Those pull indic_transliteration, PEFT, and the Qwen loader. This eval only
# needs HuggingFaceEmbeddings + FAISS.load_local for the saved index.
print("eval: importing FAISS + HuggingFaceEmbeddings", flush=True)
from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
print("eval: langchain FAISS ok", flush=True)

SAMPLE_PATH = ROOT / "eval" / "data" / "faiss_baseline_sample.json"
RESULTS_DIR = ROOT / "eval" / "results"

DEFAULT_SEED = 42
DEFAULT_N = 500
DEFAULT_EVAL_K = 10

PROXY_LIMITATION = (
    "LIMITATION: This is a translation-retrieval proxy, not an evaluation of "
    "real user questions. Each query is the English `en` field; the gold "
    "document is the Sanskrit `sn` from the SAME CSV row. Success means FAISS "
    "recovered the paired verse from its translation, not that answers to "
    "open questions are correct."
)


def eligible_rows(df: pd.DataFrame) -> pd.DataFrame:
    sn = df["sn"].fillna("").astype(str).str.strip()
    en = df["en"].fillna("").astype(str).str.strip()
    return df.loc[(sn != "") & (en != "")].copy()


def build_frozen_sample(csv_path: Path, n: int, seed: int) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    df.index.name = "row_index"
    eligible = eligible_rows(df).reset_index()

    counts = eligible["book"].value_counts()
    books = sorted(counts.index.tolist())
    if len(eligible) < n:
        raise ValueError(f"Only {len(eligible)} eligible rows, cannot sample {n}")

    alloc: Dict[str, int] = {}
    remaining = n
    for i, book in enumerate(books):
        if i == len(books) - 1:
            alloc[book] = remaining
        else:
            share = int(round(n * int(counts[book]) / len(eligible)))
            share = max(1, min(share, int(counts[book]), remaining - (len(books) - i - 1)))
            alloc[book] = share
            remaining -= share

    examples: List[Dict[str, Any]] = []
    rng_seed = seed
    for book, take in alloc.items():
        subset = eligible[eligible["book"] == book]
        if take > len(subset):
            raise ValueError(f"Book {book}: need {take}, have {len(subset)}")
        picked = subset.sample(n=take, random_state=rng_seed).sort_values("row_index")
        rng_seed += 1
        for _, row in picked.iterrows():
            examples.append(
                {
                    "row_index": int(row["row_index"]),
                    "book": str(row["book"]),
                    "chapter": str(row["chapter"]),
                    "shloka_num": str(row["shloka"]),
                    "query": str(row["en"]).strip(),
                    "gold_sn": str(row["sn"]).strip(),
                }
            )

    examples.sort(key=lambda x: x["row_index"])
    book_counts = dict(Counter(ex["book"] for ex in examples))
    return {
        "description": (
            "Frozen FAISS baseline eval sample. Translation-retrieval proxy: "
            "query=en, gold=same-row sn. Do not regenerate between experiments "
            "unless you intentionally create a new named sample file."
        ),
        "limitation": PROXY_LIMITATION,
        "corpus": str(csv_path.as_posix()),
        "seed": seed,
        "n": len(examples),
        "book_counts": book_counts,
        "gold_id_field": "row_index",
        "gold_id_reason": (
            "row_index is unique and already stored in FAISS Document.metadata. "
            "book+chapter+shloka_num is NOT unique in combined.csv."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "examples": examples,
    }


def save_sample(sample: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")


def load_sample(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def production_index_exists() -> bool:
    path = Config.FAISS_INDEX_PATH
    if not os.path.isdir(path):
        return False
    names = set(os.listdir(path))
    return "index.faiss" in names and "index.pkl" in names


def load_or_rebuild_production_index(allow_rebuild: bool):
    """
    Load the production FAISS index with the production embedding constructor.

    Existing-index path matches models.load_embedding_model + FAISS.load_local
    used by rag_chain.initialize_vector_store (no encode_kwargs, prefixes, or
    normalize_L2). Rebuild still lazy-imports the production helpers.
    """
    if production_index_exists():
        print(f"Loading existing FAISS index from {Config.FAISS_INDEX_PATH}", flush=True)
        print(f"Embedding constructor: HuggingFaceEmbeddings(model_name={Config.EMBEDDINGS_MODEL})", flush=True)
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDINGS_MODEL)
        print("eval: embeddings loaded", flush=True)
        store = FAISS.load_local(
            folder_path=Config.FAISS_INDEX_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        print("eval: FAISS.load_local ok", flush=True)
        return store, False

    print("")
    print("=" * 70)
    print("NO COMPATIBLE FAISS INDEX FOUND")
    print("=" * 70)
    print(f"Looked at: {Config.FAISS_INDEX_PATH}")
    print("Why rebuild is required: evaluation must search the CURRENT")
    print("production index, and that folder is missing on disk.")
    print("Rebuild operation: embed all ~93,030 Sanskrit `sn` texts with")
    print(f"  {Config.EMBEDDINGS_MODEL}")
    print("  HuggingFaceEmbeddings(model_name=...) only")
    print("  no query:/passage: prefixes")
    print("  no embedding normalization")
    print("  FAISS.from_documents → IndexFlatL2 (Euclidean)")
    print("Production configuration is preserved. This does not change")
    print("RETRIEVER_K, FastAPI, document page_content, or generation.")
    print("=" * 70)

    if not allow_rebuild:
        print("Refusing to rebuild. Re-run with --allow-rebuild if you want")
        print("this script to create the production index, then evaluate.")
        sys.exit(2)

    print("\n--allow-rebuild set. Building production FAISS index...")
    from data_loader import load_and_chunk_itihasa_data
    from rag_chain import initialize_vector_store

    docs = load_and_chunk_itihasa_data(Config.DATA_PATH)
    if not docs:
        raise RuntimeError(f"No documents loaded from {Config.DATA_PATH}")
    print(f"Loaded {len(docs)} documents from {Config.DATA_PATH}")
    store = initialize_vector_store(docs)
    if store is None:
        raise RuntimeError("FAISS rebuild failed")
    return store, True


def row_index_from_doc(doc) -> Optional[int]:
    raw = doc.metadata.get("row_index")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def retrieve_top_k(vector_store, query: str, k: int) -> List[Tuple[Any, float]]:
    """
    Direct FAISS k-NN. Does not use Config.RETRIEVER_K / get_retriever(),
    so production k stays 5.
    Returned score is L2 distance (lower = closer). Not cosine similarity.
    """
    return vector_store.similarity_search_with_score(query, k=k)


def rank_of_gold(hits: Sequence[Tuple[Any, float]], gold_row_index: int) -> Optional[int]:
    for i, (doc, _dist) in enumerate(hits, start=1):
        if row_index_from_doc(doc) == gold_row_index:
            return i
    return None


def metrics_from_ranks(ranks: Sequence[Optional[int]], k_eval: int) -> Dict[str, Any]:
    n = len(ranks)
    if n == 0:
        return {
            "n": 0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "found_in_top_k": 0,
            "missing_in_top_k": 0,
            "mean_rank_when_found": None,
        }
    cap10 = min(10, k_eval)
    hits1 = sum(1 for r in ranks if r is not None and r <= 1)
    hits5 = sum(1 for r in ranks if r is not None and r <= 5)
    hits10 = sum(1 for r in ranks if r is not None and r <= cap10)
    mrr = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / n
    found = [r for r in ranks if r is not None]
    return {
        "n": n,
        "recall_at_1": hits1 / n,
        "recall_at_5": hits5 / n,
        "recall_at_10": hits10 / n,
        "mrr": mrr,
        "found_in_top_k": len(found),
        "missing_in_top_k": n - len(found),
        "mean_rank_when_found": (sum(found) / len(found)) if found else None,
    }


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def print_group(title: str, m: Dict[str, Any], eval_k: int) -> None:
    mean_rank = m["mean_rank_when_found"]
    mean_rank_s = f"{mean_rank:.4f}" if mean_rank is not None else "n/a"
    print(title)
    print(f"Examples: {m['n']}")
    print(f"Recall@1: {pct(m['recall_at_1'])}")
    print(f"Recall@5: {pct(m['recall_at_5'])}")
    print(f"Recall@10: {pct(m['recall_at_10'])}")
    print(f"MRR: {m['mrr']:.4f}")
    print(f"Average rank when found in Top {eval_k}: {mean_rank_s}")
    print(f"Not in Top {eval_k}: {m['missing_in_top_k']}")
    print("")


def run_evaluation(
    vector_store,
    sample: Dict[str, Any],
    eval_k: int,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    examples = sample["examples"]
    rows_out: List[Dict[str, Any]] = []
    ranks_all: List[Optional[int]] = []
    ranks_by_book: Dict[str, List[Optional[int]]] = {}

    t0 = time.time()
    for i, ex in enumerate(examples, start=1):
        gold_id = int(ex["row_index"])
        hits = retrieve_top_k(vector_store, ex["query"], k=eval_k)
        rank = rank_of_gold(hits, gold_id)
        ranks_all.append(rank)
        ranks_by_book.setdefault(ex["book"], []).append(rank)

        retrieved_ids = [row_index_from_doc(d) for d, _ in hits]
        l2_distances = [float(dist) for _d, dist in hits]
        retrieved_meta = [
            {
                "row_index": row_index_from_doc(d),
                "book": d.metadata.get("book"),
                "chapter": d.metadata.get("chapter"),
                "shloka_num": d.metadata.get("shloka_num"),
                "l2_distance": float(dist),
            }
            for d, dist in hits
        ]

        rows_out.append(
            {
                "query": ex["query"],
                "gold_row_index": gold_id,
                "gold_book": ex["book"],
                "gold_chapter": ex["chapter"],
                "gold_shloka_num": ex["shloka_num"],
                "gold_sn": ex.get("gold_sn", ""),
                "retrieved_top10_row_index": retrieved_ids,
                "retrieved_top10_l2_distance": l2_distances,
                "retrieved_top10": retrieved_meta,
                "gold_rank": rank,
                "hit_at_1": bool(rank is not None and rank <= 1),
                "hit_at_5": bool(rank is not None and rank <= 5),
                "hit_at_10": bool(rank is not None and rank <= 10),
            }
        )
        if i % 25 == 0 or i == len(examples):
            print(f"  evaluated {i}/{len(examples)} ({time.time() - t0:.1f}s)", flush=True)
            if checkpoint_path is not None:
                checkpoint_path.write_text(
                    json.dumps(
                        {
                            "n_done": i,
                            "n_total": len(examples),
                            "elapsed_sec": time.time() - t0,
                            "overall_so_far": metrics_from_ranks(ranks_all, eval_k),
                            "rows": rows_out,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

    return {
        "rows": rows_out,
        "overall": metrics_from_ranks(ranks_all, eval_k),
        "by_book": {b: metrics_from_ranks(rs, eval_k) for b, rs in ranks_by_book.items()},
        "elapsed_sec": time.time() - t0,
    }


def flatten_result_row(row: Dict[str, Any]) -> Dict[str, Any]:
    ids = row["retrieved_top10_row_index"]
    dists = row["retrieved_top10_l2_distance"]
    return {
        "query": row["query"],
        "gold_row_index": row["gold_row_index"],
        "gold_book": row["gold_book"],
        "gold_chapter": row["gold_chapter"],
        "gold_shloka_num": row["gold_shloka_num"],
        "retrieved_top10_row_index": "|".join("" if x is None else str(x) for x in ids),
        "retrieved_top10_l2_distance": "|".join(f"{x:.6f}" for x in dists),
        "gold_rank": "" if row["gold_rank"] is None else row["gold_rank"],
        "hit_at_1": int(row["hit_at_1"]),
        "hit_at_5": int(row["hit_at_5"]),
        "hit_at_10": int(row["hit_at_10"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sample-path", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Sample size when creating a new sample file")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--eval-k",
        type=int,
        default=DEFAULT_EVAL_K,
        help="Retrieve this many neighbors for metrics (does not change production RETRIEVER_K)",
    )
    parser.add_argument("--prepare-sample-only", action="store_true")
    parser.add_argument(
        "--force-new-sample",
        action="store_true",
        help="Overwrite the frozen sample. Do not use between comparable experiments.",
    )
    parser.add_argument(
        "--allow-rebuild",
        action="store_true",
        help="If the production FAISS folder is missing, rebuild it with UNCHANGED production settings",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Score only the first N frozen examples (smoke test). 0 = all.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if Config.DEMO_MODE:
        print("ERROR: SANSKRITY_DEMO is set. Refuse to evaluate the demo MiniLM path.")
        print("Unset SANSKRITY_DEMO so Config.EMBEDDINGS_MODEL is intfloat/multilingual-e5-large.")
        return 2

    print(PROXY_LIMITATION)
    print("")
    print("Production settings this eval must NOT change:")
    print(f"  EMBEDDINGS_MODEL = {Config.EMBEDDINGS_MODEL}")
    print(f"  FAISS_INDEX_PATH = {Config.FAISS_INDEX_PATH}")
    print(f"  DATA_PATH        = {Config.DATA_PATH}")
    print(f"  RETRIEVER_K      = {Config.RETRIEVER_K} (production; eval retrieves {args.eval_k})")
    print("")

    if args.prepare_sample_only or not args.sample_path.exists() or args.force_new_sample:
        if args.sample_path.exists() and args.prepare_sample_only and not args.force_new_sample:
            existing = load_sample(args.sample_path)
            print(f"Sample already exists at {args.sample_path} (n={existing.get('n')}).")
            print("Not overwriting. Pass --force-new-sample to replace it.")
            return 0
        print(f"Creating frozen sample n={args.n} seed={args.seed} from {Config.DATA_PATH}")
        sample = build_frozen_sample(Path(Config.DATA_PATH), n=args.n, seed=args.seed)
        save_sample(sample, args.sample_path)
        print(f"Wrote {args.sample_path}")
        print(f"Book counts: {sample['book_counts']}")
        if args.prepare_sample_only:
            return 0
    else:
        sample = load_sample(args.sample_path)
        print(f"Reusing frozen sample {args.sample_path} (n={sample['n']}, seed={sample.get('seed')})")
        print(f"Book counts: {sample.get('book_counts')}")

    if args.limit and args.limit > 0:
        sample = dict(sample)
        sample["examples"] = sample["examples"][: args.limit]
        sample["n"] = len(sample["examples"])
        sample["book_counts"] = dict(Counter(ex["book"] for ex in sample["examples"]))
        print(f"--limit {args.limit}: scoring {sample['n']} examples (frozen file unchanged)")

    if args.eval_k < 10:
        print("WARNING: --eval-k < 10; Recall@10 will be limited by retrieved depth.")

    store, rebuilt = load_or_rebuild_production_index(allow_rebuild=args.allow_rebuild)
    print(f"Index rebuilt this run: {rebuilt}")
    print(f"Searching Top {args.eval_k} with similarity_search_with_score (L2 distance).")
    print("Query encoding uses the vector store's production HuggingFaceEmbeddings.")
    print("No query: prefix. No normalization. No MMR. No rerank.")
    print("")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.results_dir / "faiss_baseline_partial.json"
    result = run_evaluation(
        store, sample, eval_k=args.eval_k, checkpoint_path=checkpoint_path
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    detailed_json = args.results_dir / f"faiss_baseline_detailed_{stamp}.json"
    detailed_csv = args.results_dir / "faiss_baseline_detailed.csv"
    summary_path = args.results_dir / "faiss_baseline_summary.json"

    summary = {
        "limitation": PROXY_LIMITATION,
        "gold_id_field": "row_index",
        "embeddings_model": Config.EMBEDDINGS_MODEL,
        "faiss_index_path": Config.FAISS_INDEX_PATH,
        "production_retriever_k": Config.RETRIEVER_K,
        "eval_k": args.eval_k,
        "distance": "L2 / Euclidean (lower is closer). NOT cosine similarity.",
        "prefixes": "none",
        "normalization": False,
        "document_page_content": "Sanskrit sn only",
        "sample_path": str(args.sample_path),
        "sample_n": sample["n"],
        "book_counts": sample.get("book_counts"),
        "index_rebuilt_this_run": rebuilt,
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
    pd.DataFrame([flatten_result_row(r) for r in result["rows"]]).to_csv(detailed_csv, index=False)

    overall = result["overall"]
    rama = result["by_book"].get("ramayana")
    maha = result["by_book"].get("mahabharata")
    book_counts = sample.get("book_counts") or {}

    print("")
    print("=" * 70)
    print("CURRENT FAISS BASELINE")
    print("=" * 70)
    print(PROXY_LIMITATION)
    print("")
    print(f"Evaluation examples: {overall['n']}")
    print(f"Ramayana examples: {book_counts.get('ramayana', 'n/a')}")
    print(f"Mahabharata examples: {book_counts.get('mahabharata', 'n/a')}")
    print("")
    print(f"Recall@1: {pct(overall['recall_at_1'])}")
    print(f"Recall@5: {pct(overall['recall_at_5'])}")
    print(f"Recall@10: {pct(overall['recall_at_10'])}")
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
        print_group("RAMAYANA", rama, args.eval_k)
    if maha:
        print_group("MAHABHARATA", maha, args.eval_k)
    print(f"Detailed CSV: {detailed_csv}")
    print(f"Detailed JSON: {detailed_json}")
    print(f"Summary JSON: {summary_path}")
    print("L2 distances are Euclidean (lower = closer). They are not cosine similarity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
