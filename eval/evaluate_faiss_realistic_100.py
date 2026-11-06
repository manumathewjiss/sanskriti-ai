#!/usr/bin/env python3
"""
Frozen 100-question realistic FAISS evaluation.

READ-ONLY on indexes. Does not rebuild Sanskrit or English FAISS.
Does not create Sanskrit+English. Does not change production retrieval.
Does not modify the frozen evaluation dataset or gold labels.

Scoring: a hit at K if ANY gold row_index from relevant_row_indices is in Top K.
MRR uses the rank of the first relevant gold document.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

print("realistic-100-eval: imports starting", flush=True)

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
    print("realistic-100-eval: torch ok", torch.__version__, flush=True)
except Exception as _torch_err:
    print("realistic-100-eval: torch skip", _torch_err, flush=True)

from config import Config  # noqa: E402
from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402

import evaluate_faiss_baseline as baseline  # noqa: E402

FROZEN_PATH = ROOT / "eval" / "data" / "realistic_eval_100_frozen.json"
SANSKRIT_INDEX = Path(Config.FAISS_INDEX_PATH).resolve()
ENGLISH_INDEX = ROOT / "faiss_index_itihasa_english_only"
RESULTS_PARENT = ROOT / "eval" / "results" / "realistic_eval_100"
EVAL_K = 10
EXPECTED_NTOTAL = 93030
EXPECTED_N = 100
EXPECTED_RAMAYANA = 20
EXPECTED_MAHABHARATA = 80
EXPECTED_SINGLE_GOLD = 94
EXPECTED_MULTI_GOLD = 6


def index_exists(path: Path) -> bool:
    return path.is_dir() and (path / "index.faiss").is_file() and (path / "index.pkl").is_file()


def load_existing_index(path: Path, embeddings) -> FAISS:
    if not index_exists(path):
        raise FileNotFoundError(f"Required existing index missing: {path}")
    print(f"Loading existing index (read-only): {path}", flush=True)
    store = FAISS.load_local(
        folder_path=str(path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    print(
        f"  ntotal={store.index.ntotal} mtime_faiss={time.ctime((path / 'index.faiss').stat().st_mtime)}",
        flush=True,
    )
    return store


class FakeDoc:
    def __init__(self, row_index: Optional[int]) -> None:
        self.metadata = {"row_index": row_index}


def collect_row_indices(store: FAISS) -> set:
    found: set = set()
    docstore = store.docstore
    docs_by_id = getattr(docstore, "_dict", None)
    if isinstance(docs_by_id, dict):
        iterable = docs_by_id.values()
    else:
        iterable = [store.docstore.search(did) for did in store.index_to_docstore_id.values()]
    for doc in iterable:
        rid = baseline.row_index_from_doc(doc)
        if rid is not None:
            found.add(rid)
    return found


def validate_multi_gold_scoring() -> None:
    hits = [
        (FakeDoc(10), 0.9),
        (FakeDoc(11), 0.8),
        (FakeDoc(99), 0.7),
        (FakeDoc(12), 0.6),
    ]
    rank = first_relevant_rank(hits, [99, 12])
    if rank != 3:
        raise RuntimeError(f"Multi-gold scoring failed: expected first relevant rank 3, got {rank}")
    miss = first_relevant_rank(hits, [500])
    if miss is not None:
        raise RuntimeError(f"Multi-gold scoring failed: expected miss, got {miss}")
    print("OK  multi-gold scoring: ANY gold in Top K (synthetic rank=3)", flush=True)


def validate_frozen_file(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    questions = payload.get("questions")
    if not isinstance(questions, list) or len(questions) != EXPECTED_N:
        raise RuntimeError(
            f"Frozen JSON must contain exactly {EXPECTED_N} questions, found "
            f"{0 if not isinstance(questions, list) else len(questions)}"
        )
    if not payload.get("frozen"):
        raise RuntimeError("Frozen JSON missing frozen=true")
    ids = []
    texts = []
    primaries = []
    n_rama = 0
    n_maha = 0
    n_multi = 0
    for i, q in enumerate(questions):
        for key in ("question_id", "question", "primary_row_index", "relevant_row_indices", "book"):
            if key not in q:
                raise RuntimeError(f"Question {i} missing {key}")
        gold = [int(x) for x in q["relevant_row_indices"]]
        primary = int(q["primary_row_index"])
        if not gold or primary not in gold:
            raise RuntimeError(f"{q['question_id']}: primary_row_index must be inside relevant_row_indices")
        if not str(q["question"]).strip():
            raise RuntimeError(f"{q['question_id']}: empty question")
        ids.append(q["question_id"])
        texts.append(q["question"])
        primaries.append(primary)
        if q["book"] == "ramayana":
            n_rama += 1
        elif q["book"] == "mahabharata":
            n_maha += 1
        if len(gold) > 1:
            n_multi += 1
    if len(set(ids)) != EXPECTED_N or len(set(texts)) != EXPECTED_N:
        raise RuntimeError("Duplicate question ids or question texts")
    if len(set(primaries)) != EXPECTED_N:
        raise RuntimeError("Duplicate primary_row_index values")
    if n_rama != EXPECTED_RAMAYANA or n_maha != EXPECTED_MAHABHARATA:
        raise RuntimeError(f"Unexpected book counts: ramayana={n_rama} mahabharata={n_maha}")
    n_single = EXPECTED_N - n_multi
    if n_single != EXPECTED_SINGLE_GOLD or n_multi != EXPECTED_MULTI_GOLD:
        raise RuntimeError(f"Unexpected gold counts: single={n_single} multi={n_multi}")
    print(
        f"OK  frozen JSON: n={EXPECTED_N}, Rama={n_rama}, Maha={n_maha}, "
        f"single-gold={n_single}, multi-gold={n_multi}",
        flush=True,
    )
    return questions


def validate_index(name: str, store: FAISS, gold_ids: Sequence[int]) -> None:
    ntotal = int(store.index.ntotal)
    if ntotal != EXPECTED_NTOTAL:
        raise RuntimeError(f"{name}: unexpected ntotal={ntotal}, expected {EXPECTED_NTOTAL}")
    present = collect_row_indices(store)
    if not present:
        raise RuntimeError(f"{name}: no row_index metadata found")
    missing = [g for g in gold_ids if int(g) not in present]
    if missing:
        raise RuntimeError(f"{name}: gold row_index missing from index metadata: {missing}")
    print(f"OK  {name}: ntotal={ntotal}, row_index present, all gold ids found ({len(present)} docs with row_index)", flush=True)


def first_relevant_rank(hits: Sequence[Tuple[Any, float]], gold_ids: Sequence[int]) -> Optional[int]:
    gold = set(int(g) for g in gold_ids)
    for i, (doc, _dist) in enumerate(hits, start=1):
        rid = baseline.row_index_from_doc(doc)
        if rid is not None and rid in gold:
            return i
    return None


def hit_rows(hits: Sequence[Tuple[Any, float]], n: int) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for doc, _dist in hits[:n]:
        out.append(baseline.row_index_from_doc(doc))
    return out


def evaluate_index(name: str, store: FAISS, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    first_ranks: List[Optional[int]] = []
    t0 = time.time()
    for q in questions:
        qid = q["question_id"]
        query = q["question"]
        gold = [int(x) for x in q["relevant_row_indices"]]
        hits = baseline.retrieve_top_k(store, query, EVAL_K)
        rank = first_relevant_rank(hits, gold)
        first_ranks.append(rank)
        rows.append(
            {
                "question_id": qid,
                "question": query,
                "book": q["book"],
                "question_category": q.get("question_category"),
                "primary_row_index": int(q["primary_row_index"]),
                "gold_row_indices": gold,
                "first_relevant_rank": rank,
                "top10_row_index": hit_rows(hits, 10),
                "top10_l2": [float(d) for _doc, d in hits[:10]],
                "hit_at_1": bool(rank is not None and rank <= 1),
                "hit_at_5": bool(rank is not None and rank <= 5),
                "hit_at_10": bool(rank is not None and rank <= 10),
            }
        )
        print(f"  {name} {qid} first_relevant_rank={rank}", flush=True)
    elapsed = time.time() - t0
    return {
        "index_name": name,
        "elapsed_sec": elapsed,
        "overall": baseline.metrics_from_ranks(first_ranks, EVAL_K),
        "rows": rows,
        "first_ranks": first_ranks,
    }


def group_metrics(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    buckets: Dict[str, List[Optional[int]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key) or "unknown")].append(row["first_relevant_rank"])
    return {name: baseline.metrics_from_ranks(ranks, EVAL_K) for name, ranks in sorted(buckets.items())}


def main() -> int:
    if Config.DEMO_MODE:
        print("ERROR: SANSKRITY_DEMO is set. Unset it.")
        return 2
    if SANSKRIT_INDEX == ENGLISH_INDEX.resolve():
        raise RuntimeError("Sanskrit and English index paths must differ.")

    print("STEP 2: VALIDATE EXISTING INDEXES + FROZEN DATASET", flush=True)
    if not FROZEN_PATH.is_file():
        print(f"STOP: missing frozen dataset {FROZEN_PATH}")
        return 2
    if not index_exists(SANSKRIT_INDEX):
        print(f"STOP: Sanskrit index missing at {SANSKRIT_INDEX}")
        return 2
    if not index_exists(ENGLISH_INDEX):
        print(f"STOP: English index missing at {ENGLISH_INDEX}")
        return 2
    print(f"OK  Sanskrit index files exist: {SANSKRIT_INDEX}", flush=True)
    print(f"OK  English index files exist: {ENGLISH_INDEX.resolve()}", flush=True)

    try:
        validate_multi_gold_scoring()
        payload = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        questions = validate_frozen_file(payload)
    except Exception as err:
        print(f"STOP: validation failed before index load: {err}")
        return 2

    gold_ids = sorted({int(x) for q in questions for x in q["relevant_row_indices"]})
    print("EVAL: frozen realistic queries vs existing Sanskrit-only and English-only FAISS.")
    print("No rebuild. No prefixes. No normalization. L2. eval_k=10. Multi-gold ANY-in-Top-K.")
    print("Do not change questions or gold labels after seeing results.")
    print(f"  EMBEDDINGS_MODEL = {Config.EMBEDDINGS_MODEL}")
    print(f"  FROZEN           = {FROZEN_PATH}")
    print(f"  SANSKRIT_INDEX   = {SANSKRIT_INDEX}")
    print(f"  ENGLISH_INDEX    = {ENGLISH_INDEX.resolve()}")
    print(f"  n questions      = {len(questions)}")
    print("")

    if Config.EMBEDDINGS_MODEL != "intfloat/multilingual-e5-large":
        print(f"STOP: unexpected embedding model {Config.EMBEDDINGS_MODEL}")
        return 2

    print(f"Embedding constructor: HuggingFaceEmbeddings(model_name={Config.EMBEDDINGS_MODEL})", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDINGS_MODEL)
    sanskrit_store = load_existing_index(SANSKRIT_INDEX, embeddings)
    english_store = load_existing_index(ENGLISH_INDEX, embeddings)
    try:
        validate_index("sanskrit_only", sanskrit_store, gold_ids)
        validate_index("english_only", english_store, gold_ids)
    except Exception as err:
        print(f"STOP: index validation failed: {err}")
        return 2

    print("STEP 3: Searching Sanskrit-only...", flush=True)
    sanskrit = evaluate_index("sanskrit_only", sanskrit_store, questions)
    print("Searching English-only...", flush=True)
    english = evaluate_index("english_only", english_store, questions)

    per_question: List[Dict[str, Any]] = []
    both: List[str] = []
    english_only: List[str] = []
    sanskrit_only: List[str] = []
    neither: List[str] = []
    for s_row, e_row, q in zip(sanskrit["rows"], english["rows"], questions):
        s_rank = s_row["first_relevant_rank"]
        e_rank = e_row["first_relevant_rank"]
        s_ok = bool(s_rank is not None and s_rank <= EVAL_K)
        e_ok = bool(e_rank is not None and e_rank <= EVAL_K)
        qid = q["question_id"]
        if s_ok and e_ok:
            both.append(qid)
        elif e_ok:
            english_only.append(qid)
        elif s_ok:
            sanskrit_only.append(qid)
        else:
            neither.append(qid)
        per_question.append(
            {
                "question_id": qid,
                "question": q["question"],
                "book": q["book"],
                "question_category": q.get("question_category"),
                "gold_row_indices": [int(x) for x in q["relevant_row_indices"]],
                "sanskrit_first_relevant_rank": s_rank,
                "english_first_relevant_rank": e_rank,
                "sanskrit_top10_row_index": s_row["top10_row_index"],
                "english_top10_row_index": e_row["top10_row_index"],
                "sanskrit_success": s_ok,
                "english_success": e_ok,
            }
        )

    s_m = sanskrit["overall"]
    e_m = english["overall"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR = RESULTS_PARENT / f"run_{stamp}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "realistic_eval_100_frozen",
        "interpretation": (
            "This is the main realistic retrieval comparison (n=100 frozen questions). "
            "Do not declare a winner from the old 500-query translation benchmark. "
            "Do not change questions or gold labels after seeing these results."
        ),
        "frozen_path": str(FROZEN_PATH),
        "gold_id_field": "row_index",
        "scoring": "success if ANY relevant_row_indices appears in Top K; MRR uses first relevant rank",
        "embeddings_model": Config.EMBEDDINGS_MODEL,
        "prefixes": "none",
        "normalization": False,
        "distance": "L2 / Euclidean (lower is closer). NOT cosine similarity.",
        "eval_k": EVAL_K,
        "sanskrit_index_path": str(SANSKRIT_INDEX),
        "english_index_path": str(ENGLISH_INDEX.resolve()),
        "sanskrit_index_ntotal": int(sanskrit_store.index.ntotal),
        "english_index_ntotal": int(english_store.index.ntotal),
        "index_rebuilt_this_run": False,
        "n": len(questions),
        "sanskrit_only": s_m,
        "english_only": e_m,
        "difference_english_minus_sanskrit": {
            "recall_at_1": e_m["recall_at_1"] - s_m["recall_at_1"],
            "recall_at_5": e_m["recall_at_5"] - s_m["recall_at_5"],
            "recall_at_10": e_m["recall_at_10"] - s_m["recall_at_10"],
            "mrr": e_m["mrr"] - s_m["mrr"],
        },
        "sanskrit_n_top1": sum(1 for r in sanskrit["first_ranks"] if r is not None and r <= 1),
        "sanskrit_n_top5": sum(1 for r in sanskrit["first_ranks"] if r is not None and r <= 5),
        "sanskrit_n_top10": sum(1 for r in sanskrit["first_ranks"] if r is not None and r <= 10),
        "english_n_top1": sum(1 for r in english["first_ranks"] if r is not None and r <= 1),
        "english_n_top5": sum(1 for r in english["first_ranks"] if r is not None and r <= 5),
        "english_n_top10": sum(1 for r in english["first_ranks"] if r is not None and r <= 10),
        "found_by_both": both,
        "english_only_successes": english_only,
        "sanskrit_only_successes": sanskrit_only,
        "neither_successes": neither,
        "by_book_sanskrit": group_metrics(sanskrit["rows"], "book"),
        "by_book_english": group_metrics(english["rows"], "book"),
        "by_category_sanskrit": group_metrics(sanskrit["rows"], "question_category"),
        "by_category_english": group_metrics(english["rows"], "question_category"),
        "sanskrit_elapsed_sec": sanskrit["elapsed_sec"],
        "english_elapsed_sec": english["elapsed_sec"],
        "per_question": per_question,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = RESULTS_DIR / "faiss_realistic_100_summary.json"
    detailed_path = RESULTS_DIR / f"faiss_realistic_100_detailed_{stamp}.json"
    csv_path = RESULTS_DIR / "faiss_realistic_100_per_question.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    detailed_path.write_text(
        json.dumps({"summary": summary, "sanskrit_rows": sanskrit["rows"], "english_rows": english["rows"]}, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "question",
                "book",
                "question_category",
                "gold_row_indices",
                "sanskrit_first_relevant_rank",
                "english_first_relevant_rank",
                "sanskrit_top10_row_index",
                "english_top10_row_index",
                "sanskrit_success",
                "english_success",
            ],
        )
        w.writeheader()
        for row in per_question:
            w.writerow(
                {
                    **row,
                    "gold_row_indices": "|".join(str(x) for x in row["gold_row_indices"]),
                    "sanskrit_top10_row_index": "|".join(
                        "" if x is None else str(x) for x in row["sanskrit_top10_row_index"]
                    ),
                    "english_top10_row_index": "|".join(
                        "" if x is None else str(x) for x in row["english_top10_row_index"]
                    ),
                }
            )

    def show(label: str, m: Dict[str, Any]) -> None:
        print(label)
        print(f"  n={m['n']}")
        print(f"  Recall@1:  {baseline.pct(m['recall_at_1'])}")
        print(f"  Recall@5:  {baseline.pct(m['recall_at_5'])}")
        print(f"  Recall@10: {baseline.pct(m['recall_at_10'])}")
        print(f"  MRR:       {m['mrr']:.4f}")
        print(f"  found in Top {EVAL_K}: {m['found_in_top_k']}  missing: {m['missing_in_top_k']}")

    print("")
    print("=" * 70)
    print("FROZEN REALISTIC BENCHMARK (n=100) — main realistic comparison")
    print("=" * 70)
    show("Sanskrit-only (sn index)", s_m)
    print("")
    show("English-only (en index)", e_m)
    print("")
    print(f"both={len(both)} english_only={len(english_only)} sanskrit_only={len(sanskrit_only)} neither={len(neither)}")
    print(f"Summary: {summary_path}")
    print(f"CSV:     {csv_path}")
    print(f"Detailed:{detailed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
