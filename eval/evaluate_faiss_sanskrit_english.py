#!/usr/bin/env python3
"""
Sanskrit + English FAISS experiment (ONE variable: document representation).

CHANGE: Document.page_content = cleaned_sn + "\\n\\n" + cleaned_en
        ONE vector per CSV row (~93,030), not two vectors per row.

UNCHANGED: embedding model, no E5 prefixes, no normalization, FAISS L2,
query text, frozen 100 realistic questions, multi-gold scoring.

Does NOT write to:
  - sanskrit_faiss_index_itihasa/
  - faiss_index_itihasa_english_only/

Does NOT modify src/ production retrieval.

Usage
-----
    ./venv/bin/python eval/evaluate_faiss_sanskrit_english.py --allow-rebuild
    ./venv/bin/python eval/evaluate_faiss_sanskrit_english.py   # load + eval only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

print("skt+eng-eval: imports starting", flush=True)

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
    print("skt+eng-eval: torch ok", torch.__version__, flush=True)
except Exception as _torch_err:
    print("skt+eng-eval: torch skip", _torch_err, flush=True)

from config import Config  # noqa: E402
from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402

import evaluate_faiss_baseline as baseline  # noqa: E402

COMBINED_INDEX_PATH = ROOT / "sanskrit_english_faiss_index_itihasa"
SANSKRIT_INDEX = Path(Config.FAISS_INDEX_PATH).resolve()
ENGLISH_INDEX = (ROOT / "faiss_index_itihasa_english_only").resolve()
FROZEN_PATH = ROOT / "eval" / "data" / "realistic_eval_100_frozen.json"
ENGLISH_ONLY_SUMMARY = (
    ROOT
    / "eval"
    / "results"
    / "realistic_eval_100"
    / "run_20260825T224728Z"
    / "faiss_realistic_100_summary.json"
)
RESULTS_PARENT = ROOT / "eval" / "results" / "realistic_eval_100_sanskrit_english"
BATCH_SIZE = 512
EXPECTED_NTOTAL = 93030
EVAL_K = 10
PAGE_CONTENT_SEPARATOR = "\n\n"

# Known frozen English-only / Sanskrit-only metrics for the comparison table.
SANSKRIT_ONLY_METRICS = {
    "recall_at_1": 0.01,
    "recall_at_5": 0.05,
    "recall_at_10": 0.08,
    "mrr": 0.0292,
}
ENGLISH_ONLY_METRICS = {
    "recall_at_1": 0.29,
    "recall_at_5": 0.40,
    "recall_at_10": 0.47,
    "mrr": 0.3421,
}


def clean_text(text: str) -> str:
    """Same whitespace/control cleaning as English-only eval / src.utils.clean_text."""
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t\r")
    return text.strip()


def assert_safe_output_path() -> None:
    out = COMBINED_INDEX_PATH.resolve()
    if out == SANSKRIT_INDEX:
        raise RuntimeError(f"Refusing: combined path equals Sanskrit index {SANSKRIT_INDEX}")
    if out == ENGLISH_INDEX:
        raise RuntimeError(f"Refusing: combined path equals English index {ENGLISH_INDEX}")
    # Also refuse if somehow nested under protected dirs
    for protected in (SANSKRIT_INDEX, ENGLISH_INDEX):
        try:
            out.relative_to(protected)
            raise RuntimeError(f"Refusing: combined path is inside protected index {protected}")
        except ValueError:
            pass
    print(f"OK  safe output path: {out}", flush=True)
    print(f"OK  will not touch Sanskrit: {SANSKRIT_INDEX}", flush=True)
    print(f"OK  will not touch English:  {ENGLISH_INDEX}", flush=True)


def index_exists(path: Path) -> bool:
    return path.is_dir() and (path / "index.faiss").is_file() and (path / "index.pkl").is_file()


def file_fingerprint(path: Path) -> Dict[str, Any]:
    st = path.stat()
    # Hash only first 1MB + size for speed; full sha of 363MB is slow.
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(1024 * 1024))
    return {
        "path": str(path),
        "size": st.st_size,
        "mtime": st.st_mtime,
        "mtime_human": time.ctime(st.st_mtime),
        "sha256_first_1mb": h.hexdigest(),
    }


def build_combined_documents(csv_path: Path) -> Tuple[List[Document], int, int, int]:
    df = pd.read_csv(csv_path)
    documents: List[Document] = []
    skipped_empty_sn = 0
    skipped_empty_en = 0
    for index, row in df.iterrows():
        sn = clean_text(str(row.get("sn", "")))
        en = clean_text(str(row.get("en", "")))
        if not sn:
            skipped_empty_sn += 1
            continue
        if not en:
            skipped_empty_en += 1
            continue
        page_content = f"{sn}{PAGE_CONTENT_SEPARATOR}{en}"
        metadata = {
            "book": str(row.get("book", "Unknown")),
            "chapter": str(row.get("chapter", "Unknown")),
            "shloka_num": str(row.get("shloka", "Unknown")),
            "sanskrit_text": sn,
            "english_text": en,
            "source_file": str(csv_path),
            "row_index": index,
            "page_content_format": "sn\\n\\nen",
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents, skipped_empty_sn, skipped_empty_en, len(df)


def build_or_load_combined_index(allow_rebuild: bool, embeddings) -> Tuple[FAISS, bool, int]:
    assert_safe_output_path()
    if index_exists(COMBINED_INDEX_PATH):
        print(f"Loading existing Sanskrit+English index from {COMBINED_INDEX_PATH}", flush=True)
        store = FAISS.load_local(
            folder_path=str(COMBINED_INDEX_PATH),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        n = int(store.index.ntotal)
        print(f"Loaded ntotal={n}", flush=True)
        return store, False, n

    if not allow_rebuild:
        print(f"Sanskrit+English index missing at {COMBINED_INDEX_PATH}")
        print("Re-run with --allow-rebuild to embed sn+en into a NEW index.")
        print(f"Will not touch {SANSKRIT_INDEX}")
        print(f"Will not touch {ENGLISH_INDEX}")
        sys.exit(2)

    csv_path = Path(Config.DATA_PATH)
    print(f"Building Sanskrit+English documents from {csv_path}", flush=True)
    print(f"page_content format: cleaned_sn + {PAGE_CONTENT_SEPARATOR!r} + cleaned_en", flush=True)
    print("ONE vector per CSV row (not two).", flush=True)
    docs, skip_sn, skip_en, nrows = build_combined_documents(csv_path)
    print(f"CSV rows: {nrows}", flush=True)
    print(f"Combined documents: {len(docs)}", flush=True)
    print(f"Skipped empty sn: {skip_sn}", flush=True)
    print(f"Skipped empty en: {skip_en}", flush=True)
    print(f"Index path: {COMBINED_INDEX_PATH}", flush=True)
    if not docs:
        raise RuntimeError("No combined documents to index")
    if abs(len(docs) - EXPECTED_NTOTAL) > 5:
        raise RuntimeError(f"Unexpected doc count {len(docs)}; expected ~{EXPECTED_NTOTAL}")

    # Duplicate row_index check before embedding
    row_ids = [int(d.metadata["row_index"]) for d in docs]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("Duplicate row_index in combined documents before build")

    COMBINED_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_b = (len(docs) - 1) // BATCH_SIZE + 1
    print(
        f"FAISS.from_documents batch 1/{total_b} (size={min(BATCH_SIZE, len(docs))}) "
        f"elapsed=0.0s",
        flush=True,
    )
    store = FAISS.from_documents(docs[:BATCH_SIZE], embeddings)
    store.save_local(str(COMBINED_INDEX_PATH))
    print(
        f"  checkpoint ntotal={store.index.ntotal} saved after batch 1 "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    for start in range(BATCH_SIZE, len(docs), BATCH_SIZE):
        batch = docs[start : start + BATCH_SIZE]
        batch_i = start // BATCH_SIZE + 1
        print(
            f"add_documents batch {batch_i}/{total_b} "
            f"({start + 1}-{start + len(batch)} / {len(docs)}) "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        store.add_documents(batch)
        store.save_local(str(COMBINED_INDEX_PATH))
        print(
            f"  checkpoint ntotal={store.index.ntotal} saved after batch {batch_i} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    print(
        f"Sanskrit+English index saved ntotal={store.index.ntotal} in {time.time() - t0:.1f}s",
        flush=True,
    )
    return store, True, int(store.index.ntotal)


def collect_row_indices(store: FAISS) -> List[int]:
    found: List[int] = []
    docs_by_id = getattr(store.docstore, "_dict", None)
    if isinstance(docs_by_id, dict):
        iterable = docs_by_id.values()
    else:
        iterable = [store.docstore.search(did) for did in store.index_to_docstore_id.values()]
    for doc in iterable:
        rid = baseline.row_index_from_doc(doc)
        if rid is not None:
            found.append(rid)
    return found


def validate_combined_index(store: FAISS, gold_ids: Sequence[int]) -> None:
    ntotal = int(store.index.ntotal)
    if abs(ntotal - EXPECTED_NTOTAL) > 5:
        raise RuntimeError(f"combined: unexpected ntotal={ntotal}, expected ~{EXPECTED_NTOTAL}")
    row_ids = collect_row_indices(store)
    if len(row_ids) != ntotal:
        raise RuntimeError(
            f"combined: row_index missing on some docs ({len(row_ids)}/{ntotal})"
        )
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("combined: duplicate row_index values in index")
    missing = [g for g in gold_ids if int(g) not in set(row_ids)]
    if missing:
        raise RuntimeError(f"combined: gold row_index missing: {missing[:20]}")

    # Spot-check that page_content contains both Sanskrit and English
    sample_ids = [0, ntotal // 2, ntotal - 1]
    for sid in sample_ids:
        did = store.index_to_docstore_id[sid]
        doc = store.docstore.search(did)
        pc = doc.page_content or ""
        if PAGE_CONTENT_SEPARATOR not in pc:
            raise RuntimeError(f"combined: doc {sid} missing separator in page_content")
        sn_meta = str(doc.metadata.get("sanskrit_text", ""))
        en_meta = str(doc.metadata.get("english_text", ""))
        if sn_meta and sn_meta not in pc:
            raise RuntimeError(f"combined: doc {sid} sanskrit_text not in page_content")
        if en_meta and en_meta not in pc:
            raise RuntimeError(f"combined: doc {sid} english_text not in page_content")
        # Devanagari presence check for sn half
        parts = pc.split(PAGE_CONTENT_SEPARATOR, 1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise RuntimeError(f"combined: doc {sid} empty sn or en half")
    print(
        f"OK  combined: ntotal={ntotal}, unique row_index={len(set(row_ids))}, "
        f"sn+en present in sampled page_content",
        flush=True,
    )


def first_relevant_rank(hits: Sequence[Tuple[Any, float]], gold_ids: Sequence[int]) -> Optional[int]:
    gold = set(int(g) for g in gold_ids)
    for i, (doc, _dist) in enumerate(hits, start=1):
        rid = baseline.row_index_from_doc(doc)
        if rid is not None and rid in gold:
            return i
    return None


def hit_rows(hits: Sequence[Tuple[Any, float]], n: int) -> List[Optional[int]]:
    return [baseline.row_index_from_doc(doc) for doc, _ in hits[:n]]


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
    return {
        "index_name": name,
        "elapsed_sec": time.time() - t0,
        "overall": baseline.metrics_from_ranks(first_ranks, EVAL_K),
        "rows": rows,
        "first_ranks": first_ranks,
    }


def group_metrics(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    buckets: Dict[str, List[Optional[int]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key) or "unknown")].append(row["first_relevant_rank"])
    return {name: baseline.metrics_from_ranks(ranks, EVAL_K) for name, ranks in sorted(buckets.items())}


def load_english_only_per_question() -> Dict[str, Dict[str, Any]]:
    if not ENGLISH_ONLY_SUMMARY.is_file():
        raise FileNotFoundError(f"Missing English-only summary for comparison: {ENGLISH_ONLY_SUMMARY}")
    data = json.loads(ENGLISH_ONLY_SUMMARY.read_text(encoding="utf-8"))
    return {row["question_id"]: row for row in data["per_question"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-rebuild", action="store_true")
    args = parser.parse_args()

    if Config.DEMO_MODE:
        print("ERROR: SANSKRITY_DEMO is set. Unset it.")
        return 2
    if Config.EMBEDDINGS_MODEL != "intfloat/multilingual-e5-large":
        print(f"STOP: unexpected embedding model {Config.EMBEDDINGS_MODEL}")
        return 2

    print("=" * 70, flush=True)
    print("SANSKRIT + ENGLISH FAISS — document representation experiment", flush=True)
    print("=" * 70, flush=True)
    print("page_content = cleaned_sn + '\\n\\n' + cleaned_en  (ONE vector/row)", flush=True)
    print("Unchanged: e5-large, no prefixes, no normalization, IndexFlatL2, frozen 100.", flush=True)
    print("")

    # Fingerprint protected indexes BEFORE any work
    for label, path in (("sanskrit", SANSKRIT_INDEX), ("english", ENGLISH_INDEX)):
        if not index_exists(path):
            print(f"STOP: protected {label} index missing at {path}")
            return 2
    skt_before = file_fingerprint(SANSKRIT_INDEX / "index.faiss")
    eng_before = file_fingerprint(ENGLISH_INDEX / "index.faiss")
    print(f"OK  Sanskrit index fingerprint size={skt_before['size']} mtime={skt_before['mtime_human']}", flush=True)
    print(f"OK  English index fingerprint size={eng_before['size']} mtime={eng_before['mtime_human']}", flush=True)

    if not FROZEN_PATH.is_file():
        print(f"STOP: missing frozen dataset {FROZEN_PATH}")
        return 2
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    if not frozen.get("frozen"):
        print("STOP: frozen JSON missing frozen=true")
        return 2
    questions = frozen["questions"]
    if len(questions) != 100:
        print(f"STOP: expected 100 questions, got {len(questions)}")
        return 2
    gold_ids = sorted({int(x) for q in questions for x in q["relevant_row_indices"]})
    print(f"OK  frozen 100 loaded from {FROZEN_PATH}", flush=True)

    print(f"Embedding constructor: HuggingFaceEmbeddings(model_name={Config.EMBEDDINGS_MODEL})", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDINGS_MODEL)
    print("skt+eng-eval: embeddings loaded", flush=True)

    store, rebuilt, ntotal = build_or_load_combined_index(args.allow_rebuild, embeddings)
    print(f"Indexed documents: {ntotal}", flush=True)
    print(f"Index rebuilt this run: {rebuilt}", flush=True)

    try:
        validate_combined_index(store, gold_ids)
    except Exception as err:
        print(f"STOP: combined index validation failed: {err}")
        return 2

    # Confirm protected indexes unchanged
    skt_after = file_fingerprint(SANSKRIT_INDEX / "index.faiss")
    eng_after = file_fingerprint(ENGLISH_INDEX / "index.faiss")
    if skt_before != skt_after:
        print("STOP: Sanskrit index changed during this run")
        return 2
    if eng_before != eng_after:
        print("STOP: English index changed during this run")
        return 2
    print("OK  protected Sanskrit and English indexes unchanged", flush=True)

    print("Evaluating frozen 100 on Sanskrit+English...", flush=True)
    combined = evaluate_index("sanskrit_english", store, questions)
    eng_map = load_english_only_per_question()

    per_question: List[Dict[str, Any]] = []
    both: List[str] = []
    eng_only: List[str] = []
    comb_only: List[str] = []
    neither: List[str] = []
    eng_ok_comb_fail: List[str] = []
    comb_ok_eng_fail: List[str] = []
    rank_changed_a_lot: List[Dict[str, Any]] = []

    for c_row, q in zip(combined["rows"], questions):
        qid = q["question_id"]
        e_row = eng_map[qid]
        c_rank = c_row["first_relevant_rank"]
        e_rank = e_row["english_first_relevant_rank"]
        c_ok = bool(c_rank is not None and c_rank <= EVAL_K)
        e_ok = bool(e_rank is not None and e_rank <= EVAL_K)
        if c_ok and e_ok:
            both.append(qid)
            if e_rank is not None and c_rank is not None and abs(e_rank - c_rank) >= 3:
                rank_changed_a_lot.append(
                    {"question_id": qid, "english_rank": e_rank, "combined_rank": c_rank}
                )
        elif e_ok:
            eng_only.append(qid)
            eng_ok_comb_fail.append(qid)
        elif c_ok:
            comb_only.append(qid)
            comb_ok_eng_fail.append(qid)
        else:
            neither.append(qid)
        per_question.append(
            {
                "question_id": qid,
                "question": q["question"],
                "book": q["book"],
                "question_category": q.get("question_category"),
                "gold_row_indices": [int(x) for x in q["relevant_row_indices"]],
                "english_first_relevant_rank": e_rank,
                "combined_first_relevant_rank": c_rank,
                "combined_top10_row_index": c_row["top10_row_index"],
                "english_top10_row_index": e_row.get("english_top10_row_index"),
                "english_success": e_ok,
                "combined_success": c_ok,
            }
        )

    c_m = combined["overall"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR = RESULTS_PARENT / f"run_{stamp}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": "realistic_eval_100_sanskrit_english",
        "interpretation": (
            "Document-representation experiment only. Same frozen 100 questions, "
            "same e5-large / no prefixes / no normalization / L2. "
            "Do not declare combined better solely because it contains more text."
        ),
        "page_content_format": "cleaned_sn + '\\n\\n' + cleaned_en",
        "one_vector_per_row": True,
        "frozen_path": str(FROZEN_PATH),
        "combined_index_path": str(COMBINED_INDEX_PATH.resolve()),
        "sanskrit_index_path": str(SANSKRIT_INDEX),
        "english_index_path": str(ENGLISH_INDEX),
        "combined_index_ntotal": ntotal,
        "index_rebuilt_this_run": rebuilt,
        "protected_indexes_unchanged": True,
        "embeddings_model": Config.EMBEDDINGS_MODEL,
        "prefixes": "none",
        "normalization": False,
        "distance": "L2 / Euclidean (lower is closer). NOT cosine similarity.",
        "eval_k": EVAL_K,
        "n": len(questions),
        "sanskrit_only": SANSKRIT_ONLY_METRICS,
        "english_only": ENGLISH_ONLY_METRICS,
        "sanskrit_english": c_m,
        "difference_combined_minus_english": {
            "recall_at_1": c_m["recall_at_1"] - ENGLISH_ONLY_METRICS["recall_at_1"],
            "recall_at_5": c_m["recall_at_5"] - ENGLISH_ONLY_METRICS["recall_at_5"],
            "recall_at_10": c_m["recall_at_10"] - ENGLISH_ONLY_METRICS["recall_at_10"],
            "mrr": c_m["mrr"] - ENGLISH_ONLY_METRICS["mrr"],
        },
        "combined_n_top1": sum(1 for r in combined["first_ranks"] if r is not None and r <= 1),
        "combined_n_top5": sum(1 for r in combined["first_ranks"] if r is not None and r <= 5),
        "combined_n_top10": sum(1 for r in combined["first_ranks"] if r is not None and r <= 10),
        "found_by_both_english_and_combined": both,
        "english_only_successes": eng_only,
        "combined_only_successes": comb_only,
        "neither_successes": neither,
        "cases_english_ok_combined_fail": eng_ok_comb_fail,
        "cases_combined_ok_english_fail": comb_ok_eng_fail,
        "cases_both_ok_rank_delta_ge_3": rank_changed_a_lot,
        "by_book_combined": group_metrics(combined["rows"], "book"),
        "by_category_combined": group_metrics(combined["rows"], "question_category"),
        "combined_elapsed_sec": combined["elapsed_sec"],
        "per_question": per_question,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = RESULTS_DIR / "faiss_realistic_100_sanskrit_english_summary.json"
    detailed_path = RESULTS_DIR / f"faiss_realistic_100_sanskrit_english_detailed_{stamp}.json"
    csv_path = RESULTS_DIR / "faiss_realistic_100_sanskrit_english_per_question.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    detailed_path.write_text(
        json.dumps({"summary": summary, "combined_rows": combined["rows"]}, indent=2),
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
                "english_first_relevant_rank",
                "combined_first_relevant_rank",
                "combined_top10_row_index",
                "english_success",
                "combined_success",
            ],
        )
        w.writeheader()
        for row in per_question:
            w.writerow(
                {
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "book": row["book"],
                    "question_category": row["question_category"],
                    "gold_row_indices": "|".join(str(x) for x in row["gold_row_indices"]),
                    "english_first_relevant_rank": row["english_first_relevant_rank"],
                    "combined_first_relevant_rank": row["combined_first_relevant_rank"],
                    "combined_top10_row_index": "|".join(
                        "" if x is None else str(x) for x in row["combined_top10_row_index"]
                    ),
                    "english_success": row["english_success"],
                    "combined_success": row["combined_success"],
                }
            )

    def pct(x: float) -> str:
        return f"{100.0 * x:.2f}%"

    print("")
    print("=" * 70)
    print("THREE-WAY COMPARISON (frozen n=100)")
    print("=" * 70)
    print(f"{'Metric':12} {'Sanskrit':>12} {'English':>12} {'Skt+Eng':>12}")
    print(
        f"{'Recall@1':12} {pct(SANSKRIT_ONLY_METRICS['recall_at_1']):>12} "
        f"{pct(ENGLISH_ONLY_METRICS['recall_at_1']):>12} {pct(c_m['recall_at_1']):>12}"
    )
    print(
        f"{'Recall@5':12} {pct(SANSKRIT_ONLY_METRICS['recall_at_5']):>12} "
        f"{pct(ENGLISH_ONLY_METRICS['recall_at_5']):>12} {pct(c_m['recall_at_5']):>12}"
    )
    print(
        f"{'Recall@10':12} {pct(SANSKRIT_ONLY_METRICS['recall_at_10']):>12} "
        f"{pct(ENGLISH_ONLY_METRICS['recall_at_10']):>12} {pct(c_m['recall_at_10']):>12}"
    )
    print(
        f"{'MRR':12} {SANSKRIT_ONLY_METRICS['mrr']:.4f}{'':>6} "
        f"{ENGLISH_ONLY_METRICS['mrr']:.4f}{'':>6} {c_m['mrr']:.4f}"
    )
    print("")
    print(
        f"vs English: both={len(both)} eng_only={len(eng_only)} "
        f"comb_only={len(comb_only)} neither={len(neither)}"
    )
    print(f"Summary: {summary_path}")
    print(f"CSV:     {csv_path}")
    print(f"Detailed:{detailed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
