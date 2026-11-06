#!/usr/bin/env python3
"""
Sanskrit + English FAISS experiment: official multilingual-e5-large retrieval recipe.

CHANGE vs current Skt+Eng baseline (48% Recall@10):
  - Encoder prefix "passage: " on documents, "query: " on questions
  - sentence-transformers normalize_embeddings=True
  - FAISS IndexFlatIP (LangChain DistanceStrategy.MAX_INNER_PRODUCT)

UNCHANGED:
  - intfloat/multilingual-e5-large
  - page_content = cleaned_sn + "\\n\\n" + cleaned_en (prefixes NOT stored)
  - ONE vector per CSV row (~93,030)
  - frozen 100 questions / gold row_index labels
  - Top K = 10, multi-gold ANY-in-Top-K
  - no rerank / BM25 / hybrid / Qwen / frontend / production code

Does NOT write to existing Sanskrit-only, English-only, or Skt+Eng indexes.

Usage
-----
    ./venv/bin/python eval/evaluate_faiss_sanskrit_english_e5_prefixed.py --smoke
    ./venv/bin/python eval/evaluate_faiss_sanskrit_english_e5_prefixed.py --allow-rebuild
    ./venv/bin/python eval/evaluate_faiss_sanskrit_english_e5_prefixed.py   # load + eval only
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

NEW_INDEX_PATH = ROOT / "sanskrit_english_e5_prefixed_norm_faiss_index_itihasa"
PREV_COMBINED_INDEX = ROOT / "sanskrit_english_faiss_index_itihasa"
SANSKRIT_INDEX = (ROOT / "sanskrit_faiss_index_itihasa").resolve()
ENGLISH_INDEX = (ROOT / "faiss_index_itihasa_english_only").resolve()
PROJECTS_CLONE = Path("/Users/manumathewjiss/Projects/Pramod-Gupta-Project")
FROZEN_PATH = ROOT / "eval" / "data" / "realistic_eval_100_frozen.json"
PREV_COMBINED_SUMMARY = (
    ROOT
    / "eval"
    / "results"
    / "realistic_eval_100_sanskrit_english"
    / "run_20260826T205241Z"
    / "faiss_realistic_100_sanskrit_english_summary.json"
)
RESULTS_PARENT = ROOT / "eval" / "results" / "realistic_eval_100_sanskrit_english_e5_prefixed_norm"
BATCH_SIZE = 512
EXPECTED_NTOTAL = 93030
EVAL_K = 10
PAGE_CONTENT_SEPARATOR = "\n\n"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
HF_ENCODE_KWARGS = {"normalize_embeddings": True}
FAISS_NORMALIZE_L2 = False
FAISS_DISTANCE_STRATEGY_NAME = "MAX_INNER_PRODUCT"

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
PREV_SKT_ENG_METRICS = {
    "recall_at_1": 0.29,
    "recall_at_5": 0.40,
    "recall_at_10": 0.48,
    "mrr": 0.3382,
}


from langchain_core.embeddings import Embeddings


class E5PrefixedEmbeddings(Embeddings):
    """Add official E5 retrieval prefixes at encode time only."""

    def __init__(self, inner: Any):
        super().__init__()
        self.inner = inner

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [PASSAGE_PREFIX + t for t in texts]
        return self.inner.embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        return self.inner.embed_query(QUERY_PREFIX + text)


class RecordingEmbeddings:
    """Smoke-only inner encoder: records prefixed strings, returns unit vectors."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.document_inputs: List[str] = []
        self.query_inputs: List[str] = []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.document_inputs.extend(list(texts))
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        self.query_inputs.append(text)
        return self._vec(text)

    def _vec(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [((digest[i] / 255.0) - 0.5) for i in range(self.dim)]
        n = sum(x * x for x in raw) ** 0.5
        if n == 0.0:
            raw[0] = 1.0
            n = 1.0
        return [x / n for x in raw]


def faiss_index_kwargs() -> Dict[str, Any]:
    from langchain_community.vectorstores.utils import DistanceStrategy

    return {
        "distance_strategy": DistanceStrategy.MAX_INNER_PRODUCT,
        "normalize_L2": FAISS_NORMALIZE_L2,
    }


def disable_mps() -> None:
    try:
        import torch

        torch.backends.mps.is_available = lambda: False  # type: ignore[method-assign]
        torch.backends.mps.is_built = lambda: False  # type: ignore[method-assign]
        print("e5-prefixed-eval: torch ok", torch.__version__, flush=True)
    except Exception as err:
        print("e5-prefixed-eval: torch skip", err, flush=True)


def make_e5_embeddings() -> E5PrefixedEmbeddings:
    from config import Config
    from langchain_community.embeddings import HuggingFaceEmbeddings

    inner = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDINGS_MODEL,
        encode_kwargs=dict(HF_ENCODE_KWARGS),
    )
    return E5PrefixedEmbeddings(inner)


def protected_index_paths() -> List[Path]:
    names = (
        "sanskrit_faiss_index_itihasa",
        "faiss_index_itihasa_english_only",
        "sanskrit_english_faiss_index_itihasa",
    )
    paths = [SANSKRIT_INDEX, ENGLISH_INDEX, PREV_COMBINED_INDEX.resolve()]
    for name in names:
        paths.append((ROOT / name).resolve())
        paths.append((PROJECTS_CLONE / name).resolve())
    # unique, never include the new experiment path
    out: List[Path] = []
    seen = set()
    new = NEW_INDEX_PATH.resolve()
    for p in paths:
        if p == new:
            continue
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def clean_text(text: str) -> str:
    """Same whitespace/control cleaning as English-only eval / src.utils.clean_text."""
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t\r")
    return text.strip()


def assert_safe_output_path() -> None:
    out = NEW_INDEX_PATH.resolve()
    for protected in protected_index_paths():
        if out == protected:
            raise RuntimeError(f"Refusing: new index path equals protected index {protected}")
        try:
            out.relative_to(protected)
            raise RuntimeError(f"Refusing: new index path is inside protected index {protected}")
        except ValueError:
            pass
    print(f"OK  safe output path: {out}", flush=True)
    for protected in protected_index_paths():
        print(f"OK  will not touch: {protected}", flush=True)


def index_exists(path: Path) -> bool:
    return path.is_dir() and (path / "index.faiss").is_file() and (path / "index.pkl").is_file()


def file_fingerprint(path: Path) -> Dict[str, Any]:
    st = path.stat()
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


def fingerprint_existing_protected() -> Dict[str, Dict[str, Any]]:
    snaps: Dict[str, Dict[str, Any]] = {}
    for path in protected_index_paths():
        faiss_file = path / "index.faiss"
        if faiss_file.is_file():
            snaps[str(path)] = file_fingerprint(faiss_file)
    return snaps


def assert_protected_unchanged(before: Dict[str, Dict[str, Any]]) -> None:
    after = fingerprint_existing_protected()
    if before != after:
        raise RuntimeError("A protected FAISS index changed during this run")
    print("OK  protected indexes unchanged", flush=True)


def _row_to_document(csv_path: Path, index: int, row: Any):
    from langchain_core.documents import Document

    sn = clean_text(str(row.get("sn", "")))
    en = clean_text(str(row.get("en", "")))
    if not sn:
        return None, True, False
    if not en:
        return None, False, True
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
        "e5_prefix_in_page_content": False,
    }
    return Document(page_content=page_content, metadata=metadata), False, False


def build_combined_documents(csv_path: Path) -> Tuple[List[Any], int, int, int]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    documents: List[Any] = []
    skipped_empty_sn = 0
    skipped_empty_en = 0
    for index, row in df.iterrows():
        doc, skip_sn, skip_en = _row_to_document(csv_path, int(index), row)
        if skip_sn:
            skipped_empty_sn += 1
            continue
        if skip_en:
            skipped_empty_en += 1
            continue
        documents.append(doc)
    return documents, skipped_empty_sn, skipped_empty_en, len(df)


def count_combined_documents_csv(csv_path: Path) -> Tuple[int, int, int, int, Dict[str, Any]]:
    """Stdlib CSV scan used by smoke. Does not embed and does not import pandas."""
    n_rows = 0
    n_docs = 0
    skip_sn = 0
    skip_en = 0
    sample: Optional[Dict[str, Any]] = None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            n_rows += 1
            sn = clean_text(str(row.get("sn", "")))
            en = clean_text(str(row.get("en", "")))
            if not sn:
                skip_sn += 1
                continue
            if not en:
                skip_en += 1
                continue
            n_docs += 1
            if sample is None:
                sample = {
                    "row_index": index,
                    "page_content": f"{sn}{PAGE_CONTENT_SEPARATOR}{en}",
                    "sanskrit_text": sn,
                    "english_text": en,
                }
    return n_docs, skip_sn, skip_en, n_rows, sample or {}


def faiss_from_documents(docs: List[Any], embeddings) -> Any:
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents(docs, embeddings, **faiss_index_kwargs())


def load_experiment_index(embeddings) -> Any:
    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(
        folder_path=str(NEW_INDEX_PATH),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
        **faiss_index_kwargs(),
    )


def assert_indexflat_ip(store: Any) -> None:
    from langchain_community.vectorstores.utils import DistanceStrategy

    idx = store.index
    name = type(idx).__name__
    if "IndexFlatIP" not in name:
        raise RuntimeError(f"Expected IndexFlatIP, got {name}")
    if store.distance_strategy != DistanceStrategy.MAX_INNER_PRODUCT:
        raise RuntimeError(f"Expected MAX_INNER_PRODUCT, got {store.distance_strategy}")
    if getattr(store, "_normalize_L2", None):
        raise RuntimeError("FAISS normalize_L2 must be False (normalization is in encode_kwargs)")
    print(f"OK  FAISS metric: {name} / {store.distance_strategy}", flush=True)


def build_or_load_index(allow_rebuild: bool, embeddings) -> Tuple[Any, bool, int]:
    from config import Config

    assert_safe_output_path()
    if index_exists(NEW_INDEX_PATH):
        print(f"Loading existing E5-prefixed index from {NEW_INDEX_PATH}", flush=True)
        store = load_experiment_index(embeddings)
        assert_indexflat_ip(store)
        n = int(store.index.ntotal)
        print(f"Loaded ntotal={n}", flush=True)
        if abs(n - EXPECTED_NTOTAL) <= 5:
            return store, False, n
        if not allow_rebuild:
            print(
                f"STOP: existing new-experiment index is incomplete "
                f"(ntotal={n}, expected ~{EXPECTED_NTOTAL}). Re-run with --allow-rebuild."
            )
            sys.exit(2)
        print(
            f"Incomplete new-experiment index ntotal={n}; rebuilding this experiment "
            f"folder only (protected indexes untouched).",
            flush=True,
        )

    if not allow_rebuild:
        print(f"E5-prefixed index missing at {NEW_INDEX_PATH}")
        print("Re-run with --allow-rebuild to embed sn+en with E5 prefixes + normalization.")
        print("Will not touch protected indexes.")
        sys.exit(2)

    csv_path = Path(Config.DATA_PATH)
    print(f"Building Sanskrit+English documents from {csv_path}", flush=True)
    print(f"page_content format: cleaned_sn + {PAGE_CONTENT_SEPARATOR!r} + cleaned_en", flush=True)
    print("Prefixes applied only at encode time, not stored in page_content.", flush=True)
    print("ONE vector per CSV row (not two).", flush=True)
    docs, skip_sn, skip_en, nrows = build_combined_documents(csv_path)
    print(f"CSV rows: {nrows}", flush=True)
    print(f"Combined documents: {len(docs)}", flush=True)
    print(f"Skipped empty sn: {skip_sn}", flush=True)
    print(f"Skipped empty en: {skip_en}", flush=True)
    print(f"Index path: {NEW_INDEX_PATH}", flush=True)
    if not docs:
        raise RuntimeError("No combined documents to index")
    if abs(len(docs) - EXPECTED_NTOTAL) > 5:
        raise RuntimeError(f"Unexpected doc count {len(docs)}; expected ~{EXPECTED_NTOTAL}")

    row_ids = [int(d.metadata["row_index"]) for d in docs]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("Duplicate row_index in combined documents before build")

    NEW_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_b = (len(docs) - 1) // BATCH_SIZE + 1
    print(
        f"FAISS.from_documents batch 1/{total_b} (size={min(BATCH_SIZE, len(docs))}) "
        f"elapsed=0.0s metric=IndexFlatIP",
        flush=True,
    )
    store = faiss_from_documents(docs[:BATCH_SIZE], embeddings)
    assert_indexflat_ip(store)
    store.save_local(str(NEW_INDEX_PATH))
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
        store.save_local(str(NEW_INDEX_PATH))
        print(
            f"  checkpoint ntotal={store.index.ntotal} saved after batch {batch_i} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    print(
        f"E5-prefixed index saved ntotal={store.index.ntotal} in {time.time() - t0:.1f}s",
        flush=True,
    )
    return store, True, int(store.index.ntotal)


def collect_row_indices(store: Any) -> List[int]:
    import evaluate_faiss_baseline as baseline

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


def validate_combined_index(store: Any, gold_ids: Sequence[int]) -> None:
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

    sample_ids = [0, ntotal // 2, ntotal - 1]
    for sid in sample_ids:
        did = store.index_to_docstore_id[sid]
        doc = store.docstore.search(did)
        pc = doc.page_content or ""
        if pc.startswith(PASSAGE_PREFIX) or pc.startswith(QUERY_PREFIX):
            raise RuntimeError(f"combined: doc {sid} stored a prefix in page_content")
        if PAGE_CONTENT_SEPARATOR not in pc:
            raise RuntimeError(f"combined: doc {sid} missing separator in page_content")
        sn_meta = str(doc.metadata.get("sanskrit_text", ""))
        en_meta = str(doc.metadata.get("english_text", ""))
        if sn_meta and sn_meta not in pc:
            raise RuntimeError(f"combined: doc {sid} sanskrit_text not in page_content")
        if en_meta and en_meta not in pc:
            raise RuntimeError(f"combined: doc {sid} english_text not in page_content")
        parts = pc.split(PAGE_CONTENT_SEPARATOR, 1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise RuntimeError(f"combined: doc {sid} empty sn or en half")
    print(
        f"OK  combined: ntotal={ntotal}, unique row_index={len(set(row_ids))}, "
        f"sn+en present, prefixes not stored in page_content",
        flush=True,
    )


def first_relevant_rank(hits: Sequence[Tuple[Any, float]], gold_ids: Sequence[int]) -> Optional[int]:
    import evaluate_faiss_baseline as baseline

    gold = set(int(g) for g in gold_ids)
    for i, (doc, _dist) in enumerate(hits, start=1):
        rid = baseline.row_index_from_doc(doc)
        if rid is not None and rid in gold:
            return i
    return None


def hit_rows(hits: Sequence[Tuple[Any, float]], n: int) -> List[Optional[int]]:
    import evaluate_faiss_baseline as baseline

    return [baseline.row_index_from_doc(doc) for doc, _ in hits[:n]]


def evaluate_index(name: str, store: Any, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    import evaluate_faiss_baseline as baseline
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
                "top10_inner_product": [float(d) for _doc, d in hits[:10]],
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
    import evaluate_faiss_baseline as baseline

    buckets: Dict[str, List[Optional[int]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key) or "unknown")].append(row["first_relevant_rank"])
    return {name: baseline.metrics_from_ranks(ranks, EVAL_K) for name, ranks in sorted(buckets.items())}


def load_frozen_questions() -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[int]]:
    if not FROZEN_PATH.is_file():
        raise FileNotFoundError(f"Missing frozen dataset {FROZEN_PATH}")
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    if not frozen.get("frozen"):
        raise RuntimeError("frozen JSON missing frozen=true")
    questions = frozen["questions"]
    if len(questions) != 100:
        raise RuntimeError(f"expected 100 questions, got {len(questions)}")
    gold_ids = sorted({int(x) for q in questions for x in q["relevant_row_indices"]})
    return frozen, questions, gold_ids


def run_smoke() -> int:
    print("=" * 70, flush=True)
    print("SMOKE: E5 prefixed + normalized Skt+Eng FAISS (no full index build)", flush=True)
    print("=" * 70, flush=True)

    before = fingerprint_existing_protected()
    assert_safe_output_path()
    if NEW_INDEX_PATH.resolve() in {p.resolve() for p in protected_index_paths()}:
        raise RuntimeError("new index path collides with a protected index")
    if RESULTS_PARENT.resolve() == (
        ROOT / "eval" / "results" / "realistic_eval_100_sanskrit_english"
    ).resolve():
        raise RuntimeError("results path collides with previous Skt+Eng results")
    print(f"OK  new index path: {NEW_INDEX_PATH.resolve()}", flush=True)
    print(f"OK  new results parent: {RESULTS_PARENT.resolve()}", flush=True)

    if HF_ENCODE_KWARGS != {"normalize_embeddings": True}:
        raise RuntimeError(f"unexpected HF_ENCODE_KWARGS={HF_ENCODE_KWARGS}")
    if FAISS_DISTANCE_STRATEGY_NAME != "MAX_INNER_PRODUCT":
        raise RuntimeError("FAISS distance strategy name must be MAX_INNER_PRODUCT")
    if FAISS_NORMALIZE_L2 is not False:
        raise RuntimeError("FAISS normalize_L2 must be False")
    print("OK  encode_kwargs normalize_embeddings=True", flush=True)
    print("OK  FAISS kwargs planned: MAX_INNER_PRODUCT, normalize_L2=False", flush=True)

    _frozen, questions, gold_ids = load_frozen_questions()
    q0 = questions[0]["question"]
    if q0.startswith(QUERY_PREFIX) or q0.startswith(PASSAGE_PREFIX):
        raise RuntimeError("frozen questions already contain E5 prefixes")
    print(f"OK  frozen 100 loaded from {FROZEN_PATH} gold_ids={len(gold_ids)}", flush=True)
    print(f"OK  EVAL_K={EVAL_K}", flush=True)

    csv_path = ROOT / "res" / "combined.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV missing: {csv_path}")
    n_docs, skip_sn, skip_en, nrows, sample = count_combined_documents_csv(csv_path)
    if abs(n_docs - EXPECTED_NTOTAL) > 5:
        raise RuntimeError(f"doc count {n_docs} != ~{EXPECTED_NTOTAL}")
    pc = str(sample["page_content"])
    if pc.startswith(PASSAGE_PREFIX) or pc.startswith(QUERY_PREFIX):
        raise RuntimeError("page_content must not store E5 prefixes")
    if PAGE_CONTENT_SEPARATOR not in pc:
        raise RuntimeError("missing sn/en separator")
    print(
        f"OK  one vector/row: csv={nrows} docs={n_docs} skip_sn={skip_sn} skip_en={skip_en}",
        flush=True,
    )

    recorder = RecordingEmbeddings(dim=8)
    wrapped = E5PrefixedEmbeddings(recorder)
    doc_text = pc
    query_text = questions[0]["question"]
    wrapped.embed_documents([doc_text, "second verse"])
    wrapped.embed_query(query_text)
    if recorder.document_inputs != [PASSAGE_PREFIX + doc_text, PASSAGE_PREFIX + "second verse"]:
        raise RuntimeError(f"passage prefix mismatch: {recorder.document_inputs}")
    if any(t.startswith(QUERY_PREFIX) for t in recorder.document_inputs):
        raise RuntimeError("a document embed received query: prefix")
    if recorder.query_inputs != [QUERY_PREFIX + query_text]:
        raise RuntimeError(f"query prefix mismatch: {recorder.query_inputs}")
    if recorder.query_inputs[0].startswith(PASSAGE_PREFIX):
        raise RuntimeError("query received passage prefix")
    print("OK  passage: prefix applied only to documents", flush=True)
    print("OK  query: prefix applied only to queries", flush=True)

    faiss_src = (
        ROOT
        / "venv"
        / "lib"
        / "python3.13"
        / "site-packages"
        / "langchain_community"
        / "vectorstores"
        / "faiss.py"
    )
    if not faiss_src.is_file():
        raise FileNotFoundError(f"Cannot inspect LangChain FAISS wrapper at {faiss_src}")
    wrapper = faiss_src.read_text(encoding="utf-8")
    if "if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:" not in wrapper:
        raise RuntimeError("Installed LangChain FAISS wrapper missing MAX_INNER_PRODUCT branch")
    if "index = faiss.IndexFlatIP(len(embeddings[0]))" not in wrapper:
        raise RuntimeError("Installed LangChain FAISS wrapper does not construct IndexFlatIP")
    print(
        "OK  installed LangChain FAISS: MAX_INNER_PRODUCT -> IndexFlatIP "
        "(no live numpy/FAISS import in smoke)",
        flush=True,
    )

    if index_exists(NEW_INDEX_PATH):
        raise RuntimeError("smoke must not create the full experiment index")
    assert_protected_unchanged(before)
    print("OK  smoke did not write the full 93,030 index", flush=True)
    print("SMOKE PASSED", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-rebuild", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Lightweight checks only. Does not embed 93,030 documents or write the experiment index.",
    )
    args = parser.parse_args()

    if args.smoke:
        return run_smoke()

    from config import Config

    disable_mps()
    if Config.DEMO_MODE:
        print("ERROR: SANSKRITY_DEMO is set. Unset it.")
        return 2
    if Config.EMBEDDINGS_MODEL != "intfloat/multilingual-e5-large":
        print(f"STOP: unexpected embedding model {Config.EMBEDDINGS_MODEL}")
        return 2

    print("=" * 70, flush=True)
    print("SANSKRIT + ENGLISH FAISS — E5 prefixes + normalization experiment", flush=True)
    print("=" * 70, flush=True)
    print("page_content = cleaned_sn + '\\n\\n' + cleaned_en  (ONE vector/row)", flush=True)
    print("encode: passage:/query: prefixes, normalize_embeddings=True, IndexFlatIP", flush=True)
    print("")

    before = fingerprint_existing_protected()
    frozen, questions, gold_ids = load_frozen_questions()
    print(f"OK  frozen 100 loaded from {FROZEN_PATH}", flush=True)

    print(
        f"Embedding constructor: HuggingFaceEmbeddings("
        f"model_name={Config.EMBEDDINGS_MODEL}, encode_kwargs={HF_ENCODE_KWARGS}) "
        f"+ E5PrefixedEmbeddings",
        flush=True,
    )
    embeddings = make_e5_embeddings()
    print("e5-prefixed-eval: embeddings loaded", flush=True)

    store, rebuilt, ntotal = build_or_load_index(args.allow_rebuild, embeddings)
    print(f"Indexed documents: {ntotal}", flush=True)
    print(f"Index rebuilt this run: {rebuilt}", flush=True)

    try:
        validate_combined_index(store, gold_ids)
        assert_indexflat_ip(store)
    except Exception as err:
        print(f"STOP: combined index validation failed: {err}")
        return 2

    assert_protected_unchanged(before)

    print("Evaluating frozen 100 on E5-prefixed Sanskrit+English...", flush=True)
    combined = evaluate_index("sanskrit_english_e5_prefixed", store, questions)

    prev_map: Dict[str, Dict[str, Any]] = {}
    if PREV_COMBINED_SUMMARY.is_file():
        prev = json.loads(PREV_COMBINED_SUMMARY.read_text(encoding="utf-8"))
        prev_map = {row["question_id"]: row for row in prev.get("per_question", [])}

    per_question: List[Dict[str, Any]] = []
    both: List[str] = []
    prev_only: List[str] = []
    new_only: List[str] = []
    neither: List[str] = []

    for c_row, q in zip(combined["rows"], questions):
        qid = q["question_id"]
        c_rank = c_row["first_relevant_rank"]
        p_row = prev_map.get(qid, {})
        p_rank = p_row.get("combined_first_relevant_rank")
        c_ok = bool(c_rank is not None and c_rank <= EVAL_K)
        p_ok = bool(p_rank is not None and p_rank <= EVAL_K) if p_rank is not None else None
        if p_ok is True and c_ok:
            both.append(qid)
        elif p_ok is True and not c_ok:
            prev_only.append(qid)
        elif p_ok is False and c_ok:
            new_only.append(qid)
        elif p_ok is False and not c_ok:
            neither.append(qid)
        per_question.append(
            {
                "question_id": qid,
                "question": q["question"],
                "book": q["book"],
                "question_category": q.get("question_category"),
                "gold_row_indices": [int(x) for x in q["relevant_row_indices"]],
                "prev_skt_eng_first_relevant_rank": p_rank,
                "e5_prefixed_first_relevant_rank": c_rank,
                "e5_prefixed_top10_row_index": c_row["top10_row_index"],
                "prev_skt_eng_success": p_ok,
                "e5_prefixed_success": c_ok,
            }
        )

    c_m = combined["overall"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR = RESULTS_PARENT / f"run_{stamp}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": "realistic_eval_100_sanskrit_english_e5_prefixed_norm",
        "interpretation": (
            "Official multilingual-e5-large retrieval recipe on the same Skt+Eng "
            "documents and frozen 100 questions: query:/passage: prefixes, "
            "normalize_embeddings=True, FAISS IndexFlatIP. Bundled recipe test, "
            "not a single-variable ablation."
        ),
        "page_content_format": "cleaned_sn + '\\n\\n' + cleaned_en",
        "one_vector_per_row": True,
        "frozen_path": str(FROZEN_PATH),
        "index_path": str(NEW_INDEX_PATH.resolve()),
        "index_ntotal": ntotal,
        "index_rebuilt_this_run": rebuilt,
        "protected_indexes_unchanged": True,
        "embeddings_model": Config.EMBEDDINGS_MODEL,
        "prefixes": "query: on questions; passage: on documents (encode-time only)",
        "normalization": True,
        "normalization_impl": "HuggingFaceEmbeddings encode_kwargs normalize_embeddings=True",
        "faiss_normalize_L2": False,
        "distance": "IndexFlatIP / MAX_INNER_PRODUCT (higher is closer). Unit vectors so IP = cosine.",
        "eval_k": EVAL_K,
        "n": len(questions),
        "sanskrit_only": SANSKRIT_ONLY_METRICS,
        "english_only": ENGLISH_ONLY_METRICS,
        "sanskrit_english_no_prefix_l2": PREV_SKT_ENG_METRICS,
        "sanskrit_english_e5_prefixed_norm": c_m,
        "difference_e5_minus_prev_combined": {
            "recall_at_1": c_m["recall_at_1"] - PREV_SKT_ENG_METRICS["recall_at_1"],
            "recall_at_5": c_m["recall_at_5"] - PREV_SKT_ENG_METRICS["recall_at_5"],
            "recall_at_10": c_m["recall_at_10"] - PREV_SKT_ENG_METRICS["recall_at_10"],
            "mrr": c_m["mrr"] - PREV_SKT_ENG_METRICS["mrr"],
        },
        "e5_n_top1": sum(1 for r in combined["first_ranks"] if r is not None and r <= 1),
        "e5_n_top5": sum(1 for r in combined["first_ranks"] if r is not None and r <= 5),
        "e5_n_top10": sum(1 for r in combined["first_ranks"] if r is not None and r <= 10),
        "found_by_both_prev_and_e5": both,
        "prev_only_successes": prev_only,
        "e5_only_successes": new_only,
        "neither_successes": neither,
        "by_book": group_metrics(combined["rows"], "book"),
        "by_category": group_metrics(combined["rows"], "question_category"),
        "elapsed_sec": combined["elapsed_sec"],
        "per_question": per_question,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = RESULTS_DIR / "faiss_realistic_100_sanskrit_english_e5_prefixed_norm_summary.json"
    detailed_path = RESULTS_DIR / f"faiss_realistic_100_sanskrit_english_e5_prefixed_norm_detailed_{stamp}.json"
    csv_path = RESULTS_DIR / "faiss_realistic_100_sanskrit_english_e5_prefixed_norm_per_question.csv"
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
                "prev_skt_eng_first_relevant_rank",
                "e5_prefixed_first_relevant_rank",
                "e5_prefixed_top10_row_index",
                "prev_skt_eng_success",
                "e5_prefixed_success",
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
                    "prev_skt_eng_first_relevant_rank": row["prev_skt_eng_first_relevant_rank"],
                    "e5_prefixed_first_relevant_rank": row["e5_prefixed_first_relevant_rank"],
                    "e5_prefixed_top10_row_index": "|".join(
                        "" if x is None else str(x) for x in row["e5_prefixed_top10_row_index"]
                    ),
                    "prev_skt_eng_success": row["prev_skt_eng_success"],
                    "e5_prefixed_success": row["e5_prefixed_success"],
                }
            )

    def pct(x: float) -> str:
        return f"{100.0 * x:.2f}%"

    print("")
    print("=" * 70)
    print("COMPARISON (frozen n=100)")
    print("=" * 70)
    print(f"{'Metric':12} {'Sanskrit':>12} {'English':>12} {'Skt+Eng':>12} {'E5 recipe':>12}")
    print(
        f"{'Recall@1':12} {pct(SANSKRIT_ONLY_METRICS['recall_at_1']):>12} "
        f"{pct(ENGLISH_ONLY_METRICS['recall_at_1']):>12} {pct(PREV_SKT_ENG_METRICS['recall_at_1']):>12} "
        f"{pct(c_m['recall_at_1']):>12}"
    )
    print(
        f"{'Recall@5':12} {pct(SANSKRIT_ONLY_METRICS['recall_at_5']):>12} "
        f"{pct(ENGLISH_ONLY_METRICS['recall_at_5']):>12} {pct(PREV_SKT_ENG_METRICS['recall_at_5']):>12} "
        f"{pct(c_m['recall_at_5']):>12}"
    )
    print(
        f"{'Recall@10':12} {pct(SANSKRIT_ONLY_METRICS['recall_at_10']):>12} "
        f"{pct(ENGLISH_ONLY_METRICS['recall_at_10']):>12} {pct(PREV_SKT_ENG_METRICS['recall_at_10']):>12} "
        f"{pct(c_m['recall_at_10']):>12}"
    )
    print(
        f"{'MRR':12} {SANSKRIT_ONLY_METRICS['mrr']:.4f}{'':>6} "
        f"{ENGLISH_ONLY_METRICS['mrr']:.4f}{'':>6} {PREV_SKT_ENG_METRICS['mrr']:.4f}{'':>6} {c_m['mrr']:.4f}"
    )
    print("")
    print(f"Summary: {summary_path}")
    print(f"CSV:     {csv_path}")
    print(f"Detailed:{detailed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
