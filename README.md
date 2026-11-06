# Sanskriti AI

Sanskriti AI is a research project for asking questions about Sanskrit scripture and tracing answers back to source passages. It began as a local Sanskrit/English retrieval assistant and grew into a series of Bhagavad Gita retrieval and answer-generation experiments.

This repository documents the **Sanskriti research project**. The separate [BookRAG cloud application](https://bookrag.vercel.app/) is a later, general-PDF product and is not deployed from this repository.

## What I built and studied

- A local FastAPI application with a browser interface for scripture questions, source citations, optional file context, and speech input (`src/`, `frontend/`).
- Local semantic retrieval using `intfloat/multilingual-e5-large` embeddings and a FAISS index, with a Qwen-based answer generator (`src/`).
- Bhagavad Gita experiments comparing E5 retrieval, BM25 keyword retrieval, hybrid retrieval, citation-aware selection, and local Qwen generation (`gita/`).
- Frozen evaluation questions and saved result summaries so the reported results can be inspected without rerunning local models (`gita/*frozen.json`, `gita/*summary.json`, `gita/*report.json`).
- Additional Sanskrit/English FAISS evaluation scripts (`eval/`).

## Research evidence

The Gita scripts and saved summaries distinguish retrieval from answer generation. For example, the 40-question held-out retrieval summary reports **36/40 source-level top-1 hits for E5** and **40/40 source-level top-5 hits**. These are results on that frozen evaluation set, not a claim of general accuracy. See [`gita/gita_holdout_eval_summary.json`](gita/gita_holdout_eval_summary.json) and [`gita/gita_holdout_frozen.json`](gita/gita_holdout_frozen.json). Other summaries in `gita/` record BM25, hybrid, citation, and generation experiments.

## Repository map

| Path | Contents |
| --- | --- |
| `src/` | Local application, model loading, retrieval, citations, and API |
| `frontend/` | Browser interface for the local application |
| `gita/` | Bhagavad Gita research scripts, frozen questions, and result summaries |
| `eval/` | Earlier Sanskrit/English retrieval evaluation scripts |
| `tests/` | Unit tests for core application helpers |
| `SETUP.md` | Local setup notes and data requirements |

## Running locally

This is a research repository, not a hosted service. The local application needs the scripture corpus, a local E5 model download, a Qwen model or adapter, and a FAISS index. Large model files, indexes, raw corpus files, generated logs, and private configuration are intentionally **not** included here. See [`SETUP.md`](SETUP.md) before attempting a local run. Some historical experiment scripts also expect the saved corpus and index files and will not run from a fresh clone alone.

## Scope and limitations

The saved results describe specific datasets and local configurations. They do not establish performance on arbitrary books or production traffic. This repository preserves the research code and compact evaluation records; it does not contain the cloud PDF library, Supabase services, or the BookRAG deployment configuration. No model weights or third-party corpus files are distributed here.

## Author

Manu Mathew Jiss — computer science graduate research and development project.
