# Sanskrity Setup Guide

> **Archive note:** This public repository contains source code and compact research results. It does not distribute the scripture corpus (`res/combined.csv`), local Qwen weights/adapters, or FAISS indexes. The historical steps below require you to supply those files and may require changes for your operating system. The local application is separate from the deployed BookRAG cloud product.

## Prerequisites

- Python 3.8+
- 10 GB+ disk space (model weights)
- 8 GB+ RAM
- CUDA-capable GPU (recommended — CPU works but is slow)

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root (already gitignored):

```
MONGODB_URL=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
```

> MongoDB is optional. The server runs fully locally without it. Only needed if you use the legacy MongoDB migration scripts.

### 3. Start the server

```bash
python src/server.py
```

On first run with no existing FAISS index the server will:
1. Load Whisper STT model (`tiny`)
2. Load the embedding model (`multilingual-e5-large`, 1024-dim)
3. Build the FAISS index from `res/combined.csv` — **this only happens once**, the index is saved to `sanskrit_faiss_index_itihasa/`
4. Load the LLM (Qwen fine-tuned with LoRA, 4-bit quantized)
5. Start on `http://localhost:8000`

Subsequent starts skip step 3 and load the saved index directly.

### 4. Open the interface

```
http://localhost:8000
```

API docs: `http://localhost:8000/docs`

---

## Development (auto-reload)

```bash
uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
```

---

## Architecture

```
User Query (text or voice)
  ↓
[EN/Sanskrit script filter — rejects unsupported scripts]
  ↓
FAISS semantic search (local index, multilingual-e5-large embeddings)
  ↓
TextRank-weighted reranker
  (PageRank over shloka similarity graph × query cosine similarity)
  ↓
User context search (in-memory cosine similarity)
  ↓
LLM prompt (Qwen fine-tuned, 4-bit quantized)
  ↓
Response sanitizer (EN/Sanskrit output filter, repetition collapse, truncation)
  ↓
Answer + citation (book, chapter, shloka)
```

### Storage
| Data | Where |
|---|---|
| Scripture vectors | `sanskrit_faiss_index_itihasa/` (local FAISS, persistent) |
| User-uploaded context | In-memory (`LocalContextStore`, cleared on restart) |
| Logs | `logs/sanskrit_rag.log` |
| Uploaded files | Processed in memory, not saved to disk |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Frontend |
| `GET` | `/health` | Model/index status |
| `POST` | `/api/query` | RAG query |
| `POST` | `/api/context/upload` | Upload PDF / DOCX / TXT |
| `POST` | `/api/context/text` | Add text context directly |
| `GET` | `/api/context/list` | List uploaded contexts |
| `DELETE` | `/api/context/{filename}` | Remove a context |
| `POST` | `/api/stt` | Transcribe audio (Whisper) |

---

## Query Rules

- Queries must be in **English or Sanskrit (Devanagari) only**
- Other scripts (Arabic, Cyrillic, CJK, etc.) are rejected with HTTP 400
- Minimum query length: 3 characters
- Maximum query length: 1000 characters

---

## Configuration (`src/config.py`)

| Setting | Default | Description |
|---|---|---|
| `EMBEDDINGS_MODEL` | `intfloat/multilingual-e5-large` | 1024-dim embedding model |
| `RETRIEVER_K` | `5` | Shlokas retrieved per query |
| `ENABLE_RERANKING` | `True` | TextRank-weighted reranking |
| `MAX_TOKENS` | `256` | LLM max output tokens |
| `TEMPERATURE` | `0.02` | LLM temperature |
| `REPETITION_PENALTY` | `1.2` | Repetition penalty |
| `MAX_FILE_SIZE_MB` | `10` | Upload size limit |
| `SERVER_PORT` | `8000` | Web server port |

---

## File Structure

```
sanskrity/
├── src/
│   ├── server.py              # FastAPI server (fully local)
│   ├── rag_chain.py           # FAISS + TextRank reranker
│   ├── chat.py                # CLI interface
│   ├── models.py              # LLM + embedding model loading
│   ├── data_loader.py         # CSV → LangChain Documents
│   ├── utils.py               # Filters, transliteration, validation
│   ├── config.py              # All settings
│   └── logger.py
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── res/
│   └── combined.csv           # Scripture data (Sanskrit + English)
├── sanskrit_faiss_index_itihasa/  # Built on first run
├── logs/
├── .env                       # MongoDB URL (gitignored)
├── CHANGELOG.md
└── requirements.txt
```

---

## Troubleshooting

**LLM loads slowly on first run** — model weights are downloaded and quantized. Subsequent starts are faster.

**FAISS index missing** — delete the `sanskrit_faiss_index_itihasa/` folder and restart; it will rebuild from `res/combined.csv`.

**STT not working** — Whisper STT runs on the backend. Send audio to `POST /api/stt`. Requires microphone permissions in the browser. Works on HTTPS or localhost.

**Query rejected (400)** — Check that the query is in English or Devanagari. Other scripts are blocked by the input filter.

**File upload fails** — Max 10 MB. Supported formats: PDF, DOCX, TXT only.

**Out of GPU memory** — Reduce `MAX_TOKENS` or `RETRIEVER_K` in `config.py`, or switch to CPU (`device_map="cpu"`).
