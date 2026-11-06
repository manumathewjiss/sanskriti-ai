# Changelog

## 2026-05-31

### Security
- **Removed hardcoded MongoDB credentials from `src/config.py`** (`842afac`)
  - `MONGODB_URL` default value containing plaintext password stripped from source
  - Credential now loaded exclusively from `.env` (already gitignored)

### Features
- **TextRank-weighted shloka reranking in `src/rag_chain.py`** (`c2d6bb0`)
  - Replaced simple cosine-similarity reranker with a PageRank-over-similarity-graph approach
  - Adapted from the TextRank reward function in the legal summarization paper (Prasad et al., AAAI):
    `score_i = query_cosine_sim_i × pagerank_i`
  - Builds an N×N pairwise cosine similarity matrix over retrieved shlokas, runs power-iteration PageRank (d=0.85, 50 steps), multiplies by per-shloka query relevance
  - Shlokas that are both query-relevant and semantically central to the retrieved set rank highest

- **Embedding cache in `src/rag_chain.py`** (`c2d6bb0`)
  - Module-level `_emb_cache` dict keyed by shloka text content
  - Each shloka is embedded exactly once across all queries; subsequent lookups are O(1)
  - Query embedding is still computed fresh each call (unavoidable)
  - Helper `_embed_with_cache(emb, texts)` handles cache-miss batching

### Improvements
- **Hard English/Sanskrit-only input gate in `src/utils.py`** (`c2d6bb0`)
  - New `assert_english_or_sanskrit_only(text, threshold=0.15)` function
  - Raises `ValueError` if >15% of non-whitespace characters fall outside Devanagari + Latin ranges
  - Reuses `keep_sanskrit_english_chars` as the canonical allowed-set definition — input gate and output sanitizer share the same character policy
  - `validate_query()` now calls this, covering both CLI and web paths

- **Server-side script filter in `src/server.py`** (`c2d6bb0`)
  - `/api/query` endpoint calls `assert_english_or_sanskrit_only` before any retrieval
  - Returns HTTP 400 with the specific rejection ratio if query contains unsupported script
