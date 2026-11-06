"""
RAG (Retrieval-Augmented Generation) chain module.
Handles FAISS vector store initialization and document retrieval.
"""

import os
from typing import Dict, List, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from config import Config
from data_loader import create_citation_string
from logger import get_logger
from models import load_embedding_model
from utils import ensure_directory

logger = get_logger(__name__)

# Embedding cache: text -> vector. Shloka content is static so vectors never expire.
_emb_cache: Dict[str, List[float]] = {}


def _embed_with_cache(emb, texts: List[str]) -> List[List[float]]:
    """Embed texts, returning cached vectors for any previously seen content."""
    results: List[Optional[List[float]]] = [None] * len(texts)
    miss_texts: List[str] = []
    miss_indices: List[int] = []

    for i, t in enumerate(texts):
        if t in _emb_cache:
            results[i] = _emb_cache[t]
        else:
            miss_texts.append(t)
            miss_indices.append(i)

    if miss_texts:
        new_vecs = emb.embed_documents(miss_texts)
        for idx, vec in zip(miss_indices, new_vecs):
            _emb_cache[texts[idx]] = vec
            results[idx] = vec

    return results  # type: ignore[return-value]


# ============================================================================
# VECTOR STORE INITIALIZATION
# ============================================================================

def initialize_vector_store(
    chunk_docs: Optional[List[Document]] = None,
    embedding_model_name: str = Config.EMBEDDINGS_MODEL,
    force_rebuild: bool = False
) -> Optional[FAISS]:
    """
    Load or create FAISS vector store with HuggingFace embeddings.
    
    Implements smart persistence:
    - Load existing index from disk if available
    - Create new index from documents if not found
    - Gracefully handle missing data with fallback
    
    Args:
        chunk_docs: Optional list of documents to create new index from
        embedding_model_name: HuggingFace embedding model name
        force_rebuild: Force rebuild even if index exists
    
    Returns:
        FAISS vector store instance or None on critical failure
    
    Example:
        >>> from data_loader import load_and_chunk_itihasa_data
        >>> docs = load_and_chunk_itihasa_data("data.csv")
        >>> vector_store = initialize_vector_store(docs)
        >>> retriever = vector_store.as_retriever(k=5)
    """
    logger.info(f"Initializing vector store...")
    logger.info(f"Index path: {Config.FAISS_INDEX_PATH}")
    
    try:
        # Load embedding model
        embedding_model = load_embedding_model(embedding_model_name)
        
        # Check if index exists and should be used
        index_exists = os.path.exists(Config.FAISS_INDEX_PATH)
        
        if index_exists and not force_rebuild:
            logger.info("Loading existing FAISS index from disk...")
            
            try:
                vector_store = FAISS.load_local(
                    folder_path=Config.FAISS_INDEX_PATH,
                    embeddings=embedding_model,
                    allow_dangerous_deserialization=True
                )
                
                logger.info("✓ FAISS index loaded successfully")
                return vector_store
            
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                logger.info("Will create new index...")
        
        # Create new index
        if chunk_docs is None or len(chunk_docs) == 0:
            logger.error("No documents available for FAISS index creation")
            return None
        
        logger.info(f"Creating new FAISS index from {len(chunk_docs)} documents...")
        vector_store = FAISS.from_documents(chunk_docs, embedding_model)
        
        # Save index
        logger.info(f"Saving FAISS index to {Config.FAISS_INDEX_PATH}...")
        ensure_directory(Config.FAISS_INDEX_PATH)
        vector_store.save_local(Config.FAISS_INDEX_PATH)
        
        logger.info("✓ New FAISS index created and saved")
        return vector_store
    
    except Exception as e:
        logger.error(f"Critical error initializing vector store: {e}")
        return None


def get_retriever(
    vector_store: FAISS,
    k: int = Config.RETRIEVER_K,
    score_threshold: float = Config.RETRIEVER_SCORE_THRESHOLD
):
    """
    Create a retriever from a FAISS vector store.
    
    Args:
        vector_store: FAISS vector store instance
        k: Number of documents to retrieve
        score_threshold: Minimum similarity score (0.0 = no threshold)
    
    Returns:
        Retriever instance
    """
    logger.info(f"Creating retriever with k={k}, score_threshold={score_threshold}")
    
    try:
        # Use similarity search retriever without score threshold for better compatibility
        if score_threshold > 0.0:
            retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold}
            )
        else:
            # Simple similarity search without threshold
            retriever = vector_store.as_retriever(
                search_kwargs={"k": k}
            )
        logger.info("✓ Retriever created successfully")
        return retriever
    
    except Exception as e:
        logger.error(f"Failed to create retriever: {e}")
        raise


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

def create_qa_prompt() -> PromptTemplate:
    """
    Create the main question-answering prompt template.
    
    Returns:
        PromptTemplate: Configured prompt template
    """
    template = """### Human:
You are a Sanskrit scripture scholar helping to interpret verses from the Ramayana and Mahabharata.

CONTEXT:
{context}

QUESTION: {question}

The most relevant verse from the scriptures is: {context_citation_1}

Your task is to explain this verse based on its English translation. Focus on:
- The literal meaning and what actions/emotions are described
- Any philosophical or moral significance that is evident
- Key themes or concepts mentioned

IMPORTANT: You only have access to the verse translation itself, not the full story context. If something is unclear from the verse alone (such as who exactly is speaking), acknowledge this rather than guessing. Explain only what is clearly evident from the translation.

Provide all responses in English only.

### Assistant:
The most relevant verse is **{context_citation_1}**.

**Sanskrit Text:** {sanskrit_1}

**English Translation:** {english_1}

**Explanation:**"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question",
            "context_citation_1",
            "sanskrit_1",
            "english_1"
        ]
    )
    
    logger.debug("✓ QA prompt template created")
    return prompt


def create_summary_prompt() -> PromptTemplate:
    """
    Create a prompt template for document summarization.
    
    Returns:
        PromptTemplate: Summary prompt template
    """
    template = """### Human:
Summarize the following Sanskrit text and its English translation:

Sanskrit: {sanskrit_text}
English: {english_text}

### Assistant:
Summary: {summary}"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["sanskrit_text", "english_text", "summary"]
    )
    
    logger.debug("✓ Summary prompt template created")
    return prompt


def create_translation_prompt() -> PromptTemplate:
    """
    Create a prompt template for translation tasks.
    
    Returns:
        PromptTemplate: Translation prompt template
    """
    template = """### Human:
Translate the following Sanskrit verse into English:

Sanskrit: {sanskrit_text}

### Assistant:
English Translation: {english_translation}"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["sanskrit_text", "english_translation"]
    )
    
    logger.debug("✓ Translation prompt template created")
    return prompt


# ============================================================================
# DOCUMENT FORMATTING
# ============================================================================

def format_retrieved_documents(
    retrieved_docs: List[Document],
    include_metadata: bool = True
) -> Tuple[str, Dict]:
    """
    Format retrieved documents into context string and citation information.
    
    Args:
        retrieved_docs: List of retrieved Document objects
        include_metadata: Whether to include metadata in output
    
    Returns:
        Tuple of (formatted_context, first_doc_citation_dict)
    
    Example:
        >>> context, citation = format_retrieved_documents(docs)
        >>> print(context)  # Formatted context for LLM
        >>> print(citation['citation'])  # "Bhagavad Gita 2.47"
    """
    if not retrieved_docs:
        logger.warning("No documents retrieved for formatting")
        return "", {}
    
    context_parts = []
    
    for i, doc in enumerate(retrieved_docs):
        meta = doc.metadata
        sanskrit_text = doc.page_content
        english_text = meta.get("english_text", "Translation not available")
        citation = create_citation_string(meta)
        
        if include_metadata:
            context_block = f"""
--- SHLOKA {i+1} (Source: {citation}) ---
Sanskrit: {sanskrit_text}
English Translation: {english_text}
"""
        else:
            context_block = f"{sanskrit_text}\n{english_text}\n"
        
        context_parts.append(context_block)
    
    context = "\n".join(context_parts)
    
    # Extract first document citation info
    first_meta = retrieved_docs[0].metadata
    citation_dict = {
        "citation": create_citation_string(first_meta),
        "sanskrit": retrieved_docs[0].page_content,
        "english": first_meta.get("english_text", "Translation not available"),
        "book": first_meta.get('book'),
        "chapter": first_meta.get('chapter'),
        "shloka": first_meta.get('shloka_num')
    }
    
    logger.debug(f"Formatted {len(retrieved_docs)} documents for context")
    return context, citation_dict


def format_document_for_display(doc: Document) -> str:
    """
    Format a single document for CLI display.
    
    Args:
        doc: Document to format
    
    Returns:
        str: Formatted document string
    """
    meta = doc.metadata
    citation = create_citation_string(meta)
    sanskrit = doc.page_content
    english = meta.get("english_text", "Translation not available")
    
    formatted = f"""
Citation: {citation}
Sanskrit: {sanskrit}
English: {english}
"""
    return formatted.strip()


# ============================================================================
# RETRIEVAL UTILITIES
# ============================================================================

def search_vector_store(
    vector_store: FAISS,
    query: str,
    k: int = Config.RETRIEVER_K,
    score_threshold: float = Config.RETRIEVER_SCORE_THRESHOLD
) -> List[Document]:
    """
    Search the vector store for similar documents.
    
    Args:
        vector_store: FAISS vector store instance
        query: Query string
        k: Number of results to return
        score_threshold: Minimum similarity score
    
    Returns:
        List of relevant Document objects
    """
    try:
        logger.debug(f"Searching vector store with query: '{query[:50]}...'")
        
        results = vector_store.similarity_search_with_score(
            query,
            k=k,
            score_threshold=score_threshold
        )
        
        documents = [doc for doc, score in results]
        scores = [score for doc, score in results]
        
        if documents:
            avg_score = sum(scores) / len(scores)
            logger.debug(f"Found {len(documents)} documents, avg score: {avg_score:.3f}")
        else:
            logger.warning(f"No documents found for query: {query}")
        
        return documents
    
    except Exception as e:
        logger.error(f"Error searching vector store: {e}")
        return []


def get_document_statistics(vector_store: FAISS) -> dict:
    """
    Get statistics about the vector store.
    
    Args:
        vector_store: FAISS vector store instance
    
    Returns:
        dict: Statistics about the vector store
    """
    try:
        # Try to get index statistics (this may not always be available)
        stats = {
            "vector_store_type": "FAISS",
            "index_path": Config.FAISS_INDEX_PATH,
            "note": "Detailed statistics depend on FAISS implementation"
        }
        
        logger.debug("✓ Vector store statistics retrieved")
        return stats
    
    except Exception as e:
        logger.error(f"Failed to get vector store statistics: {e}")
        return {"error": str(e)}


def rerank_documents(query: str, retrieved_docs: List[Document]) -> List[Document]:
    """
    Re-rank shlokas using TextRank-weighted relevance scoring.

    Adapted from the TextRank-weighted semantic similarity reward used in
    GRPO fine-tuning for legal document summarization (Prasad et al., AAAI):

        score_i = query_sim_i * pagerank_i

    where pagerank_i is derived from a PageRank (TextRank) pass over the
    pairwise cosine-similarity graph built from all retrieved shlokas.
    Shlokas that are both query-relevant AND semantically central to the
    retrieved set are ranked highest.
    """
    if not retrieved_docs:
        logger.debug("No documents to rerank")
        return []

    try:
        import numpy as np
        from utils import is_sanskrit

        emb = load_embedding_model()
        query_is_sanskrit = is_sanskrit(query)
        logger.debug(f"Query language: {'Sanskrit' if query_is_sanskrit else 'English'}")

        # Choose embedding text based on query language
        texts = []
        for doc in retrieved_docs:
            if query_is_sanskrit:
                texts.append(doc.page_content or "")
            else:
                texts.append(doc.metadata.get("english_text", "") or doc.page_content or "")

        # Query is never cached (changes every call); doc texts are.
        doc_vecs_list = _embed_with_cache(emb, texts)
        query_vecs_list = emb.embed_documents([query])
        all_vecs = np.array([query_vecs_list[0]] + doc_vecs_list, dtype=np.float32)
        q_vec = all_vecs[0]
        doc_vecs = all_vecs[1:]

        # L2-normalize for cosine via dot product
        def _norm(mat):
            norms = np.linalg.norm(mat, axis=-1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return mat / norms

        q_norm = _norm(q_vec.reshape(1, -1))[0]
        doc_norms = _norm(doc_vecs)

        # --- TextRank: PageRank over shloka similarity graph ---
        n = len(retrieved_docs)
        sim_matrix = doc_norms @ doc_norms.T          # (N, N) cosine similarities
        np.fill_diagonal(sim_matrix, 0.0)             # no self-loops
        row_sums = sim_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        transition = sim_matrix / row_sums             # row-stochastic

        # Power iteration (damping d=0.85, 50 steps is sufficient for N<=20)
        pr = np.ones(n, dtype=np.float32) / n
        d = 0.85
        for _ in range(50):
            pr = (1.0 - d) / n + d * (transition.T @ pr)

        # --- Query-shloka cosine similarity ---
        query_sims = doc_norms @ q_norm               # (N,)

        # --- Combined score ---
        combined = query_sims * pr

        order = np.argsort(combined)[::-1]
        reranked = [retrieved_docs[i] for i in order]

        for rank, i in enumerate(order[:3]):
            doc = retrieved_docs[i]
            cit = f"{doc.metadata.get('book')} {doc.metadata.get('chapter')}.{doc.metadata.get('shloka_num')}"
            logger.debug(
                f"  TextRank Rank #{rank+1}: {cit} "
                f"(q_sim={query_sims[i]:.4f}, pr={pr[i]:.4f}, combined={combined[i]:.4f})"
            )

        return reranked

    except Exception as e:
        logger.warning(f"TextRank reranking failed: {e}; returning original order")
        return retrieved_docs
