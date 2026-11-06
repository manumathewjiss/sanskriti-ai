"""
Chat interface module for Sanskrit Scripture RAG Assistant.
Handles user interaction and conversation flow.
"""

import sys
from typing import Optional, Callable
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

from config import Config
from logger import get_logger
from rag_chain import (
    format_retrieved_documents,
    format_document_for_display,
    rerank_documents,
)
from config import Config
from utils import validate_query, truncate_text, keep_sanskrit_english_chars

logger = get_logger(__name__)


# ============================================================================
# DISPLAY UTILITIES
# ============================================================================

def print_header() -> None:
    """Print application header."""
    print("\n" + "="*70)
    print("🏛️  SANSKRIT SCRIPTURE RESEARCH ASSISTANT")
    print("="*70)
    print("\nPowered by fine-tuned Qwen LLM + FAISS semantic search")
    print("\nFeatures:")
    print("  • Ask questions about Sanskrit scriptures")
    print("  • Get citations from original sources")
    print("  • View both Sanskrit and English translations")
    print("  • Context-aware responses using RAG")
    print("\nExample questions:")
    print("  'What is the nature of the self according to the Gita?'")
    print("  'Explain the concept of dharma'")
    print("  'What does the Ramayana say about duty?'")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("="*70 + "\n")


def print_retrieved_sources(retrieved_docs) -> None:
    """
    Print retrieved source documents in formatted manner.
    
    Args:
        retrieved_docs: List of retrieved Document objects
    """
    print("\n" + "="*70)
    print("RETRIEVED SOURCES:")
    print("="*70)
    
    if not retrieved_docs:
        print("No documents retrieved.")
        return
    
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n[{i}] {format_document_for_display(doc)}")
    
    print("\n" + "-"*70)


def print_answer(result_text: str) -> None:
    """
    Print LLM answer in formatted manner.
    
    Args:
        result_text: LLM response text
    """
    print("\n" + "-"*70)
    print("ANSWER:")
    print("-"*70)
    print(result_text)
    print("-"*70 + "\n")


def print_error(message: str) -> None:
    """
    Print error message.
    
    Args:
        message: Error message
    """
    print(f"\n❌ Error: {message}")


def print_warning(message: str) -> None:
    """
    Print warning message.
    
    Args:
        message: Warning message
    """
    print(f"\n⚠️ Warning: {message}")


def print_info(message: str) -> None:
    """
    Print informational message.
    
    Args:
        message: Info message
    """
    print(f"\nℹ️ {message}")


def sanitize_response_v2(text: str, max_chars: int = None, preserve_sanskrit_blocks: bool = True) -> str:
    """
    Robust response sanitizer:
    - normalizes whitespace
    - collapses repeated phrases and tokens
    - optionally preserves Sanskrit citation blocks starting with 'Sanskrit:'
    - truncates at a sentence boundary within `max_chars` if provided

    Keep this function pure (no model calls) so it is safe to run locally.
    """
    import re

    if not text:
        return text

    if max_chars is None:
        from config import Config
        max_chars = getattr(Config, "CHAT_MAX_RESPONSE_CHARS", 3000)

    # Split out Sanskrit blocks to avoid altering them
    sanskrit_blocks = {}
    if preserve_sanskrit_blocks:
        def _extract(match):
            key = f"__SAN_{len(sanskrit_blocks)}__"
            sanskrit_blocks[key] = match.group(0)
            return key

        # capture blocks like 'Sanskrit: ...' up to a blank line or 'English:'
        text = re.sub(r"Sanskrit:\s*[^\n]+(?:\n(?:[^\n](?!English:))*?)?(?=\n\s*English:|\n\n|$)", _extract, text, flags=re.IGNORECASE)

    # normalize whitespace and remove control chars
    text = " ".join(text.split())
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]+", " ", text)

    # collapse repeated parenthetical tokens: (Rama) (Rama) ... -> (Rama)
    text = re.sub(r"(\([^)]{1,120}\))(?:\s*\1){3,}", r"\1", text)

    # collapse short phrase repetitions (1-5 words) repeated >=3 times
    text = re.sub(r"\b(\w+(?:\s+\w+){0,4})(?:\s+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)

    # collapse single-word repetition >4
    text = re.sub(r"\b(\w+)(?:\s+\1){4,}\b", r"\1", text, flags=re.IGNORECASE)

    # collapse extreme repeated characters
    text = re.sub(r"(.)\1{20,}", r"\1", text)

    # remove long runs of CJK characters (often model drift into other languages)
    text = re.sub(r"[\u4e00-\u9fff]{3,}", "", text)

    # remove long non-latin blocks if English-only is required by prompt
    text = re.sub(r"[^\x00-\x7f\s]{40,}", "", text)

    # Truncate to last sentence within max_chars
    if len(text) > max_chars:
        cut = text[:max_chars]
        # find last sentence end (., ?, !) before cut
        last_end = max(cut.rfind('.'), cut.rfind('?'), cut.rfind('!'))
        if last_end > 0:
            cut = cut[: last_end + 1]
        text = cut.rstrip() + "\n\n[Response truncated due to length]"

    # restore sanskrit blocks
    if preserve_sanskrit_blocks and sanskrit_blocks:
        for k, v in sanskrit_blocks.items():
            text = text.replace(k, v)

    # Remove any remaining non-English / non-Sanskrit characters
    text = keep_sanskrit_english_chars(text)

    return text


def refine_response_with_llm(llm: HuggingFacePipeline, text: str) -> str:
    """
    Ask the LLM to rewrite the assistant response into concise, fluent English.
    Preserves Sanskrit citation blocks (lines beginning with 'Sanskrit:'), removes
    repetition, and collapses lists when possible.
    """
    if not text:
        return text

    # Build a short, explicit rewrite prompt
    prompt = (
        "You are a helpful editor.\n"
        "Rewrite the following assistant response into clear, concise English.\n"
        "- Remove repetitive phrases and duplicate lists.\n"
        "- Keep any Sanskrit citation blocks (lines beginning with 'Sanskrit:') exactly as they are.\n"
        "- Produce a single coherent paragraph or short bullets (no long repeated sequences).\n\n"
        "Original response:\n" + text + "\n\nCleaned response:"
    )

    try:
        rewritten = llm.invoke(prompt)
        if rewritten:
            rewritten = rewritten.strip()
        return rewritten
    except Exception:
        logger.exception("Refinement LLM call failed")
        return text


# ============================================================================
# QUERY HANDLING
# ============================================================================

def get_user_query() -> Optional[str]:
    """
    Get and validate user query from input.
    
    Returns:
        str: User query or None if invalid
    """
    try:
        query = input("\nYour Question: ").strip()
        
        if not query:
            print_warning("Please enter a question.")
            return None
        
        # Check for exit commands
        if query.lower() in ["quit", "exit", "q", "bye"]:
            return "EXIT"
        
        # Validate query
        try:
            validate_query(query)
        except ValueError as e:
            print_error(str(e))
            return None
        
        return query
    
    except KeyboardInterrupt:
        print("\n")
        return "EXIT"
    
    except Exception as e:
        logger.error(f"Error getting user query: {e}")
        print_error(f"Failed to process input: {e}")
        return None


def process_query(
    query: str,
    llm: HuggingFacePipeline,
    retriever,
    qa_prompt: PromptTemplate
) -> Optional[str]:
    """
    Process a single user query through the RAG pipeline.
    
    Args:
        query: User query
        llm: Language model instance
        retriever: Document retriever
        qa_prompt: QA prompt template
    
    Returns:
        str: LLM response or None on error
    """
    try:
        logger.info(f"Processing query: {query[:60]}...")
        
        # Retrieve relevant documents
        logger.debug("Retrieving relevant documents...")
        retrieved_docs = retriever.invoke(query)
        
        if not retrieved_docs:
            print_warning("No relevant documents found. Please try a different question.")
            logger.warning(f"No documents retrieved for query: {query}")
            return None
        
        # Re-rank retrieved documents for improved relevance
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        logger.debug(f"Original order: {[doc.metadata.get('book', 'unknown') + ' ' + str(doc.metadata.get('chapter', '?')) + '.' + str(doc.metadata.get('shloka_num', '?')) for doc in retrieved_docs]}")
        
        if getattr(Config, "ENABLE_RERANKING", True):
            print(f"\n🔄 Re-ranking {len(retrieved_docs)} documents for better relevance...")
            
            try:
                logger.info(f"Re-ranking {len(retrieved_docs)} documents...")
                reranked_docs = rerank_documents(query, retrieved_docs)
                if reranked_docs and len(reranked_docs) == len(retrieved_docs):
                    # Show before/after for debugging
                    before_first = f"{retrieved_docs[0].metadata.get('book')} {retrieved_docs[0].metadata.get('chapter')}.{retrieved_docs[0].metadata.get('shloka_num')}"
                    after_first = f"{reranked_docs[0].metadata.get('book')} {reranked_docs[0].metadata.get('chapter')}.{reranked_docs[0].metadata.get('shloka_num')}"
                    
                    retrieved_docs = reranked_docs
                    logger.info("✓ Documents re-ranked successfully")
                    logger.debug(f"Re-ranked order: {[doc.metadata.get('book', 'unknown') + ' ' + str(doc.metadata.get('chapter', '?')) + '.' + str(doc.metadata.get('shloka_num', '?')) for doc in retrieved_docs]}")
                    
                    print(f"✓ Re-ranking complete: {before_first} → {after_first}")
                else:
                    logger.warning(f"Re-ranking returned unexpected results (got {len(reranked_docs) if reranked_docs else 0} docs), using original order")
                    print("⚠️ Re-ranking failed, using original retrieval order")
            except Exception as e:
                logger.error(f"Re-ranking failed with error: {e}, using original order", exc_info=True)
                print(f"⚠️ Re-ranking error: {str(e)[:100]}")
        else:
            print("ℹ️  Re-ranking disabled (using FAISS retrieval order)")

        # Display retrieved sources
        print_retrieved_sources(retrieved_docs)
        
        # Format documents for prompt
        logger.debug("Formatting documents...")
        custom_context, first_citation = format_retrieved_documents(retrieved_docs)
        
        # Create final prompt
        logger.debug("Creating final prompt...")
        final_prompt = qa_prompt.format(
            context=custom_context,
            question=query,
            context_citation_1=first_citation["citation"],
            sanskrit_1=first_citation["sanskrit"],
            english_1=first_citation["english"]
        )
        
        # Invoke LLM
        logger.info("Invoking LLM for response generation...")
        result_text = llm.invoke(final_prompt)
        
        # Clean up result
        result_text = result_text.strip() if result_text else "No response generated"
        logger.debug(f"Raw response length: {len(result_text)} chars")

        # Truncate at stopping strings (since pipeline stopping doesn't work with LangChain)
        stopping_strings = getattr(Config, "STOPPING_STRINGS", ["### Human:", "QUESTION:", "\n\n\n", "不断"])
        for stop_str in stopping_strings:
            if stop_str in result_text:
                result_text = result_text[:result_text.index(stop_str)]
                logger.info(f"✓ Truncated response at stopping string: '{stop_str[:20]}...'")
                break
        
        # Use the robust sanitizer (preserves Sanskrit citation blocks)
        try:
            result_text = sanitize_response_v2(result_text)
        except Exception:
            # fallback to minimal trimming
            result_text = (result_text or "").strip()[: getattr(Config, "CHAT_MAX_RESPONSE_CHARS", 3000)]

        # Prepend structured header with citation, Sanskrit, and English
        structured_header = f"""The most relevant verse is **{first_citation["citation"]}**.

**Sanskrit Text:** {first_citation["sanskrit"]}

**English Translation:** {first_citation["english"]}

**Explanation:** """
        
        result_text = structured_header + result_text
        
        # Skip LLM refinement - it causes more issues than it solves
        logger.info(f"✓ Response generated successfully ({len(result_text)} chars)")
        return result_text
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        print_error(f"Failed to process query: {e}")
        return None


# ============================================================================
# MAIN CHAT LOOP
# ============================================================================

def chat_with_assistant(
    llm: HuggingFacePipeline,
    retriever,
    qa_prompt: PromptTemplate,
    max_queries: Optional[int] = None
) -> int:
    """
    Run the main interactive chat loop.
    
    Args:
        llm: Language model instance
        retriever: Document retriever instance
        qa_prompt: Question-answering prompt template
        max_queries: Maximum number of queries to process (None = unlimited)
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    print_header()
    
    logger.info("Chat session started")
    
    query_count = 0
    
    while True:
        try:
            # Check query limit
            if max_queries and query_count >= max_queries:
                print_info(f"Reached maximum queries limit ({max_queries})")
                break
            
            # Get user query
            user_query = get_user_query()
            
            if user_query is None:
                # Invalid input, try again
                continue
            
            if user_query == "EXIT":
                # User wants to exit
                break
            
            query_count += 1
            
            # Process query
            result = process_query(user_query, llm, retriever, qa_prompt)
            
            if result:
                # Display answer
                print_answer(result)
                logger.info(f"✓ Query {query_count} completed successfully")
            else:
                # Error in processing
                logger.warning(f"Query {query_count} failed to generate response")
        
        except KeyboardInterrupt:
            print("\n")
            logger.info("Chat session interrupted by user")
            break
        
        except Exception as e:
            logger.error(f"Unexpected error in chat loop: {e}", exc_info=True)
            print_error(f"Unexpected error: {e}")
            print_warning("Please try again with a different question.")
    
    # Exit message
    print("\n" + "="*70)
    print("✅ Thank you for using the Sanskrit Scripture Research Assistant")
    print(f"   Processed {query_count} questions in this session")
    print("="*70 + "\n")
    
    logger.info(f"Chat session ended. Total queries: {query_count}")
    return 0


# ============================================================================
# ALTERNATIVE CHAT MODES
# ============================================================================

def chat_batch(
    queries: list,
    llm: HuggingFacePipeline,
    retriever,
    qa_prompt: PromptTemplate
) -> list:
    """
    Process multiple queries in batch mode.
    
    Args:
        queries: List of query strings
        llm: Language model instance
        retriever: Document retriever instance
        qa_prompt: Question-answering prompt template
    
    Returns:
        list: List of (query, result) tuples
    """
    logger.info(f"Starting batch chat with {len(queries)} queries")
    
    results = []
    
    for i, query in enumerate(queries, 1):
        logger.info(f"Processing batch query {i}/{len(queries)}")
        
        result = process_query(query, llm, retriever, qa_prompt)
        results.append((query, result))
    
    logger.info(f"Batch processing completed. {len(results)} results")
    return results


def chat_with_context(
    initial_context: str,
    llm: HuggingFacePipeline,
    retriever,
    qa_prompt: PromptTemplate
) -> None:
    """
    Run chat with initial context (e.g., focused on a specific book).
    
    Args:
        initial_context: Initial context information
        llm: Language model instance
        retriever: Document retriever instance
        qa_prompt: Question-answering prompt template
    """
    print_header()
    print_info(f"Context: {initial_context}")
    print("-" * 70)
    
    logger.info(f"Chat session started with context: {initial_context}")
    
    while True:
        try:
            user_query = get_user_query()
            
            if user_query is None:
                continue
            
            if user_query == "EXIT":
                break
            
            # Add context to query
            contextual_query = f"{initial_context}\n{user_query}"
            
            result = process_query(contextual_query, llm, retriever, qa_prompt)
            
            if result:
                print_answer(result)
        
        except KeyboardInterrupt:
            print("\n")
            break
        
        except Exception as e:
            logger.error(f"Error in contextual chat: {e}")
            print_error(f"Error: {e}")
    
    print_info("Chat session ended")
    logger.info("Context chat session ended")
