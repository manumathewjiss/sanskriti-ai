"""
Sanskrit Scripture RAG Assistant

A production-grade Retrieval-Augmented Generation (RAG) system for querying
Sanskrit scriptures using fine-tuned language models and semantic search.

Modules:
    config: Configuration management
    logger: Logging setup
    data_loader: Data loading and processing
    models: Model initialization and management
    utils: Utility functions
    rag_chain: RAG pipeline and vector store management
    chat: Interactive chat interface
    main: Application entry point
"""

__version__ = "1.0.0"
__author__ = "Sanskrit RAG Team"
__description__ = "Retrieval-Augmented Generation for Sanskrit Scriptures"

# Import main components for easy access
from config import Config
from logger import get_logger, setup_logger
from data_loader import load_and_chunk_itihasa_data
from models import setup_local_llm, load_embedding_model
from rag_chain import initialize_vector_store, create_qa_prompt, get_retriever
from chat import chat_with_assistant

__all__ = [
    "Config",
    "get_logger",
    "setup_logger",
    "load_and_chunk_itihasa_data",
    "setup_local_llm",
    "load_embedding_model",
    "initialize_vector_store",
    "create_qa_prompt",
    "get_retriever",
    "chat_with_assistant",
]
