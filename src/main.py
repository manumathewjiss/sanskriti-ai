"""
Main entry point for Sanskrit Scripture RAG Assistant.
Orchestrates initialization and launches the chat interface.
"""

import sys
import os
from typing import Optional

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from logger import setup_logger, get_logger
from data_loader import load_and_chunk_itihasa_data
from models import setup_local_llm
from rag_chain import initialize_vector_store, create_qa_prompt, get_retriever
from chat import chat_with_assistant
from utils import check_cuda_availability, ensure_directory

# Initialize logger
logger = setup_logger(log_file=Config.LOG_FILE)


def main() -> int:
    """
    Main application entry point.
    
    Orchestrates:
    1. Configuration validation
    2. Environment setup (CUDA, directories)
    3. Data loading
    4. Vector store initialization
    5. Model setup
    6. Chat interface startup
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    logger.info("="*70)
    logger.info("Sanskrit Scripture RAG Assistant - Startup")
    logger.info("="*70)
    
    try:
        # =====================================================================
        # 1. CONFIGURATION & VALIDATION
        # =====================================================================
        
        logger.info("Step 1: Validating configuration...")
        
        if not Config.validate():
            logger.error("Configuration validation failed")
            return 1
        
        Config.print_config()
        
        # =====================================================================
        # 2. ENVIRONMENT SETUP
        # =====================================================================
        
        logger.info("Step 2: Setting up environment...")
        
        # Check CUDA
        cuda_available = check_cuda_availability()
        
        # Create necessary directories
        ensure_directory(os.path.dirname(Config.FAISS_INDEX_PATH))
        ensure_directory(os.path.dirname(Config.LOG_FILE))
        
        logger.info("✓ Environment setup complete")
        
        # =====================================================================
        # 3. DATA LOADING
        # =====================================================================
        
        logger.info("Step 3: Loading Sanskrit scripture data...")
        
        chunks = load_and_chunk_itihasa_data(Config.DATA_PATH)
        
        if not chunks:
            logger.error("Failed to load data. Aborting.")
            return 1
        
        logger.info(f"✓ Loaded {len(chunks)} documents")
        
        # =====================================================================
        # 4. VECTOR STORE INITIALIZATION
        # =====================================================================
        
        logger.info("Step 4: Initializing vector store...")
        
        vector_store = initialize_vector_store(chunks)
        
        if vector_store is None:
            logger.error("Failed to initialize vector store")
            return 1
        
        # Create retriever
        retriever = get_retriever(vector_store, k=Config.RETRIEVER_K)
        logger.info("✓ Vector store and retriever initialized")
        
        # =====================================================================
        # 5. MODEL SETUP
        # =====================================================================
        
        logger.info("Step 5: Setting up language model...")
        
        llm = setup_local_llm(
            base_model_name=Config.BASE_MODEL,
            adapter_path=Config.ADAPTER_PATH
        )
        
        logger.info("✓ Language model initialized")
        
        # =====================================================================
        # 6. PROMPT SETUP
        # =====================================================================
        
        logger.info("Step 6: Creating prompt templates...")
        
        qa_prompt = create_qa_prompt()
        logger.info("✓ Prompt templates created")
        
        # =====================================================================
        # 7. START CHAT INTERFACE
        # =====================================================================
        
        logger.info("="*70)
        logger.info("✓ All systems initialized successfully")
        logger.info("="*70)
        logger.info("Starting chat interface...")
        
        exit_code = chat_with_assistant(llm, retriever, qa_prompt)
        
        logger.info("✓ Application completed successfully")
        return exit_code
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n✅ Shutting down...")
        return 0
    
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        print("Check the log file for more details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
