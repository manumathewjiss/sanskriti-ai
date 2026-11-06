"""
Configuration settings for the Sanskrity project.
All configurable parameters are defined here.
"""
import os
import torch
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

from transformers import BitsAndBytesConfig

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).parent.parent


class Config:
    """Centralized configuration class for all settings."""
    
    # =========================================================================
    # ENVIRONMENT & PATHS
    # =========================================================================
    
    # Project root directory (can be overridden via environment variable)
    PROJECT_DIR = os.environ.get('SANSKRIT_PROJECT_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Partial local demo: SANSKRITY_DEMO=1 (no CUDA / no Qwen weights)
    DEMO_MODE = os.environ.get("SANSKRITY_DEMO", "").strip().lower() in {"1", "true", "yes"}

    # Data paths
    DATA_PATH = os.path.join(PROJECT_DIR, "res/combined.csv")
    ADAPTER_PATH = os.path.join(PROJECT_DIR, "qwen-finetuned")
    BASE_MODEL = "qwen"
    FAISS_INDEX_PATH = os.path.join(PROJECT_DIR, "sanskrit_faiss_index_itihasa")
    LOG_FILE = os.path.join(PROJECT_DIR, "logs/sanskrit_rag.log")

    if DEMO_MODE:
        DATA_PATH = os.path.join(PROJECT_DIR, "res/combined_demo.csv")
        FAISS_INDEX_PATH = os.path.join(PROJECT_DIR, "sanskrit_faiss_index_demo")
        BASE_MODEL = "distilgpt2"
    
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # =========================================================================
    # EMBEDDING MODEL SETTINGS
    # =========================================================================
    # Embedding model for semantic search
    # Using multilingual model for Sanskrit/English texts
    EMBEDDINGS_MODEL = 'intfloat/multilingual-e5-large'  # 1024 dimensions
    # Alternative: 'sentence-transformers/all-MiniLM-L6-v2' (384 dims - smaller)
    EMBEDDINGS_BATCH_SIZE = 32
    if DEMO_MODE:
        EMBEDDINGS_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # =========================================================================
    # LLM PARAMETERS
    # =========================================================================
    
    MAX_TOKENS = 256  # Reduced from 512 to prevent over-generation
    TEMPERATURE = 0.02  # Low temperature for factual RAG responses
    TOP_P = 0.9
    TOP_K = 50
    DO_SAMPLE = True
    # Generation tuning to reduce repetition (tunable)
    REPETITION_PENALTY = 1.2
    NO_REPEAT_NGRAM_SIZE = 3
    DO_SAMPLE = False
    
    # Stopping criteria to prevent gibberish generation
    STOPPING_STRINGS = ["### Human:", "QUESTION:", "\n\n\n", "不断"]
    
    # =========================================================================
    # RETRIEVAL SETTINGS
    # =========================================================================
    
    RETRIEVER_K = 5  # Number of documents to retrieve per query
    RETRIEVER_SCORE_THRESHOLD = 0.0  # Minimum similarity score (0.0 = no threshold)
    ENABLE_RERANKING = True  # Language-aware re-ranking (Sanskrit→Sanskrit, English→English)
    
    # =========================================================================
    # TRAINING SETTINGS (for train.py)
    # =========================================================================
    
    # LoRA Configuration
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]
    
    # Training parameters
    TRAINING_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16
    LEARNING_RATE = 2e-4
    NUM_TRAIN_EPOCHS = 3
    WARMUP_RATIO = 0.05
    LOGGING_STEPS = 20
    
    # =========================================================================
    # RESOURCE SETTINGS
    # =========================================================================
    
    # Quantization configuration
    USE_4BIT_QUANTIZATION = True
    BNB_4BIT_QUANT_TYPE = "nf4"  # NormalFloat 4-bit
    BNB_4BIT_USE_DOUBLE_QUANT = True
    BNB_4BIT_COMPUTE_DTYPE = torch.float16
    
    # Model loading
    LOAD_IN_4BIT = True
    DEVICE_MAP = "auto"
    USE_GRADIENT_CHECKPOINTING = True
    if DEMO_MODE:
        USE_4BIT_QUANTIZATION = False
        LOAD_IN_4BIT = False
    
    # =========================================================================
    # CHAT INTERFACE SETTINGS
    # =========================================================================
    
    CHAT_TIMEOUT_SECONDS = 300
    MAX_CONTEXT_LENGTH = 2000
    CHAT_MAX_RESPONSE_CHARS = 3000
    
    # =============================================================================
    # MongoDB Atlas Configuration
    # =============================================================================
    MONGODB_URL = os.getenv('MONGODB_URL')
    MONGODB_DB_NAME = 'sanskrity'
    SCRIPTURE_COLLECTION = 'scripture_vectors'
    USER_CONTEXT_COLLECTION = 'user_context_vectors'
    
    # Vector search configuration
    VECTOR_DIMENSION = 1024  # multilingual-e5-large dimension
    VECTOR_SIMILARITY_METRIC = 'cosine'
    
    # =========================================================================
    # WEB SERVER SETTINGS
    # =========================================================================
    
    SERVER_HOST = '0.0.0.0'
    SERVER_PORT = 8000
    
    # File upload settings
    MAX_FILE_SIZE_MB = 10
    ALLOWED_EXTENSIONS = ['pdf', 'docx', 'txt']
    
    # Context processing
    CONTEXT_CHUNK_SIZE = 500  # characters per chunk
    CONTEXT_OVERLAP = 50  # overlap between chunks
    CONTEXT_RETRIEVAL_K = 5  # number of context chunks to retrieve
    SCRIPTURE_RETRIEVAL_K = 3  # number of scripture chunks to retrieve
    
    # Local storage paths (for backups and offline mode)
    UPLOAD_DIR = BASE_DIR / 'data' / 'uploads'  # Store uploaded files
    BACKUP_DIR = BASE_DIR / 'data' / 'backups'  # Vector backups
    VECTOR_CACHE_DIR = BASE_DIR / 'data' / 'vectors'  # Local vector cache
    
    # Storage settings
    KEEP_UPLOADED_FILES = True  # Save uploaded files locally
    AUTO_BACKUP_ENABLED = True  # Periodic backups
    BACKUP_INTERVAL_HOURS = 24  # Backup frequency
    
    # =========================================================================
    # VALIDATION & DEBUGGING
    # =========================================================================
    
    ENABLE_DEBUG_LOGGING = False
    VALIDATE_PATHS_ON_STARTUP = True
    
    @staticmethod
    def get_bnb_config() -> BitsAndBytesConfig:
        """
        Get BitsAndBytes quantization configuration for 4-bit model loading.
        
        Returns:
            BitsAndBytesConfig: Configuration for 4-bit quantization
        """
        return BitsAndBytesConfig(
            load_in_4bit=Config.LOAD_IN_4BIT,
            bnb_4bit_quant_type=Config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_use_double_quant=Config.BNB_4BIT_USE_DOUBLE_QUANT,
            bnb_4bit_compute_dtype=Config.BNB_4BIT_COMPUTE_DTYPE,
        )
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that all required paths and configurations exist.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        if not cls.VALIDATE_PATHS_ON_STARTUP:
            return True
        
        required_paths = [cls.PROJECT_DIR]
        missing_paths = []
        
        for path in required_paths:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            print(f"❌ Configuration validation failed. Missing paths:")
            for path in missing_paths:
                print(f"   - {path}")
            return False
        
        return True
    
    @staticmethod
    def is_cuda_available() -> bool:
        """
        Check if CUDA is available.
        
        Returns:
            bool: True if CUDA is available
        """
        return torch.cuda.is_available()
    
    @staticmethod
    def get_cuda_device_info() -> Optional[str]:
        """
        Get information about the current CUDA device.
        
        Returns:
            str: Device name or None if CUDA not available
        """
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return None
    
    @staticmethod
    def get_memory_info() -> dict:
        """
        Get GPU and CPU memory information.
        
        Returns:
            dict: Memory information
        """
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            info["current_device"] = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return info
    
    @classmethod
    def print_config(cls) -> None:
        """Print current configuration settings."""
        print("\n" + "="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)
        print(f"Project Directory: {cls.PROJECT_DIR}")
        print(f"Data Path: {cls.DATA_PATH}")
        print(f"Model: {cls.BASE_MODEL}")
        print(f"Adapter Path: {cls.ADAPTER_PATH}")
        print(f"FAISS Index: {cls.FAISS_INDEX_PATH}")
        print(f"Embeddings Model: {cls.EMBEDDINGS_MODEL}")
        print(f"Max Tokens: {cls.MAX_TOKENS}")
        print(f"Temperature: {cls.TEMPERATURE}")
        print(f"Retriever K: {cls.RETRIEVER_K}")
        print(f"4-bit Quantization: {cls.USE_4BIT_QUANTIZATION}")
        print("="*70 + "\n")
