"""
Model loading and initialization module for Sanskrit Scripture RAG Assistant.
Handles LLM setup, tokenizer loading, and quantization configuration.
"""

import torch
from typing import Tuple, Optional, List
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, StoppingCriteria, StoppingCriteriaList
from langchain_community.llms import HuggingFacePipeline

from config import Config
from logger import get_logger
from utils import check_cuda_availability

logger = get_logger(__name__)


# ============================================================================
# STOPPING CRITERIA
# ============================================================================

class KeywordsStoppingCriteria(StoppingCriteria):
    """
    Custom stopping criteria that stops generation when specific keywords are detected.
    Prevents the model from generating gibberish or continuing past the answer.
    """
    def __init__(self, keywords: List[str], tokenizer: AutoTokenizer):
        self.keywords = keywords
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode the current generated text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        
        # Check if any stopping keyword is in the generated text
        for keyword in self.keywords:
            if keyword in generated_text:
                return True
        
        return False


# ============================================================================
# LLM SETUP & INITIALIZATION
# ============================================================================

def setup_local_llm(
    base_model_name: str = Config.BASE_MODEL,
    adapter_path: str = Config.ADAPTER_PATH,
    use_adapter: bool = True
) -> HuggingFacePipeline:
    """
    Initialize and load the fine-tuned LLM with optional LoRA adapter.
    
    Supports graceful fallback:
    - Try to load fine-tuned model with adapter
    - Fall back to base model if adapter fails
    - Fall back to distilgpt2 if base model fails
    
    Args:
        base_model_name: Name or path of the base model
        adapter_path: Path to LoRA adapter weights
        use_adapter: Whether to use adapter (default: True)
    
    Returns:
        HuggingFacePipeline: Configured pipeline for text generation
    
    Raises:
        RuntimeError: If even fallback model cannot be loaded
    """
    logger.info(f"Setting up LLM: {base_model_name}")
    logger.info(f"Adapter path: {adapter_path}")
    logger.info(f"Use adapter: {use_adapter}")
    
    # Check CUDA availability
    check_cuda_availability()

    if getattr(Config, "DEMO_MODE", False):
        use_adapter = False
        logger.info("Demo mode: skipping LoRA adapter and 4-bit quantization")

    use_quant = Config.USE_4BIT_QUANTIZATION and Config.is_cuda_available()
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if use_quant:
        load_kwargs["quantization_config"] = Config.get_bnb_config()
    
    tokenizer = None
    model = None
    
    try:
        # Load tokenizer
        logger.info(f"Loading tokenizer from {base_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug("Set pad token to eos token")
        
        # Load base model
        logger.info(f"Loading base model: {base_model_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **load_kwargs
        )
        
        # Load adapter if requested
        if use_adapter:
            try:
                logger.info(f"Loading LoRA adapter from {adapter_path}...")
                model = PeftModel.from_pretrained(base_model, adapter_path)
                model = model.merge_and_unload()
                model.eval()
                logger.info("✓ Successfully loaded and merged LoRA adapter")
            
            except Exception as adapter_error:
                logger.warning(f"Failed to load adapter: {adapter_error}")
                logger.info("Continuing with base model only (no adapter)")
                model = base_model
                model.eval()
        else:
            model = base_model
            model.eval()
        
        logger.info("✓ Model loaded successfully")
    
    except Exception as e:
        logger.error(f"Failed to load {base_model_name}: {e}")
        logger.info("Attempting fallback to distilgpt2...")
        
        try:
            fallback_model = "distilgpt2"
            tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                device_map="auto",
                trust_remote_code=True
            )
            model.eval()
            
            logger.warning(f"⚠ Loaded fallback model: {fallback_model}")
            logger.warning("Note: This model may provide lower quality responses")
        
        except Exception as fallback_error:
            logger.error(f"Failed to load fallback model: {fallback_error}")
            raise RuntimeError("Could not load any model. Please check your environment.")
    
    # Create text generation pipeline
    logger.info("Creating text generation pipeline...")
    
    try:
        # Note: stopping_criteria is NOT supported by LangChain's HuggingFacePipeline
        # Instead, we'll handle stopping in post-processing (chat.py)
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=Config.MAX_TOKENS,
            temperature=Config.TEMPERATURE,
            do_sample=Config.DO_SAMPLE,
            top_p=Config.TOP_P,
            top_k=Config.TOP_K,
            # Generation repetition controls (configurable)
            repetition_penalty=getattr(Config, "REPETITION_PENALTY", 1.2),
            no_repeat_ngram_size=getattr(Config, "NO_REPEAT_NGRAM_SIZE", 3),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,  # Prevent warnings
            return_full_text=False,
            device_map="auto"
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("✓ LLM pipeline initialized successfully")
        logger.info(f"  Max tokens: {Config.MAX_TOKENS}, Temperature: {Config.TEMPERATURE}")
        
        return llm
    
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise


def load_embedding_model(model_name: str = Config.EMBEDDINGS_MODEL):
    """
    Load HuggingFace embedding model.
    
    Args:
        model_name: HuggingFace model name or path
    
    Returns:
        HuggingFaceEmbeddings: Embedding model instance
    
    Raises:
        RuntimeError: If model cannot be loaded
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    try:
        logger.info(f"Loading embedding model: {model_name}...")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info("✓ Embedding model loaded successfully")
        return embeddings
    
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise RuntimeError(f"Could not load embedding model {model_name}: {e}")


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def get_model_device() -> str:
    """
    Get the device being used by the model.
    
    Returns:
        str: Device identifier ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


def get_model_memory_usage(model: AutoModelForCausalLM) -> dict:
    """
    Get memory usage information for a model.
    
    Args:
        model: Model to check
    
    Returns:
        dict: Memory usage information
    """
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "device": "cuda"
            }
        else:
            return {
                "device": "cpu",
                "note": "CPU memory tracking not implemented"
            }
    
    except Exception as e:
        logger.error(f"Failed to get memory usage: {e}")
        return {"error": str(e)}


def log_model_info(model: AutoModelForCausalLM) -> None:
    """
    Log information about a model.
    
    Args:
        model: Model to log info about
    """
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model Information:")
        logger.info(f"  Total Parameters: {total_params:,}")
        logger.info(f"  Trainable Parameters: {trainable_params:,}")
        logger.info(f"  Device: {get_model_device()}")
        
        memory = get_model_memory_usage(model)
        if memory.get("device") == "cuda":
            logger.info(f"  GPU Memory Allocated: {memory['allocated_gb']:.2f} GB")
            logger.info(f"  GPU Memory Reserved: {memory['reserved_gb']:.2f} GB")
    
    except Exception as e:
        logger.error(f"Failed to log model info: {e}")


# ============================================================================
# TOKENIZER UTILITIES
# ============================================================================

def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    """
    Count the number of tokens in text.
    
    Args:
        text: Text to tokenize
        tokenizer: Tokenizer instance
    
    Returns:
        int: Number of tokens
    """
    try:
        tokens = tokenizer.encode(text)
        return len(tokens)
    
    except Exception as e:
        logger.error(f"Failed to count tokens: {e}")
        return 0


def truncate_to_tokens(text: str, tokenizer: AutoTokenizer, max_tokens: int) -> str:
    """
    Truncate text to a maximum number of tokens.
    
    Args:
        text: Text to truncate
        tokenizer: Tokenizer instance
        max_tokens: Maximum number of tokens
    
    Returns:
        str: Truncated text
    """
    try:
        tokens = tokenizer.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        truncated_text = tokenizer.decode(truncated_tokens)
        
        logger.debug(f"Truncated text from {len(tokens)} to {max_tokens} tokens")
        return truncated_text
    
    except Exception as e:
        logger.error(f"Failed to truncate to tokens: {e}")
        return text
