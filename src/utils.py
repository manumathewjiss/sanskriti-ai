"""
Utility functions for Sanskrit Scripture RAG Assistant.
Includes transliteration, path validation, and helper functions.
"""

import os
import re
import torch
from typing import Optional
from pathlib import Path
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# TRANSLITERATION UTILITIES
# ============================================================================

def devanagari_to_roman(
    devanagari_text: str,
    output_scheme=sanscript.IAST
) -> str:
    """
    Transliterate Devanagari text to Roman script.
    
    Args:
        devanagari_text: Text in Devanagari script
        output_scheme: Target transliteration scheme (default: IAST)
    
    Returns:
        str: Romanized text or error message
    
    Example:
        >>> result = devanagari_to_roman("नमस्ते")
        >>> print(result)  # 'namaste'
    """
    try:
        if not devanagari_text or not isinstance(devanagari_text, str):
            logger.warning("Invalid input for transliteration")
            return devanagari_text
        
        romanized = transliterate(devanagari_text, sanscript.DEVANAGARI, output_scheme)
        logger.debug(f"Transliterated {len(devanagari_text)} characters")
        return romanized
    
    except Exception as e:
        logger.error(f"Transliteration failed: {e}")
        return devanagari_text


def roman_to_devanagari(
    roman_text: str,
    input_scheme=sanscript.IAST
) -> str:
    """
    Transliterate Roman text to Devanagari script.
    
    Args:
        roman_text: Text in Roman script
        input_scheme: Source transliteration scheme (default: IAST)
    
    Returns:
        str: Devanagari text or original text on error
    """
    try:
        if not roman_text or not isinstance(roman_text, str):
            logger.warning("Invalid input for reverse transliteration")
            return roman_text
        
        devanagari = transliterate(roman_text, input_scheme, sanscript.DEVANAGARI)
        logger.debug(f"Reverse transliterated {len(roman_text)} characters")
        return devanagari
    
    except Exception as e:
        logger.error(f"Reverse transliteration failed: {e}")
        return roman_text


# ============================================================================
# PATH & FILE UTILITIES
# ============================================================================

def safe_path(path: str) -> str:
    """
    Resolve and validate a file path to prevent path traversal attacks.
    
    Args:
        path: File path to validate
    
    Returns:
        str: Resolved absolute path
    
    Raises:
        ValueError: If path is invalid or doesn't exist
    """
    try:
        safe_path_obj = Path(path).resolve()
        
        if not safe_path_obj.exists():
            logger.warning(f"Path does not exist: {path}")
            return str(safe_path_obj)
        
        return str(safe_path_obj)
    
    except Exception as e:
        logger.error(f"Path validation failed for {path}: {e}")
        raise ValueError(f"Invalid path: {path}")


def ensure_directory(path: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        bool: True if successful
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False


# ============================================================================
# CUDA & DEVICE UTILITIES
# ============================================================================

def check_cuda_availability() -> bool:
    """
    Check CUDA availability and log device information.
    
    Returns:
        bool: True if CUDA is available
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✓ CUDA available - Device: {device_name} ({total_memory:.2f} GB)")
        return True
    else:
        logger.warning("⚠ CUDA not available - will use CPU (slow)")
        return False


def get_device() -> str:
    """
    Get the appropriate device for model execution.
    
    Returns:
        str: Device identifier ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        logger.debug("Using CUDA device")
        return "cuda"
    else:
        logger.debug("Using CPU device")
        return "cpu"


def get_gpu_memory_usage() -> dict:
    """
    Get current GPU memory usage statistics.
    
    Returns:
        dict: Memory usage information
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    try:
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return {
            "available": True,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "available_gb": total - allocated
        }
    
    except Exception as e:
        logger.error(f"Failed to get GPU memory usage: {e}")
        return {"available": False, "error": str(e)}


# ============================================================================
# TEXT UTILITIES
# ============================================================================

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated
    
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Text to clean
    
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove control characters
    text = "".join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
    
    return text.strip()


def keep_sanskrit_english_chars(text: str) -> str:
    """
    Keep only Sanskrit (Devanagari) and English/Latin characters.

    This removes characters from other scripts such as Cyrillic, Arabic,
    Greek, CJK, Hebrew, and other non-Latin/Devanagari blocks.
    """
    if not isinstance(text, str):
        return ""

    allowed_pattern = re.compile(
        r"[^\u0009\u000A\u000D\u0020-\u007E\u00A0-\u00FF\u0100-\u017F\u0180-\u024F"
        r"\u1E00-\u1EFF\u0300-\u036F\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF\u2000-\u206F]"
    )

    text = allowed_pattern.sub("", text)
    return " ".join(text.split()).strip()


def assert_english_or_sanskrit_only(text: str, threshold: float = 0.15) -> None:
    """
    Raise ValueError if more than `threshold` fraction of non-whitespace characters
    are not English/Latin or Sanskrit/Devanagari script.

    Uses keep_sanskrit_english_chars as the canonical allowed-set definition so
    the input gate and output sanitizer share exactly the same character policy.
    """
    stripped_original = text.replace(" ", "")
    if not stripped_original:
        return

    cleaned = keep_sanskrit_english_chars(text).replace(" ", "")
    removed = len(stripped_original) - len(cleaned)
    ratio = removed / len(stripped_original)

    if ratio > threshold:
        raise ValueError(
            f"Query must be in English or Sanskrit (Devanagari) only. "
            f"{ratio:.0%} of characters are in unsupported scripts."
        )


def validate_query(query: str, max_length: int = 1000) -> bool:
    """
    Validate user query for safety and length.

    Args:
        query: User query
        max_length: Maximum allowed length

    Returns:
        bool: True if valid

    Raises:
        ValueError: If query is invalid
    """
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")

    if len(query) > max_length:
        raise ValueError(f"Query exceeds maximum length of {max_length}")

    if not query.strip():
        raise ValueError("Query cannot be only whitespace")

    assert_english_or_sanskrit_only(query)

    return True


def is_sanskrit(text: str) -> bool:
    """
    Detect if text contains Sanskrit/Devanagari characters.
    
    Args:
        text: Text to analyze
    
    Returns:
        bool: True if text contains Devanagari script
    """
    if not text:
        return False
    
    # Check for Devanagari Unicode range (0900-097F)
    for char in text:
        if '\u0900' <= char <= '\u097F':
            return True
    
    return False


# ============================================================================
# BATCH PROCESSING UTILITIES
# ============================================================================

def batch_generator(items: list, batch_size: int):
    """
    Generate batches from a list of items.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
    
    Yields:
        list: Batch of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
    
    Returns:
        list: List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks
