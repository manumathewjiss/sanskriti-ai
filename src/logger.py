"""
Logging configuration module for Sanskrit Scripture RAG Assistant.
Provides centralized logging setup with file and console handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "sanskrit_rag",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Configure and return a logger instance with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (if None, defaults to sanskrit_rag.log)
        level: Logging level (default: logging.INFO)
        console: Whether to log to console (default: True)
        file_output: Whether to log to file (default: True)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output and log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Log all levels to file
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "sanskrit_rag") -> logging.Logger:
    """
    Get a logger instance by name.
    
    Args:
        name: Logger name
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


# Module-level logger
logger = setup_logger()
