"""
Data loading and processing module for Sanskrit Scripture RAG Assistant.
Handles CSV loading, document chunking, and metadata extraction.
"""

import os
import pandas as pd
from typing import List, Optional, Tuple
from langchain_core.documents import Document

from logger import get_logger
from utils import clean_text, keep_sanskrit_english_chars

logger = get_logger(__name__)


# ============================================================================
# CSV DATA LOADING
# ============================================================================

def load_csv_data(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Load CSV file with error handling.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        pd.DataFrame: Loaded data or None on error
    """
    try:
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return None
        
        df = pd.read_csv(csv_path, encoding='utf-8')
        logger.info(f"✓ Loaded CSV with {len(df)} rows from {csv_path}")
        
        return df
    
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return None


def validate_csv_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that CSV has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        bool: True if all required columns exist
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.info(f"Available columns: {list(df.columns)}")
        return False
    
    logger.debug(f"✓ All required columns present: {required_columns}")
    return True


# ============================================================================
# DOCUMENT CREATION
# ============================================================================

def load_and_chunk_itihasa_data(
    csv_path: str,
    required_columns: Optional[List[str]] = None
) -> List[Document]:
    """
    Load CSV data and create LangChain Documents with metadata.
    
    Creates documents using Sanskrit text as primary content for embedding,
    with English translation and source metadata included.
    
    Args:
        csv_path: Path to CSV file containing shloka data
        required_columns: List of required column names (default: standard columns)
    
    Returns:
        List of LangChain Documents with metadata
    
    Example:
        >>> docs = load_and_chunk_itihasa_data("data/combined.csv")
        >>> print(len(docs))  # Number of documents created
        >>> print(docs[0].metadata)  # View metadata of first document
    """
    # Define required columns
    if required_columns is None:
        required_columns = ['sn', 'en', 'book', 'chapter', 'shloka']
    
    try:
        # Load CSV
        df = load_csv_data(csv_path)
        
        if df is None or len(df) == 0:
            logger.warning(f"No data loaded from {csv_path}")
            return []
        
        # Validate columns
        if not validate_csv_columns(df, required_columns):
            logger.error("CSV validation failed")
            return []
        
        documents = []
        error_count = 0
        
        for index, row in df.iterrows():
            try:
                # Extract content and metadata
                sanskrit_text = str(row.get('sn', '')).strip()
                english_text = str(row.get('en', '')).strip()
                
                # Skip empty documents
                if not sanskrit_text:
                    logger.debug(f"Skipping row {index}: empty Sanskrit text")
                    continue
                
                # Clean text
                sanskrit_text = clean_text(sanskrit_text)
                english_text = clean_text(english_text)
                english_text = keep_sanskrit_english_chars(english_text)
                
                # Create metadata
                metadata = {
                    "book": str(row.get('book', 'Unknown')),
                    "chapter": str(row.get('chapter', 'Unknown')),
                    "shloka_num": str(row.get('shloka', 'Unknown')),
                    "english_text": english_text,
                    "source_file": csv_path,
                    "row_index": index
                }
                
                # Create document
                doc = Document(page_content=sanskrit_text, metadata=metadata)
                documents.append(doc)
            
            except Exception as e:
                logger.warning(f"Error processing row {index}: {e}")
                error_count += 1
                continue
        
        logger.info(f"✓ Created {len(documents)} documents from {csv_path}")
        
        if error_count > 0:
            logger.warning(f"⚠ Encountered {error_count} errors while processing rows")
        
        return documents
    
    except Exception as e:
        logger.error(f"Critical error in load_and_chunk_itihasa_data: {e}")
        return []


# ============================================================================
# DOCUMENT FILTERING & PROCESSING
# ============================================================================

def filter_documents_by_book(
    documents: List[Document],
    book_name: str
) -> List[Document]:
    """
    Filter documents by book name.
    
    Args:
        documents: List of documents to filter
        book_name: Book name to filter by
    
    Returns:
        List of filtered documents
    """
    filtered = [
        doc for doc in documents
        if doc.metadata.get('book', '').lower() == book_name.lower()
    ]
    
    logger.debug(f"Filtered {len(filtered)} documents for book: {book_name}")
    return filtered


def filter_documents_by_chapter(
    documents: List[Document],
    chapter_num: str
) -> List[Document]:
    """
    Filter documents by chapter number.
    
    Args:
        documents: List of documents to filter
        chapter_num: Chapter number to filter by
    
    Returns:
        List of filtered documents
    """
    filtered = [
        doc for doc in documents
        if doc.metadata.get('chapter', '') == str(chapter_num)
    ]
    
    logger.debug(f"Filtered {len(filtered)} documents for chapter: {chapter_num}")
    return filtered


def get_document_summary(documents: List[Document]) -> dict:
    """
    Get summary statistics about loaded documents.
    
    Args:
        documents: List of documents to summarize
    
    Returns:
        dict: Summary statistics
    """
    if not documents:
        return {"total": 0}
    
    books = set()
    chapters = set()
    total_chars = 0
    
    for doc in documents:
        books.add(doc.metadata.get('book', 'Unknown'))
        chapters.add(doc.metadata.get('chapter', 'Unknown'))
        total_chars += len(doc.page_content)
    
    summary = {
        "total_documents": len(documents),
        "unique_books": len(books),
        "unique_chapters": len(chapters),
        "total_characters": total_chars,
        "average_doc_length": total_chars // len(documents) if documents else 0,
        "books": sorted(list(books)),
        "chapters": sorted(list(chapters), key=lambda x: (str(x), x))
    }
    
    logger.info(f"Document summary: {summary['total_documents']} total, "
                f"{summary['unique_books']} books, "
                f"{summary['unique_chapters']} chapters")
    
    return summary


# ============================================================================
# DOCUMENT UTILITIES
# ============================================================================

def create_citation_string(metadata: dict) -> str:
    """
    Create a citation string from document metadata.
    
    Args:
        metadata: Document metadata dictionary
    
    Returns:
        str: Formatted citation string
    
    Example:
        >>> citation = create_citation_string({"book": "Bhagavad Gita", "chapter": "2", "shloka_num": "47"})
        >>> print(citation)  # "Bhagavad Gita 2.47"
    """
    book = metadata.get('book', 'Unknown')
    chapter = metadata.get('chapter', 'Unknown')
    shloka = metadata.get('shloka_num', 'Unknown')
    
    return f"{book} {chapter}.{shloka}"


def format_document_for_display(doc: Document) -> str:
    """
    Format a document for display in CLI.
    
    Args:
        doc: Document to format
    
    Returns:
        str: Formatted document string
    """
    citation = create_citation_string(doc.metadata)
    sanskrit = doc.page_content
    english = doc.metadata.get('english_text', 'Translation not available')
    
    formatted = f"""
Citation: {citation}
Sanskrit: {sanskrit}
English: {english}
"""
    return formatted.strip()


def get_unique_books(documents: List[Document]) -> List[str]:
    """
    Get list of unique books from documents.
    
    Args:
        documents: List of documents
    
    Returns:
        List of unique book names
    """
    books = set(doc.metadata.get('book', 'Unknown') for doc in documents)
    return sorted(list(books))


def get_unique_chapters(documents: List[Document], book: Optional[str] = None) -> List[str]:
    """
    Get list of unique chapters from documents.
    
    Args:
        documents: List of documents
        book: Optional book name to filter by
    
    Returns:
        List of unique chapter numbers
    """
    if book:
        chapters = set(
            doc.metadata.get('chapter', 'Unknown')
            for doc in documents
            if doc.metadata.get('book', '') == book
        )
    else:
        chapters = set(doc.metadata.get('chapter', 'Unknown') for doc in documents)
    
    return sorted(list(chapters), key=lambda x: (str(x), x))
