"""
Context processing module for handling PDF, DOCX, and text uploads.
Extracts text, chunks it, creates embeddings, and stores in MongoDB.
"""

import io
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import PyPDF2
from docx import Document

from config import Config
from logger import get_logger
from database import MongoDB
from models import load_embedding_model
from mongodb_vector_store import insert_embeddings

logger = get_logger(__name__)

# Global embedding model (loaded once)
_embedding_model = None


def get_embedding_model():
    """Get or load the embedding model (singleton pattern)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model...")
        _embedding_model = load_embedding_model(Config.EMBEDDINGS_MODEL)
        logger.info("✓ Embedding model loaded")
    return _embedding_model


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF file bytes.
    
    Args:
        file_bytes: PDF file content as bytes
    
    Returns:
        Extracted text
    
    Raises:
        Exception: If PDF processing fails
    """
    try:
        pdf_file = io.BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_parts = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():
                text_parts.append(text)
        
        full_text = "\n\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} characters from PDF ({len(pdf_reader.pages)} pages)")
        return full_text
    
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract text from DOCX file bytes.
    
    Args:
        file_bytes: DOCX file content as bytes
    
    Returns:
        Extracted text
    
    Raises:
        Exception: If DOCX processing fails
    """
    try:
        docx_file = io.BytesIO(file_bytes)
        doc = Document(docx_file)
        
        text_parts = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                text_parts.append(text)
        
        full_text = "\n\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} characters from DOCX ({len(doc.paragraphs)} paragraphs)")
        return full_text
    
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX: {e}")
        raise


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk (defaults to Config.CONTEXT_CHUNK_SIZE)
        overlap: Number of overlapping characters (defaults to Config.CONTEXT_OVERLAP)
    
    Returns:
        List of text chunks
    """
    if chunk_size is None:
        chunk_size = Config.CONTEXT_CHUNK_SIZE
    if overlap is None:
        overlap = Config.CONTEXT_OVERLAP
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('. ')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size // 2:  # Only break if it's not too early
                chunk = chunk[:break_point + 1]
                end = start + len(chunk)
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        # Prevent infinite loop
        if start <= end - chunk_size:
            start = end
    
    logger.info(f"Split text into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


def create_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for a list of texts.
    
    Args:
        texts: List of text strings
    
    Returns:
        List of embedding vectors
    """
    try:
        model = get_embedding_model()
        
        # HuggingFace embeddings encode method
        embeddings = model.embed_documents(texts)
        
        logger.info(f"Created {len(embeddings)} embeddings (dim={len(embeddings[0])})")
        return embeddings
    
    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        raise


async def store_user_context(
    db: MongoDB,
    text: str,
    filename: str,
    file_type: str
) -> Dict[str, Any]:
    """
    Process text, create embeddings, and store in MongoDB.
    
    Args:
        db: MongoDB instance
        text: Text content to store
        filename: Original filename (for reference)
        file_type: File type ("pdf", "docx", "text")
    
    Returns:
        Dictionary with storage information
    """
    try:
        # Chunk the text
        chunks = chunk_text(text)
        
        if not chunks:
            raise ValueError("No text chunks generated")
        
        # Create embeddings
        embeddings = create_embeddings(chunks)
        
        # Prepare metadata for each chunk
        metadata_list = []
        for i in range(len(chunks)):
            metadata_list.append({
                "original_filename": filename,
                "file_type": file_type,
                "upload_date": datetime.utcnow(),
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
        
        # Store in MongoDB
        inserted_ids = await insert_embeddings(
            db,
            Config.USER_CONTEXT_COLLECTION,
            chunks,
            embeddings,
            metadata_list
        )
        
        result = {
            "filename": filename,
            "file_type": file_type,
            "chunks_stored": len(inserted_ids),
            "total_characters": len(text),
            "uploaded_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Stored user context: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Failed to store user context: {e}")
        raise


async def process_file_upload(
    db: MongoDB,
    file_bytes: bytes,
    filename: str,
    file_type: str
) -> Dict[str, Any]:
    """
    Process uploaded file (PDF, DOCX, or TXT) and store as vectors.
    Also saves file locally for backup.
    
    Args:
        db: MongoDB instance
        file_bytes: File content bytes
        filename: Original filename
        file_type: File extension ("pdf", "docx", "txt")
    
    Returns:
        Dictionary with processing results
    """
    try:
        # Save file locally for backup (if enabled)
        local_path = None
        if Config.KEEP_UPLOADED_FILES:
            Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            local_path = Config.UPLOAD_DIR / filename
            
            async with aiofiles.open(local_path, 'wb') as f:
                await f.write(file_bytes)
            logger.info(f"✓ Saved file locally: {local_path}")
        
        # Extract text based on file type
        if file_type == "pdf":
            text = extract_text_from_pdf(file_bytes)
        elif file_type == "docx":
            text = extract_text_from_docx(file_bytes)
        elif file_type == "txt":
            text = file_bytes.decode('utf-8')
            logger.info(f"Loaded {len(text)} characters from TXT file")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        if not text or len(text.strip()) < 10:
            raise ValueError("Extracted text is too short or empty")
        
        # Store the context
        result = await store_user_context(db, text, filename, file_type)
        
        # Add local path to result
        if local_path:
            result['local_path'] = str(local_path)
        
        logger.info(f"✓ Successfully processed {filename} ({file_type})")
        return result
    
    except Exception as e:
        logger.error(f"Failed to process file upload: {e}")
        raise


async def list_user_contexts(db: MongoDB) -> List[Dict[str, Any]]:
    """
    List all user-uploaded contexts with summary information.
    
    Args:
        db: MongoDB instance
    
    Returns:
        List of context summaries
    """
    try:
        collection = db.get_collection(Config.USER_CONTEXT_COLLECTION)
        
        # Aggregate to get unique files and their chunk counts
        pipeline = [
            {
                "$group": {
                    "_id": "$metadata.original_filename",
                    "file_type": {"$first": "$metadata.file_type"},
                    "upload_date": {"$first": "$metadata.upload_date"},
                    "chunk_count": {"$sum": 1},
                    "sample_chunk_id": {"$first": "$_id"}
                }
            },
            {
                "$sort": {"upload_date": -1}
            }
        ]
        
        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list(length=None)
        
        # Format results
        contexts = []
        for r in results:
            contexts.append({
                "filename": r["_id"],
                "file_type": r["file_type"],
                "upload_date": r["upload_date"].isoformat() if r["upload_date"] else None,
                "chunk_count": r["chunk_count"],
                "id": str(r["sample_chunk_id"])  # Use first chunk ID as reference
            })
        
        logger.info(f"Found {len(contexts)} user contexts")
        return contexts
    
    except Exception as e:
        logger.error(f"Failed to list user contexts: {e}")
        raise


async def delete_user_context(db: MongoDB, filename: str) -> int:
    """
    Delete all chunks associated with a filename.
    
    Args:
        db: MongoDB instance
        filename: Original filename to delete
    
    Returns:
        Number of deleted chunks
    """
    try:
        deleted_count = await db.delete_many(
            Config.USER_CONTEXT_COLLECTION,
            {"metadata.original_filename": filename}
        )
        
        logger.info(f"Deleted {deleted_count} chunks for file: {filename}")
        return deleted_count
    
    except Exception as e:
        logger.error(f"Failed to delete user context: {e}")
        raise
