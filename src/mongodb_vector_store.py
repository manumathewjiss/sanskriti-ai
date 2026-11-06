"""
MongoDB Atlas Vector Search module.
Handles vector similarity search and embedding storage.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from config import Config
from logger import get_logger
from database import MongoDB

logger = get_logger(__name__)


async def vector_search(
    db: MongoDB,
    collection_name: str,
    query_embedding: List[float],
    k: int = 5,
    filter_query: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search using MongoDB Atlas Vector Search.
    
    Args:
        db: MongoDB instance
        collection_name: Name of collection to search
        query_embedding: Query embedding vector
        k: Number of results to return
        filter_query: Additional filter criteria (optional)
    
    Returns:
        List of matching documents with similarity scores
    """
    try:
        collection = db.get_collection(collection_name)
        
        # Build aggregation pipeline for vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": f"{collection_name}_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": k * 10,  # Oversample for better results
                    "limit": k
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        # Add filter stage if provided
        if filter_query:
            pipeline.insert(1, {"$match": filter_query})
        
        # Execute search
        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list(length=None)
        
        logger.debug(f"Vector search returned {len(results)} results from {collection_name}")
        return results
    
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise


async def insert_embeddings(
    db: MongoDB,
    collection_name: str,
    texts: List[str],
    embeddings: List[List[float]],
    metadata_list: Optional[List[Dict[str, Any]]] = None
) -> List[Any]:
    """
    Insert text chunks with embeddings into MongoDB collection.
    
    Args:
        db: MongoDB instance
        collection_name: Name of collection
        texts: List of text chunks
        embeddings: List of embedding vectors (same length as texts)
        metadata_list: Optional list of metadata dicts (same length as texts)
    
    Returns:
        List of inserted document IDs
    """
    if len(texts) != len(embeddings):
        raise ValueError("texts and embeddings must have same length")
    
    if metadata_list and len(metadata_list) != len(texts):
        raise ValueError("metadata_list must have same length as texts")
    
    # Build documents
    documents = []
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        doc = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata_list[i] if metadata_list else {},
            "created_at": datetime.utcnow()
        }
        documents.append(doc)
    
    # Bulk insert
    try:
        inserted_ids = await db.insert_many(collection_name, documents)
        logger.info(f"Inserted {len(inserted_ids)} documents into {collection_name}")
        return inserted_ids
    
    except Exception as e:
        logger.error(f"Failed to insert embeddings: {e}")
        raise


async def create_vector_index(db: MongoDB, collection_name: str):
    """
    Create a vector search index for a collection.
    
    Note: This function provides the index definition. The actual index
    must be created in MongoDB Atlas UI or via Atlas API.
    
    Args:
        db: MongoDB instance
        collection_name: Name of collection
    
    Returns:
        Index definition (for reference)
    """
    index_definition = {
        "name": f"{collection_name}_vector_index",
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": Config.VECTOR_DIMENSION,
                    "similarity": Config.VECTOR_SIMILARITY_METRIC
                }
            ]
        }
    }
    
    logger.info(f"Vector index definition for {collection_name}:")
    logger.info(f"Index name: {index_definition['name']}")
    logger.info(f"Vector dimension: {Config.VECTOR_DIMENSION}")
    logger.info(f"Similarity metric: {Config.VECTOR_SIMILARITY_METRIC}")
    logger.info("⚠ Note: Index must be created in MongoDB Atlas UI")
    
    return index_definition


async def search_scripture(
    db: MongoDB,
    query_embedding: List[float],
    k: int = None
) -> List[Tuple[str, Dict[str, Any], float]]:
    """
    Search scripture vectors for relevant passages.
    
    Args:
        db: MongoDB instance
        query_embedding: Query embedding vector
        k: Number of results (defaults to Config.SCRIPTURE_RETRIEVAL_K)
    
    Returns:
        List of tuples: (text, metadata, score)
    """
    if k is None:
        k = Config.SCRIPTURE_RETRIEVAL_K
    
    results = await vector_search(
        db,
        Config.SCRIPTURE_COLLECTION,
        query_embedding,
        k=k
    )
    
    return [(r['text'], r.get('metadata', {}), r.get('score', 0.0)) for r in results]


async def search_user_context(
    db: MongoDB,
    query_embedding: List[float],
    k: int = None,
    filename_filter: Optional[str] = None
) -> List[Tuple[str, Dict[str, Any], float]]:
    """
    Search user context vectors for relevant information.
    
    Args:
        db: MongoDB instance
        query_embedding: Query embedding vector
        k: Number of results (defaults to Config.CONTEXT_RETRIEVAL_K)
        filename_filter: Optional filter by original filename
    
    Returns:
        List of tuples: (text, metadata, score)
    """
    if k is None:
        k = Config.CONTEXT_RETRIEVAL_K
    
    filter_query = None
    if filename_filter:
        filter_query = {"metadata.original_filename": filename_filter}
    
    results = await vector_search(
        db,
        Config.USER_CONTEXT_COLLECTION,
        query_embedding,
        k=k,
        filter_query=filter_query
    )
    
    return [(r['text'], r.get('metadata', {}), r.get('score', 0.0)) for r in results]


async def get_collection_stats(db: MongoDB, collection_name: str) -> Dict[str, Any]:
    """
    Get statistics about a vector collection.
    
    Args:
        db: MongoDB instance
        collection_name: Name of collection
    
    Returns:
        Dictionary with collection statistics
    """
    collection = db.get_collection(collection_name)
    
    try:
        total_docs = await db.count_documents(collection_name)
        
        # Get a sample document to check structure
        sample = await collection.find_one({})
        
        stats = {
            "collection": collection_name,
            "total_documents": total_docs,
            "has_embeddings": bool(sample and "embedding" in sample),
            "vector_dimension": len(sample.get("embedding", [])) if sample else 0
        }
        
        logger.info(f"Collection stats for {collection_name}: {stats}")
        return stats
    
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        return {
            "collection": collection_name,
            "error": str(e)
        }
