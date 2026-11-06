"""
One-time migration script to load scripture data into MongoDB Atlas.
Loads data from combined CSV, creates embeddings, and stores in scripture_vectors collection.
"""

import asyncio
import pandas as pd
from typing import List
from datetime import datetime

from config import Config
from logger import get_logger
from database import get_database, MongoDB
from models import load_embedding_model
from mongodb_vector_store import insert_embeddings, create_vector_index, get_collection_stats
from data_loader import load_and_chunk_itihasa_data

logger = get_logger(__name__)


async def migrate_scripture_data():
    """
    Migrate scripture data from CSV to MongoDB with vector embeddings.
    """
    logger.info("=" * 70)
    logger.info("SCRIPTURE DATA MIGRATION TO MONGODB ATLAS")
    logger.info("=" * 70)
    
    try:
        # Connect to MongoDB
        logger.info("\n1. Connecting to MongoDB Atlas...")
        db = await get_database()
        
        # Check if data already exists
        collection = db.get_collection(Config.SCRIPTURE_COLLECTION)
        existing_count = await db.count_documents(Config.SCRIPTURE_COLLECTION)
        
        if existing_count > 0:
            logger.warning(f"⚠ Collection already contains {existing_count} documents")
            response = input("Do you want to clear and re-migrate? (yes/no): ")
            if response.lower() == 'yes':
                logger.info("Clearing existing data...")
                await db.delete_many(Config.SCRIPTURE_COLLECTION, {})
                logger.info("✓ Existing data cleared")
            else:
                logger.info("Migration cancelled")
                return
        
        # Load scripture data
        logger.info("\n2. Loading scripture data from CSV...")
        try:
            documents = load_and_chunk_itihasa_data(Config.DATA_PATH)
            logger.info(f"✓ Loaded {len(documents)} document chunks")
        except FileNotFoundError:
            logger.error(f"CSV file not found: {Config.DATA_PATH}")
            logger.info("Using default data path: res/combined.csv")
            documents = load_and_chunk_itihasa_data("res/combined.csv")
        
        if not documents:
            raise ValueError("No documents loaded from CSV")
        
        # Extract texts and metadata
        logger.info("\n3. Preparing texts for embedding...")
        texts = [doc.page_content for doc in documents]
        metadata_list = [doc.metadata for doc in documents]
        
        logger.info(f"Texts prepared: {len(texts)} chunks")
        
        # Load embedding model
        logger.info("\n4. Loading embedding model...")
        embedding_model = load_embedding_model(Config.EMBEDDINGS_MODEL)
        logger.info("✓ Embedding model loaded")
        
        # Create embeddings in batches
        logger.info("\n5. Creating embeddings...")
        batch_size = Config.EMBEDDINGS_BATCH_SIZE
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
            
            batch_embeddings = embedding_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        logger.info(f"✓ Created {len(all_embeddings)} embeddings (dim={len(all_embeddings[0])})")
        
        # Insert into MongoDB
        logger.info("\n6. Inserting data into MongoDB...")
        inserted_ids = await insert_embeddings(
            db,
            Config.SCRIPTURE_COLLECTION,
            texts,
            all_embeddings,
            metadata_list
        )
        
        logger.info(f"✓ Inserted {len(inserted_ids)} documents into {Config.SCRIPTURE_COLLECTION}")
        
        # Display vector index info
        logger.info("\n7. Vector Search Index Configuration:")
        index_def = await create_vector_index(db, Config.SCRIPTURE_COLLECTION)
        
        # Get collection stats
        logger.info("\n8. Verifying migration...")
        stats = await get_collection_stats(db, Config.SCRIPTURE_COLLECTION)
        
        logger.info("\n" + "=" * 70)
        logger.info("MIGRATION COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Total documents: {stats['total_documents']}")
        logger.info(f"Vector dimension: {stats['vector_dimension']}")
        logger.info(f"Collection: {Config.SCRIPTURE_COLLECTION}")
        logger.info("\n⚠ IMPORTANT: Create the vector search index in MongoDB Atlas UI:")
        logger.info(f"   Index name: {index_def['name']}")
        logger.info(f"   Index type: vectorSearch")
        logger.info(f"   Path: embedding")
        logger.info(f"   Dimensions: {Config.VECTOR_DIMENSION}")
        logger.info(f"   Similarity: {Config.VECTOR_SIMILARITY_METRIC}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        raise
    
    finally:
        # Close database connection
        await db.close()


async def test_vector_search():
    """
    Test vector search functionality after migration.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TESTING VECTOR SEARCH")
    logger.info("=" * 70)
    
    try:
        # Connect to MongoDB
        db = await get_database()
        
        # Load embedding model
        embedding_model = load_embedding_model(Config.EMBEDDINGS_MODEL)
        
        # Test query
        test_query = "What is dharma?"
        logger.info(f"\nTest query: '{test_query}'")
        
        # Create query embedding
        query_embedding = embedding_model.embed_query(test_query)
        logger.info(f"Query embedding created (dim={len(query_embedding)})")
        
        # Perform vector search
        from mongodb_vector_store import search_scripture
        results = await search_scripture(db, query_embedding, k=3)
        
        logger.info(f"\n✓ Vector search returned {len(results)} results:")
        for i, (text, metadata, score) in enumerate(results, 1):
            logger.info(f"\n--- Result {i} (score: {score:.4f}) ---")
            logger.info(f"Text: {text[:200]}...")
            logger.info(f"Source: {metadata.get('source', 'unknown')}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ VECTOR SEARCH TEST PASSED")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Vector search test failed: {e}", exc_info=True)
        logger.error("\n⚠ Make sure you've created the vector search index in MongoDB Atlas!")
    
    finally:
        await db.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test only
        asyncio.run(test_vector_search())
    else:
        # Run migration
        asyncio.run(migrate_scripture_data())
        
        # Ask if user wants to test
        response = input("\nRun vector search test? (yes/no): ")
        if response.lower() == 'yes':
            asyncio.run(test_vector_search())
