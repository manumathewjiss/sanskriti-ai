"""
Migration script with verbose logging to debug silent failures.
"""

import asyncio
import pandas as pd
import sys
from datetime import datetime

# Add verbose logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

from config import Config
from logger import get_logger
from database import get_database
from models import load_embedding_model
from mongodb_vector_store import insert_embeddings, create_vector_index, get_collection_stats
from data_loader import load_and_chunk_itihasa_data

logger = get_logger(__name__)


async def migrate_with_progress():
    """Migrate scripture data with detailed progress reporting."""
    print("\n" + "=" * 70)
    print("SCRIPTURE DATA MIGRATION (VERBOSE MODE)")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Embedding model: {Config.EMBEDDINGS_MODEL}")
    print(f"Vector dimension: {Config.VECTOR_DIMENSION}")
    print("=" * 70 + "\n")
    
    try:
        # 1. Connect to MongoDB
        print("[1/6] Connecting to MongoDB Atlas...")
        sys.stdout.flush()
        db = await get_database()
        print("✓ Connected!\n")
        sys.stdout.flush()
        
        # Check existing data
        existing_count = await db.count_documents(Config.SCRIPTURE_COLLECTION)
        if existing_count > 0:
            print(f"⚠ Warning: Collection has {existing_count:,} existing documents")
            print("These will be overwritten if you continue.\n")
        
        # 2. Load CSV data
        print("[2/6] Loading scripture CSV...")
        sys.stdout.flush()
        documents = load_and_chunk_itihasa_data(Config.DATA_PATH)
        print(f"✓ Loaded {len(documents):,} document chunks\n")
        sys.stdout.flush()
        
        # 3. Prepare texts
        print("[3/6] Extracting texts...")
        sys.stdout.flush()
        texts = [doc.page_content for doc in documents]
        metadata_list = [doc.metadata for doc in documents]
        print(f"✓ Prepared {len(texts):,} texts\n")
        sys.stdout.flush()
        
        # 4. Load embedding model (THIS IS THE SLOW PART)
        print("[4/6] Loading embedding model (this may take 5-10 minutes)...")
        print(f"Model: {Config.EMBEDDINGS_MODEL}")
        print("Downloading from HuggingFace if not cached...")
        sys.stdout.flush()
        
        embedding_model = load_embedding_model(Config.EMBEDDINGS_MODEL)
        print("✓ Model loaded!\n")
        sys.stdout.flush()
        
        # 5. Create embeddings in batches
        print("[5/6] Creating embeddings...")
        batch_size = Config.EMBEDDINGS_BATCH_SIZE
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch_texts = texts[i:i + batch_size]
            
            print(f"  Batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...", end='', flush=True)
            batch_embeddings = embedding_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            print(" ✓")
            sys.stdout.flush()
        
        print(f"✓ Created {len(all_embeddings):,} embeddings (dim={len(all_embeddings[0])})\n")
        sys.stdout.flush()
        
        # 6. Insert into MongoDB
        print("[6/6] Inserting into MongoDB...")
        sys.stdout.flush()
        inserted_ids = await insert_embeddings(
            db,
            Config.SCRIPTURE_COLLECTION,
            texts,
            all_embeddings,
            metadata_list
        )
        print(f"✓ Inserted {len(inserted_ids):,} documents\n")
        sys.stdout.flush()
        
        # Get stats
        stats = await get_collection_stats(db, Config.SCRIPTURE_COLLECTION)
        
        print("\n" + "=" * 70)
        print("MIGRATION COMPLETE!")
        print("=" * 70)
        print(f"Documents: {stats['total_documents']:,}")
        print(f"Dimension: {stats['vector_dimension']}")
        print(f"End time: {datetime.now()}")
        print("\n⚠ NEXT STEP: Create vector search index in MongoDB Atlas UI")
        print(f"   Index name: {Config.SCRIPTURE_COLLECTION}_vector_index")
        print(f"   Dimensions: {Config.VECTOR_DIMENSION}")
        print(f"   Similarity: {Config.VECTOR_SIMILARITY_METRIC}")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(migrate_with_progress())
