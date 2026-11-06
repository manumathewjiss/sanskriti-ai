"""
Optimized migration with better chunking strategy.
Combines multiple verses into larger chunks to reduce document count and storage.
"""

import asyncio
import sys
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

from config import Config
from logger import get_logger
from database import get_database
from models import load_embedding_model
from mongodb_vector_store import insert_embeddings, get_collection_stats
from data_loader import load_csv_data

logger = get_logger(__name__)


def create_optimized_chunks(df, target_chunk_size=800):
    """
    Combine multiple verses into larger chunks to reduce document count.
    
    Instead of 93k tiny documents, create ~10-15k larger chunks.
    This drastically reduces MongoDB storage overhead.
    """
    chunks = []
    current_chunk_text = []
    current_chunk_metadata = []
    current_size = 0
    
    for idx, row in df.iterrows():
        sanskrit = str(row.get('sn', '')).strip()
        if not sanskrit:
            continue
            
        verse_text = sanskrit
        verse_len = len(verse_text)
        
        # If adding this verse exceeds target, save current chunk
        if current_size + verse_len > target_chunk_size and current_chunk_text:
            # Combine all verses in chunk
            combined_text = " । ".join(current_chunk_text)
            
            # Use metadata from first verse in chunk
            metadata = {
                "book": str(current_chunk_metadata[0].get('book', 'Unknown')),
                "chapter": str(current_chunk_metadata[0].get('chapter', 'Unknown')),
                "shloka_start": str(current_chunk_metadata[0].get('shloka', 'Unknown')),
                "shloka_end": str(current_chunk_metadata[-1].get('shloka', 'Unknown')),
                "verse_count": len(current_chunk_text),
                "english_text": current_chunk_metadata[0].get('en', '')
            }
            
            chunks.append((combined_text, metadata))
            
            # Start new chunk
            current_chunk_text = []
            current_chunk_metadata = []
            current_size = 0
        
        # Add verse to current chunk
        current_chunk_text.append(verse_text)
        current_chunk_metadata.append(row)
        current_size += verse_len
    
    # Add final chunk
    if current_chunk_text:
        combined_text = " । ".join(current_chunk_text)
        metadata = {
            "book": str(current_chunk_metadata[0].get('book', 'Unknown')),
            "chapter": str(current_chunk_metadata[0].get('chapter', 'Unknown')),
            "shloka_start": str(current_chunk_metadata[0].get('shloka', 'Unknown')),
            "shloka_end": str(current_chunk_metadata[-1].get('shloka', 'Unknown')),
            "verse_count": len(current_chunk_text),
            "english_text": current_chunk_metadata[0].get('en', '')
        }
        chunks.append((combined_text, metadata))
    
    return chunks


async def optimized_migration():
    """Migrate with optimized chunking."""
    print("\n" + "=" * 70)
    print("OPTIMIZED SCRIPTURE MIGRATION")
    print("Strategy: Combine verses → Larger chunks → Less MongoDB overhead")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Embedding model: {Config.EMBEDDINGS_MODEL}")
    print(f"Vector dimension: {Config.VECTOR_DIMENSION}")
    print(f"Target chunk size: 800 chars")
    print("=" * 70 + "\n")
    
    try:
        # 1. Connect to MongoDB
        print("[1/6] Connecting to MongoDB...")
        sys.stdout.flush()
        db = await get_database()
        print("✓ Connected!\n")
        
        # Check/warn about existing data
        existing_count = await db.count_documents(Config.SCRIPTURE_COLLECTION)
        if existing_count > 0:
            print(f"⚠ WARNING: {existing_count:,} existing documents found!")
            print("Please delete them in MongoDB Atlas UI first.\n")
            return
        
        # 2. Load CSV
        print("[2/6] Loading scripture CSV...")
        sys.stdout.flush()
        df = load_csv_data(Config.DATA_PATH)
        print(f"✓ Loaded {len(df):,} verses\n")
        
        # 3. Create optimized chunks
        print("[3/6] Creating optimized chunks (combining verses)...")
        sys.stdout.flush()
        chunks = create_optimized_chunks(df, target_chunk_size=800)
        texts = [c[0] for c in chunks]
        metadata_list = [c[1] for c in chunks]
        
        print(f"✓ Created {len(chunks):,} chunks from {len(df):,} verses")
        print(f"  Reduction: {len(df)} → {len(chunks)} ({100 * (1 - len(chunks)/len(df)):.1f}% fewer docs)")
        print(f"  Avg chunk size: {sum(len(t) for t in texts) // len(texts)} chars\n")
        sys.stdout.flush()
        
        # 4. Load embedding model
        print("[4/6] Loading embedding model...")
        sys.stdout.flush()
        embedding_model = load_embedding_model(Config.EMBEDDINGS_MODEL)
        print("✓ Model loaded!\n")
        
        # 5. Create embeddings
        print("[5/6] Creating embeddings...")
        batch_size = Config.EMBEDDINGS_BATCH_SIZE
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch_texts = texts[i:i + batch_size]
            
            print(f"  Batch {batch_num}/{total_batches} ({len(batch_texts)} chunks)...", end='', flush=True)
            batch_embeddings = embedding_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            print(" ✓")
            sys.stdout.flush()
        
        print(f"✓ Created {len(all_embeddings):,} embeddings\n")
        
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
        
        # Get stats
        stats = await get_collection_stats(db, Config.SCRIPTURE_COLLECTION)
        
        print("\n" + "=" * 70)
        print("MIGRATION COMPLETE!")
        print("=" * 70)
        print(f"Documents: {stats['total_documents']:,}")
        print(f"Dimension: {stats['vector_dimension']}")
        print(f"Storage estimate: ~{stats['total_documents'] * 1.6 / 1024:.1f} MB (vs ~150MB before)")
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
    asyncio.run(optimized_migration())
