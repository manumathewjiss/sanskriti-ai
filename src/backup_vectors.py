"""
Backup utility for exporting MongoDB vectors to local files.
Enables offline operation and data recovery.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from database import get_database
from config import Config
from logger import get_logger

logger = get_logger(__name__)


async def backup_collection(collection_name: str, output_dir: Path):
    """
    Backup a MongoDB collection to local JSON files.
    
    Args:
        collection_name: Name of collection to backup
        output_dir: Directory to save backup files
    """
    db = await get_database()
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all documents
        logger.info(f"Backing up {collection_name}...")
        collection = db.get_collection(collection_name)
        
        # Count documents
        total_docs = await db.count_documents(collection_name, {})
        logger.info(f"Found {total_docs:,} documents")
        
        if total_docs == 0:
            logger.warning(f"No documents to backup in {collection_name}")
            return
        
        # Fetch in batches to avoid memory issues
        batch_size = 1000
        batch_num = 0
        total_saved = 0
        
        cursor = collection.find({})
        
        while True:
            # Get batch
            batch = await cursor.to_list(length=batch_size)
            if not batch:
                break
            
            # Save batch to file
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"{collection_name}_{timestamp}_batch{batch_num:04d}.json"
            filepath = output_dir / filename
            
            # Convert ObjectId to string for JSON serialization
            for doc in batch:
                doc['_id'] = str(doc['_id'])
                if 'created_at' in doc:
                    doc['created_at'] = doc['created_at'].isoformat()
                if 'upload_date' in doc.get('metadata', {}):
                    doc['metadata']['upload_date'] = doc['metadata']['upload_date'].isoformat()
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(batch, f, ensure_ascii=False, indent=2)
            
            total_saved += len(batch)
            batch_num += 1
            logger.info(f"Saved batch {batch_num}: {len(batch)} docs → {filename}")
        
        logger.info(f"✓ Backup complete: {total_saved:,} documents saved to {output_dir}")
        
        # Create metadata file
        metadata = {
            "collection": collection_name,
            "backup_date": datetime.utcnow().isoformat(),
            "total_documents": total_saved,
            "num_batches": batch_num,
            "vector_dimension": Config.VECTOR_DIMENSION
        }
        
        metadata_file = output_dir / f"{collection_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata saved: {metadata_file}")
        
    except Exception as e:
        logger.error(f"Backup failed: {e}", exc_info=True)
        raise
    finally:
        await db.close()


async def backup_all_collections():
    """Backup all vector collections."""
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    backup_root = Config.BACKUP_DIR / timestamp
    
    logger.info("=" * 70)
    logger.info("VECTOR BACKUP UTILITY")
    logger.info("=" * 70)
    
    collections = [
        Config.SCRIPTURE_COLLECTION,
        Config.USER_CONTEXT_COLLECTION
    ]
    
    for coll_name in collections:
        coll_backup_dir = backup_root / coll_name
        await backup_collection(coll_name, coll_backup_dir)
    
    logger.info("=" * 70)
    logger.info(f"✓ ALL BACKUPS COMPLETE: {backup_root}")
    logger.info("=" * 70)


async def restore_collection(collection_name: str, backup_dir: Path):
    """
    Restore a collection from backup files.
    
    Args:
        collection_name: Name of collection to restore
        backup_dir: Directory containing backup JSON files
    """
    db = await get_database()
    
    try:
        # Find all batch files
        batch_files = sorted(backup_dir.glob(f"{collection_name}_*_batch*.json"))
        
        if not batch_files:
            logger.error(f"No backup files found in {backup_dir}")
            return
        
        logger.info(f"Found {len(batch_files)} backup files")
        
        total_restored = 0
        
        for batch_file in batch_files:
            # Load batch
            with open(batch_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            # Convert string IDs back if needed
            # (MongoDB will create new ObjectIds automatically)
            for doc in documents:
                if '_id' in doc:
                    del doc['_id']  # Let MongoDB create new IDs
            
            # Insert into MongoDB
            if documents:
                await db.insert_many(collection_name, documents)
                total_restored += len(documents)
                logger.info(f"Restored {len(documents)} docs from {batch_file.name}")
        
        logger.info(f"✓ Restore complete: {total_restored:,} documents")
        
    except Exception as e:
        logger.error(f"Restore failed: {e}", exc_info=True)
        raise
    finally:
        await db.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        # Restore mode
        if len(sys.argv) < 4:
            print("Usage: python backup_vectors.py restore <collection_name> <backup_dir>")
            sys.exit(1)
        
        coll = sys.argv[2]
        backup_path = Path(sys.argv[3])
        asyncio.run(restore_collection(coll, backup_path))
    else:
        # Backup mode
        asyncio.run(backup_all_collections())
