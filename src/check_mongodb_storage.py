"""
Quick script to check what's currently in MongoDB and how much space it's using.
"""

import asyncio
from database import get_database
from config import Config

async def check_storage():
    """Check MongoDB collections and sizes."""
    print("=" * 70)
    print("MONGODB STORAGE ANALYSIS")
    print("=" * 70)
    
    db = await get_database()
    
    try:
        # List all collections
        collections = await db.db.list_collection_names()
        print(f"\nFound {len(collections)} collections:")
        
        total_size = 0
        for coll_name in collections:
            collection = db.get_collection(coll_name)
            
            # Get collection stats
            stats = await db.db.command("collStats", coll_name)
            
            size_mb = stats.get('size', 0) / (1024 * 1024)
            doc_count = stats.get('count', 0)
            
            print(f"\n📁 {coll_name}:")
            print(f"   Documents: {doc_count:,}")
            print(f"   Size: {size_mb:.2f} MB")
            
            total_size += size_mb
        
        print(f"\n" + "=" * 70)
        print(f"TOTAL SIZE: {total_size:.2f} MB")
        print(f"QUOTA: 512 MB (Free Tier)")
        print(f"AVAILABLE: {512 - total_size:.2f} MB")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error checking storage: {e}")
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(check_storage())
