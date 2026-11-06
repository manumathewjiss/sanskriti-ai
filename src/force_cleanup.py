"""
Direct cleanup script - deletes all scripture_vectors without prompting.
"""

import asyncio
from database import get_database
from config import Config

async def force_cleanup():
    """Delete all scripture vectors."""
    print("Deleting old scripture_vectors...")
    
    db = await get_database()
    
    try:
        # Delete all documents
        deleted = await db.delete_many(Config.SCRIPTURE_COLLECTION, {})
        print(f"✓ Deleted {deleted:,} documents")
        print("✓ Ready for migration!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(force_cleanup())
