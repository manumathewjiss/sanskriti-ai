"""
Script to clean up MongoDB collections to free space.
Use this to delete old data before migrating new scripture data.
"""

import asyncio
from database import get_database

async def cleanup_mongodb():
    """Delete all data from MongoDB to free up space."""
    print("=" * 70)
    print("MONGODB CLEANUP UTILITY")
    print("=" * 70)
    
    db = await get_database()
    
    try:
        # List all collections
        collections = await db.db.list_collection_names()
        print(f"\nFound {len(collections)} collections")
        
        if not collections:
            print("No collections to clean up!")
            return
        
        print("\nCollections found:")
        for coll in collections:
            count = await db.count_documents(coll, {})
            print(f"  - {coll}: {count:,} documents")
        
        # Confirm deletion
        print("\n⚠️  WARNING: This will DELETE ALL DATA from all collections!")
        response = input("Type 'DELETE' to confirm: ")
        
        if response != "DELETE":
            print("Cleanup cancelled.")
            return
        
        # Delete all collections
        total_deleted = 0
        for coll_name in collections:
            count = await db.count_documents(coll_name, {})
            if count > 0:
                deleted = await db.delete_many(coll_name, {})
                print(f"✓ Deleted {deleted:,} documents from {coll_name}")
                total_deleted += deleted
        
        print(f"\n✓ Total deleted: {total_deleted:,} documents")
        print(f"✓ MongoDB cleaned up successfully!")
        print("\nYou can now run the migration script.")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(cleanup_mongodb())
