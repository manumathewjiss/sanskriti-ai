import sys
import os
from pathlib import Path

# Add src to python path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

import asyncio
from dotenv import load_dotenv

# Load env from .env file in parent directory
env_path = current_dir.parent / '.env'
load_dotenv(env_path)

from database import get_database
from mongodb_vector_store import search_scripture, get_collection_stats
from models import load_embedding_model

async def debug_rag():
    print("=== STARTING RAG DEBUG ===")
    
    # 1. Connect to DB
    print("\n1. Connecting to MongoDB...")
    try:
        db = await get_database()
        print("✓ Connected")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # 2. Check Collections
    print("\n2. Checking Collections...")
    try:
        collections = await db.db.list_collection_names()
        print(f"Found collections: {collections}")
        
        if 'scripture_vectors' not in collections:
            print("❌ 'scripture_vectors' collection MISSING!")
        else:
            stats = await get_collection_stats(db, 'scripture_vectors')
            print(f"Scripture Stats: {stats}")
            
            # Check for Search Indexes
            print("\n   Checking Search Indexes...")
            try:
                collection = db.get_collection('scripture_vectors')
                indexes = await collection.list_search_indexes().to_list(length=None)
                if not indexes:
                    print("   ❌ NO SEARCH INDEXES FOUND! Vector search will not work.")
                    print("      Please create 'scripture_vectors_vector_index' in Atlas UI.")
                else:
                    print(f"   ✓ Found {len(indexes)} search indexes:")
                    for idx in indexes:
                        print(f"      - Name: {idx.get('name')}")
                        print(f"        Type: {idx.get('type', 'search')}")
                        print(f"        Latest Definition: {idx.get('latestDefinition')}")
            except Exception as e:
                print(f"   ⚠️ Could not list search indexes (might lack permissions): {e}")
    except Exception as e:
        print(f"❌ Failed to list collections: {e}")
        return

    # 3. Test Embedding Model
    print("\n3. Testing Embedding Model (this may take a moment)...")
    try:
        model = load_embedding_model()
        query = "प्रविश्य तु महारण्यं दण्डकारण्यमात्मवान्"
        print(f"Embedding query: '{query}'")
        emb = model.embed_query(query)
        print(f"✓ Embedded successfully. Dimension: {len(emb)}")
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return

    # 4. vector Search Test
    print("\n4. Running Vector Search...")
    try:
        results = await search_scripture(db, emb, k=5)
        print(f"Search returned {len(results)} results")
        
        if not results:
            print("❌ NO RESULTS FOUND! Possible causes:")
            print("   - Vector index missing or not yet built")
            print("   - Dimensionality mismatch")
            print("   - Filter criteria mismatch")
            
        for i, (text, meta, score) in enumerate(results):
            print(f"\nResult {i+1} (Score: {score:.4f}):")
            print(f"Text: {text[:100]}...")
            print(f"Meta: {meta}")
    except Exception as e:
        print(f"❌ Search failed: {e}")
        
    print("\n=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(debug_rag())
