"""
Analyze scripture text lengths to determine optimal embedding size.
"""

import asyncio
from database import get_database
from config import Config

async def analyze_text_lengths():
    """Analyze the text chunks in MongoDB."""
    print("=" * 70)
    print("SCRIPTURE TEXT LENGTH ANALYSIS")
    print("=" * 70)
    
    db = await get_database()
    
    try:
        collection = db.get_collection(Config.SCRIPTURE_COLLECTION)
        
        # Sample some documents to analyze
        sample_size = 100
        cursor = collection.find({}).limit(sample_size)
        docs = await cursor.to_list(length=sample_size)
        
        if not docs:
            print("No documents found!")
            return
        
        text_lengths = [len(doc.get('text', '')) for doc in docs]
        
        print(f"\n📊 Analysis of {len(docs)} sample documents:")
        print(f"   Min text length: {min(text_lengths)} chars")
        print(f"   Max text length: {max(text_lengths)} chars")
        print(f"   Average text length: {sum(text_lengths) / len(text_lengths):.1f} chars")
        print(f"   Median text length: {sorted(text_lengths)[len(text_lengths)//2]} chars")
        
        # Show text length distribution
        print("\n📈 Text Length Distribution:")
        ranges = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000)]
        for min_len, max_len in ranges:
            count = sum(1 for l in text_lengths if min_len <= l < max_len)
            pct = (count / len(text_lengths)) * 100
            bar = "█" * int(pct / 2)
            print(f"   {min_len:4d}-{max_len:4d} chars: {bar} {pct:5.1f}% ({count} docs)")
        
        # Calculate embedding overhead
        print("\n💾 Storage Analysis per Document:")
        
        # Check one doc's embedding size
        if docs and 'embedding' in docs[0]:
            embedding_dim = len(docs[0]['embedding'])
            embedding_bytes = embedding_dim * 4  # float32
            
            avg_text_bytes = sum(len(doc.get('text', '').encode('utf-8')) for doc in docs) / len(docs)
            avg_metadata_bytes = 100  # approximate
            avg_total_bytes = embedding_bytes + avg_text_bytes + avg_metadata_bytes
            
            print(f"   Embedding: {embedding_dim} dimensions × 4 bytes = {embedding_bytes:,} bytes ({embedding_bytes/1024:.1f} KB)")
            print(f"   Text: ~{avg_text_bytes:.0f} bytes ({avg_text_bytes/1024:.1f} KB)")
            print(f"   Metadata: ~{avg_metadata_bytes} bytes")
            print(f"   TOTAL: ~{avg_total_bytes:,.0f} bytes ({avg_total_bytes/1024:.1f} KB)")
            print(f"\n   Embedding is {(embedding_bytes/avg_total_bytes)*100:.1f}% of total document size!")
        
        # Show sample texts
        print("\n📝 Sample Texts (first 3):")
        for i, doc in enumerate(docs[:3], 1):
            text = doc.get('text', '')
            print(f"\n{i}. [{len(text)} chars] {text[:200]}...")
        
        print("\n" + "=" * 70)
        print("RECOMMENDATION:")
        print("=" * 70)
        print("For short texts (<500 chars), smaller embeddings work well:")
        print("  - 384 dims (all-MiniLM-L6-v2): ~70% space savings")
        print("  - 768 dims (multilingual-e5-base): ~25% space savings")
        print("  - 1024 dims (current): Best quality, highest cost")
        print("\nWith 549MB @ 1024 dims, switching to 384 dims would save ~380MB!")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(analyze_text_lengths())
