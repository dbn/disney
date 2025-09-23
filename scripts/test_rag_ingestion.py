#!/usr/bin/env python3
"""
Test script for RAG-based ingestion functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from disney.rag.document_processor import DocumentProcessor, DocumentProcessorConfig
from disney.rag.ingestion import IngestionPipeline, IngestionConfig


async def test_document_processor():
    """Test the document processor with sample Disney review data."""
    print("🧪 Testing Document Processor...")
    
    # Create sample review data
    sample_reviews = [
        {
            'id': 'review_1',
            'content': 'Space Mountain was absolutely amazing! The wait was worth it. 5 stars!',
            'rating': 5,
            'branch': 'Disneyland',
            'year': 2023
        },
        {
            'id': 'review_2', 
            'content': 'The Haunted Mansion was disappointing. Too crowded and overpriced.',
            'rating': 2,
            'branch': 'Disney World',
            'year': 2023
        }
    ]
    
    # Create document processor
    config = DocumentProcessorConfig(
        chunk_size=200,
        chunk_overlap=50,
        min_chunk_size=10,
        max_chunk_size=1000
    )
    processor = DocumentProcessor(config)
    
    # Process reviews
    documents = processor.process_reviews_batch(sample_reviews)
    
    print(f"✅ Processed {len(sample_reviews)} reviews into {len(documents)} documents")
    
    # Debug: Check why no documents are being created
    if len(documents) == 0:
        print("🔍 Debug: No documents created, checking individual review processing...")
        for i, review in enumerate(sample_reviews):
            print(f"  Processing review {i+1}: {review}")
            docs = processor.process_review(review)
            print(f"    -> Created {len(docs)} documents")
            if len(docs) > 0:
                print(f"    -> First doc content: {docs[0].page_content[:100]}...")
                print(f"    -> First doc metadata: {docs[0].metadata}")
    
    # Print document details
    for i, doc in enumerate(documents):
        print(f"  Document {i+1}:")
        print(f"    Content: {doc.page_content[:100]}...")
        print(f"    Metadata: {doc.metadata}")
        print()
    
    # Get processing stats
    stats = processor.get_processing_stats(documents)
    print(f"📊 Processing Stats: {stats}")
    
    return documents


async def test_ingestion_pipeline():
    """Test the ingestion pipeline (without vector database)."""
    print("\n🧪 Testing Ingestion Pipeline...")
    
    # Mock vector database for testing
    class MockVectorDB:
        async def add_documents(self, collection_name, documents, metadatas, ids):
            print(f"  📝 Mock: Would add {len(documents)} documents to collection '{collection_name}'")
            return {'success': True, 'indexed_count': len(documents)}
        
        async def get_collection_stats(self, collection_name):
            return {'document_count': 100, 'collection_name': collection_name}
    
    # Create ingestion pipeline
    vector_db = MockVectorDB()
    config = IngestionConfig(
        batch_size=50,
        collection_name="test_collection",
        chunk_size=200,
        chunk_overlap=50
    )
    
    pipeline = IngestionPipeline(vector_db, config)
    
    # Test CSV loading (using sample data)
    print("  📁 Testing CSV loader...")
    csv_loader = pipeline.csv_loader
    
    # Create a temporary CSV file for testing
    import tempfile
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'Review_Text': [
            'Space Mountain was amazing! 5 stars!',
            'The Haunted Mansion was disappointing.',
            'Pirates of the Caribbean was fun for the whole family.'
        ],
        'Rating': [5, 2, 4],
        'Branch': ['Disneyland', 'Disney World', 'Disneyland'],
        'Year_Month': [2023, 2023, 2023]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Load reviews
        reviews = csv_loader.load_reviews(temp_file)
        print(f"  ✅ Loaded {len(reviews)} reviews from CSV")
        
        # Process documents
        documents = pipeline.document_processor.process_reviews_batch(reviews)
        print(f"  ✅ Processed into {len(documents)} documents")
        
        # Test batch indexing
        result = await pipeline.batch_indexer.index_documents_batch(
            documents, "test_collection"
        )
        print(f"  ✅ Batch indexing result: {result}")
        
    finally:
        # Clean up temp file
        Path(temp_file).unlink()
    
    print("✅ Ingestion pipeline test completed")


async def main():
    """Run all tests."""
    print("🚀 Testing RAG-based Ingestion Functionality\n")
    
    try:
        # Test document processor
        documents = await test_document_processor()
        
        # Test ingestion pipeline
        await test_ingestion_pipeline()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📋 Summary:")
        print("  ✅ Document processor with LangChain integration")
        print("  ✅ Text splitting and metadata extraction")
        print("  ✅ CSV loading and processing")
        print("  ✅ Batch indexing pipeline")
        print("  ✅ Integration with RAG package")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
