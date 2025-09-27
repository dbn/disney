#!/usr/bin/env python3
"""
Test script for RAG-based ingestion functionality.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from disney.pipeline.ingest import DataIngester


async def test_ingestion_pipeline():
    """Test the complete ingestion pipeline."""
    print("🧪 Testing Ingestion Pipeline...")
    
    # Mock vector database
    class MockVectorDB:
        async def add_documents(self, documents):
            print(f"  📝 Mock: Would add {len(documents)} documents")
            return True
        
        async def get_collection_stats(self):
            return {'document_count': 100, 'collection_name': 'test_collection'}
    
    # Create data ingester
    vector_db = MockVectorDB()
    ingester = DataIngester(chroma_host="localhost", chroma_port=8000)
    ingester.retrieval_manager = vector_db
    
    # Create temporary CSV file for testing
    import tempfile
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'Review_Text': [
            'Space Mountain was absolutely amazing! The wait was worth it.',
            'The Haunted Mansion was disappointing. Too crowded.',
            'Pirates of the Caribbean was fun but the line was long.'
        ],
        'Rating': [5, 2, 4],
        'Year_Month': ['2023-6', '2023-7', '2023-8'],
        'Branch': ['Disneyland', 'Disney World', 'Disneyland'],
        'Review_ID': ['1', '2', '3'],
        'Reviewer_Location': ['USA', 'Canada', 'UK']
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Test the data ingester with the temp file
        with patch('disney.pipeline.ingest.settings') as mock_settings:
            mock_settings.data_path = temp_file
            mock_settings.chroma_host = "localhost"
            mock_settings.chroma_port = 8000
            
            # Run the ingestion pipeline
            result = await ingester.run_ingestion_pipeline()
            print(f"  ✅ Ingestion result: {result}")
        
    finally:
        # Clean up temp file
        Path(temp_file).unlink()
    
    return result


async def test_metadata_extraction():
    """Test metadata extraction functionality."""
    print("🧪 Testing Metadata Extraction...")
    
    from disney.pipeline.metadata_extractor import MetadataExtractor
    
    extractor = MetadataExtractor()
    
    # Test sample data
    sample_data = {
        'Rating': 4,
        'Year_Month': '2019-4',
        'Branch': 'Disneyland_HongKong',
        'Review_ID': '670772142',
        'Reviewer_Location': 'Australia'
    }
    
    metadata = extractor.extract_all_metadata(sample_data)
    print(f"  ✅ Extracted metadata: {metadata}")
    
    # Test month extraction specifically
    print(f"  📅 Month extraction test:")
    print(f"    '2019-4' -> month: {extractor.extract_month('2019-4')}")
    print(f"    '2020-12' -> month: {extractor.extract_month('2020-12')}")
    print(f"    '2021-6' -> month: {extractor.extract_month('2021-6')}")
    
    return metadata


async def main():
    """Main test function."""
    print("🚀 Starting RAG Ingestion Tests...")
    print()
    
    try:
        # Test metadata extraction
        metadata = await test_metadata_extraction()
        print()
        
        # Test ingestion pipeline
        result = await test_ingestion_pipeline()
        print()
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())