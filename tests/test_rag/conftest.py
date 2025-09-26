"""Test fixtures and configuration for RAG ingestion tests."""

import asyncio
import tempfile
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.schema import Document

from disney.rag.ingestion import IngestionPipeline, IngestionConfig
from disney.rag.document_processor import DocumentProcessorConfig


@pytest.fixture
def sample_reviews_data():
    """Sample Disney reviews data for testing."""
    return [
        {
            'id': 'review_1',
            'content': 'Space Mountain was absolutely amazing! The wait was worth it. 5 stars!',
            'rating': 5,
            'branch': 'Disneyland',
            'year': 2023,
            'source_file': 'test.csv',
            'row_index': 0
        },
        {
            'id': 'review_2',
            'content': 'The Haunted Mansion was disappointing. Too crowded and overpriced.',
            'rating': 2,
            'branch': 'Disney World',
            'year': 2023,
            'source_file': 'test.csv',
            'row_index': 1
        },
        {
            'id': 'review_3',
            'content': 'Pirates of the Caribbean was fun for the whole family. Great experience!',
            'rating': 4,
            'branch': 'Disneyland',
            'year': 2023,
            'source_file': 'test.csv',
            'row_index': 2
        }
    ]


@pytest.fixture
def sample_documents():
    """Sample LangChain documents for testing."""
    return [
        Document(
            page_content="Space Mountain was absolutely amazing! The wait was worth it. 5 stars!",
            metadata={
                'id': 'review_1',
                'rating': 5,
                'branch': 'Disneyland',
                'year': 2023,
                'source_file': 'test.csv',
                'row_index': 0
            }
        ),
        Document(
            page_content="The Haunted Mansion was disappointing. Too crowded and overpriced.",
            metadata={
                'id': 'review_2',
                'rating': 2,
                'branch': 'Disney World',
                'year': 2023,
                'source_file': 'test.csv',
                'row_index': 1
            }
        )
    ]


@pytest.fixture
def mock_vector_db():
    """Mock vector database for unit tests."""
    mock_db = AsyncMock()
    mock_db.add_documents.return_value = {
        'success': True,
        'indexed_count': 2,
        'processing_time_seconds': 1.5,
        'indexing_rate': 1.33
    }
    mock_db.get_collection_stats.return_value = {
        'collection_name': 'test_collection',
        'document_count': 2,
        'last_updated': '2023-01-01T00:00:00',
        'embedding_model': 'all-MiniLM-L6-v2'
    }
    mock_db.delete_collection.return_value = {'success': True}
    return mock_db


@pytest.fixture
def ingestion_config():
    """Default ingestion configuration for tests."""
    return IngestionConfig(
        batch_size=50,
        chunk_size=200,
        chunk_overlap=50,
        collection_name="test_collection",
        text_column="Review_Text"
    )


@pytest.fixture
def doc_processor_config():
    """Default document processor configuration for tests."""
    return DocumentProcessorConfig(
        chunk_size=200,
        chunk_overlap=50,
        min_chunk_size=10,
        max_chunk_size=1000
    )


@pytest.fixture
def temp_csv_file(sample_reviews_data):
    """Create a temporary CSV file for testing."""
    # Create DataFrame
    df = pd.DataFrame([
        {
            'Review_Text': review['content'],
            'Rating': review['rating'],
            'Branch': review['branch'],
            'Year_Month': review['year']
        }
        for review in sample_reviews_data
    ])
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    Path(temp_file).unlink(missing_ok=True)


@pytest.fixture
def temp_empty_csv_file():
    """Create an empty CSV file for testing."""
    df = pd.DataFrame(columns=['Review_Text', 'Rating', 'Branch', 'Year_Month'])
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    Path(temp_file).unlink(missing_ok=True)


@pytest.fixture
def temp_malformed_csv_file():
    """Create a malformed CSV file for testing."""
    # Create malformed CSV content
    malformed_content = """Review_Text,Rating,Branch,Year_Month
"Space Mountain was amazing!",5,Disneyland,2023
"Invalid rating",invalid,Disney World,2023
"Missing rating",,Disneyland,2023
"Valid review",4,Disney World,2023"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(malformed_content)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    Path(temp_file).unlink(missing_ok=True)


@pytest.fixture
def ingestion_pipeline(mock_vector_db, ingestion_config, doc_processor_config):
    """Create an ingestion pipeline for testing."""
    return IngestionPipeline(
        vector_db=mock_vector_db,
        config=ingestion_config,
        doc_config=doc_processor_config
    )


@pytest.fixture
def mock_csv_loader():
    """Mock CSV loader for testing."""
    mock_loader = MagicMock()
    mock_loader.load_reviews.return_value = [
        {
            'id': 'review_1',
            'content': 'Space Mountain was amazing!',
            'rating': 5,
            'branch': 'Disneyland',
            'year': 2023
        }
    ]
    return mock_loader


@pytest.fixture
def mock_document_processor():
    """Mock document processor for testing."""
    mock_processor = MagicMock()
    mock_processor.process_reviews_batch.return_value = [
        Document(
            page_content="Space Mountain was amazing!",
            metadata={'id': 'review_1', 'rating': 5}
        )
    ]
    mock_processor.get_processing_stats.return_value = {
        'total_documents': 1,
        'avg_chunk_size': 25,
        'total_chunks': 1
    }
    return mock_processor


@pytest.fixture
def mock_batch_indexer():
    """Mock batch indexer for testing."""
    mock_indexer = MagicMock()
    mock_indexer.index_documents_batch.return_value = {
        'success': True,
        'indexed_count': 1,
        'processing_time_seconds': 1.0,
        'indexing_rate': 1.0
    }
    return mock_indexer


@pytest.fixture
def expected_ingestion_result():
    """Expected result structure for successful ingestion."""
    return {
        'success': True,
        'file_path': 'test.csv',
        'collection_name': 'test_collection',
        'total_reviews': 3,
        'total_documents': 3,
        'indexed_documents': 3,
        'processing_stats': {
            'total_documents': 3,
            'avg_chunk_size': 50,
            'total_chunks': 3
        },
        'indexing_result': {
            'success': True,
            'indexed_count': 3,
            'processing_time_seconds': 1.5,
            'indexing_rate': 2.0
        },
        'validation_result': {
            'success': True,
            'expected_count': 3,
            'actual_count': 3,
            'validation_passed': True
        },
        'total_processing_time_seconds': 2.5,
        'processing_rate': 1.2,
        'timestamp': '2023-01-01T00:00:00'
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test data for different scenarios
@pytest.fixture
def large_reviews_data():
    """Large dataset for performance testing."""
    return [
        {
            'id': f'review_{i}',
            'content': f'This is review number {i}. It contains some sample content about Disney attractions.',
            'rating': (i % 5) + 1,
            'branch': 'Disneyland' if i % 2 == 0 else 'Disney World',
            'year': 2023,
            'source_file': 'large_test.csv',
            'row_index': i
        }
        for i in range(100)  # 100 reviews for performance testing
    ]


@pytest.fixture
def edge_case_reviews_data():
    """Edge case data for testing various scenarios."""
    return [
        {
            'id': 'review_empty_content',
            'content': '',  # Empty content
            'rating': 3,
            'branch': 'Disneyland',
            'year': 2023,
            'source_file': 'edge_case.csv',
            'row_index': 0
        },
        {
            'id': 'review_very_long',
            'content': 'This is a very long review. ' * 100,  # Very long content
            'rating': 4,
            'branch': 'Disney World',
            'year': 2023,
            'source_file': 'edge_case.csv',
            'row_index': 1
        },
        {
            'id': 'review_special_chars',
            'content': 'Special chars: !@#$%^&*()_+{}|:"<>?[]\\;\',./',
            'rating': 2,
            'branch': 'Disneyland',
            'year': 2023,
            'source_file': 'edge_case.csv',
            'row_index': 2
        }
    ]
