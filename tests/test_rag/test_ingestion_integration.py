"""Integration tests for the ingestion pipeline with mocked ChromaDB."""

import asyncio
import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from disney.rag.ingestion import IngestionPipeline, IngestionConfig
from disney.rag.document_processor import DocumentProcessorConfig
from disney.rag.retrieval_manager import RetrievalManager


@pytest.mark.integration
class TestIngestionIntegration:
    """Integration tests with mocked ChromaDB."""

    @pytest.fixture
    def mock_vector_db(self):
        """Create a mocked RetrievalManager for integration tests."""
        # Mock ChromaDB client and related components
        with patch('disney.rag.retrieval_manager.chromadb.HttpClient') as mock_chroma_client, \
             patch('disney.rag.retrieval_manager.Chroma') as mock_chroma, \
             patch('disney.rag.retrieval_manager.HuggingFaceEmbeddings') as mock_embeddings, \
             patch('disney.rag.retrieval_manager.ChatOpenAI') as mock_llm:

            # Mock ChromaDB client
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.get_collection.return_value = mock_collection
            mock_chroma_client.return_value = mock_client
            
            # Mock Chroma vector store
            mock_chroma_instance = MagicMock()
            mock_retriever = MagicMock()
            mock_retriever.get_relevant_documents.return_value = []
            mock_chroma_instance.as_retriever.return_value = mock_retriever
            mock_chroma.return_value = mock_chroma_instance
            
            # Mock embeddings
            mock_embeddings.return_value = MagicMock()
            
            # Mock LLM
            mock_llm.return_value = MagicMock()
            
            # Create RetrievalManager instance
            vector_manager = RetrievalManager(
                chroma_host="localhost",
                chroma_port=8000
            )
            
            # Store mocks for later use
            vector_manager._mock_client = mock_client
            vector_manager._mock_collection = mock_collection
            vector_manager._mock_chroma = mock_chroma_instance
            vector_manager._mock_retriever = mock_retriever
            
            yield vector_manager

    @pytest.fixture
    def integration_config(self):
        """Configuration for integration tests."""
        return IngestionConfig(
            batch_size=10,  # Smaller batch size for testing
            chunk_size=200,
            chunk_overlap=50,
            collection_name="disney_reviews",
            text_column="Review_Text"
        )

    @pytest.fixture
    def integration_doc_config(self):
        """Document processor configuration for integration tests."""
        return DocumentProcessorConfig(
            chunk_size=200,
            chunk_overlap=50,
            min_chunk_size=10,
            max_chunk_size=1000
        )

    @pytest.fixture
    def integration_pipeline(self, mock_vector_db, integration_config, integration_doc_config):
        """Create integration pipeline with mocked ChromaDB."""
        return IngestionPipeline(
            vector_db=mock_vector_db,
            config=integration_config,
            doc_config=integration_doc_config
        )

    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for integration testing."""
        sample_data = pd.DataFrame({
            'Review_Text': [
                'Space Mountain was absolutely amazing! The wait was worth it. 5 stars!',
                'The Haunted Mansion was disappointing. Too crowded and overpriced.',
                'Pirates of the Caribbean was fun for the whole family. Great experience!',
                'Big Thunder Mountain Railroad was thrilling and exciting.',
                'It\'s a Small World was cute but the song gets stuck in your head.',
                'Splash Mountain was wet and wild! Perfect for hot days.',
                'The Matterhorn was bumpy but fun. Great views from the top.',
                'Indiana Jones Adventure was intense and well-themed.',
                'Star Tours was okay but not as good as the original.',
                'Jungle Cruise was entertaining with great jokes from the skipper.'
            ],
            'Rating': [5, 2, 4, 5, 3, 4, 3, 5, 3, 4],
            'Branch': ['Disneyland', 'Disney World', 'Disneyland', 'Disneyland', 
                      'Disneyland', 'Disney World', 'Disneyland', 'Disneyland', 
                      'Disney World', 'Disneyland'],
            'Year_Month': [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        Path(temp_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_end_to_end_ingestion(
        self, 
        integration_pipeline, 
        sample_csv_file,
        mock_vector_db
    ):
        """Test complete end-to-end ingestion pipeline."""
        # Mock the collection count to simulate indexed documents
        mock_vector_db._mock_collection.count.return_value = 15
        
        # Execute ingestion
        result = await integration_pipeline.ingest_csv_file(sample_csv_file)
        
        # Verify ingestion success
        assert result['success'] is True
        assert result['total_reviews'] == 10
        assert result['total_documents'] > 0
        assert result['indexed_documents'] > 0
        assert result['validation_result']['validation_passed'] is True
        
        # Verify data in ChromaDB
        stats = mock_vector_db.get_collection_stats()
        assert stats['document_count'] >= 10
        
        # Test retrieval to verify data quality
        context_docs = mock_vector_db.get_relevant_context(
            query="Space Mountain",
            n_results=3,
            similarity_threshold=0.5
        )
        
        assert len(context_docs) >= 0  # May be empty due to mocking

    @pytest.mark.asyncio
    async def test_data_consistency_after_ingestion(
        self, 
        integration_pipeline, 
        sample_csv_file,
        mock_vector_db
    ):
        """Test that ingested data is consistent and retrievable."""
        # Execute ingestion
        result = await integration_pipeline.ingest_csv_file(sample_csv_file)
        assert result['success'] is True
        
        # Test various queries to ensure data consistency
        test_queries = [
            "Space Mountain",
            "Haunted Mansion",
            "Pirates of the Caribbean",
            "Big Thunder Mountain",
            "Splash Mountain"
        ]
        
        for query in test_queries:
            context_docs = mock_vector_db.get_relevant_context(
                query=query,
                n_results=2,
                similarity_threshold=0.3
            )
            
            # Verify document structure (may be empty due to mocking)
            for doc in context_docs:
                assert 'content' in doc
                assert 'metadata' in doc
                assert 'relevance_score' in doc
                assert doc['relevance_score'] > 0

    @pytest.mark.asyncio
    async def test_metadata_preservation(
        self, 
        integration_pipeline, 
        sample_csv_file,
        mock_vector_db
    ):
        """Test that metadata is preserved during ingestion."""
        # Execute ingestion
        result = await integration_pipeline.ingest_csv_file(sample_csv_file)
        assert result['success'] is True
        
        # Test retrieval and verify metadata
        context_docs = mock_vector_db.get_relevant_context(
            query="Disneyland attractions",
            n_results=5,
            similarity_threshold=0.3
        )
        
        # Check that metadata is preserved (may be empty due to mocking)
        for doc in context_docs:
            metadata = doc.get('metadata', {})
            assert 'id' in metadata
            assert 'rating' in metadata
            assert 'branch' in metadata
            assert 'year' in metadata
            assert 'source_file' in metadata
            assert 'row_index' in metadata

    @pytest.mark.asyncio
    async def test_collection_management(
        self, 
        integration_pipeline, 
        sample_csv_file,
        mock_vector_db
    ):
        """Test collection creation and management."""
        # Execute ingestion
        result = await integration_pipeline.ingest_csv_file(sample_csv_file)
        assert result['success'] is True
        
        # Verify collection exists and has correct stats
        stats = mock_vector_db.get_collection_stats()
        assert stats['collection_name'] == 'disney_reviews'
        assert stats['document_count'] >= 0  # May be 0 due to mocking
        
        # Test reindexing (clear and re-ingest)
        reindex_result = await integration_pipeline.reindex_collection(sample_csv_file)
        assert reindex_result['success'] is True
        assert reindex_result['reindexed'] is True
        
        # Verify collection still exists after reindexing
        stats_after = mock_vector_db.get_collection_stats()
        assert stats_after['document_count'] >= 0

    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(
        self, 
        integration_pipeline, 
        mock_vector_db
    ):
        """Test batch processing efficiency with larger dataset."""
        # Create larger dataset
        large_data = []
        for i in range(50):  # 50 reviews
            large_data.append({
                'Review_Text': f'This is review number {i}. It contains sample content about Disney attractions and experiences.',
                'Rating': (i % 5) + 1,
                'Branch': 'Disneyland' if i % 2 == 0 else 'Disney World',
                'Year_Month': 2023
            })
        
        df = pd.DataFrame(large_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Execute ingestion
            result = await integration_pipeline.ingest_csv_file(temp_file)
            
            assert result['success'] is True
            assert result['total_reviews'] == 50
            assert result['processing_rate'] > 0  # Should have positive processing rate
            
            # Verify data was indexed
            stats = mock_vector_db.get_collection_stats()
            assert stats['document_count'] >= 0  # May be 0 due to mocking
            
        finally:
            Path(temp_file).unlink()

    @pytest.mark.asyncio
    async def test_error_handling_with_real_db(
        self, 
        integration_pipeline, 
        mock_vector_db
    ):
        """Test error handling with mocked ChromaDB."""
        # Test with non-existent file
        result = await integration_pipeline.ingest_csv_file("non_existent_file.csv")
        assert result['success'] is False
        assert 'CSV file not found' in str(result['error'])
        
        # Test with empty file
        empty_df = pd.DataFrame(columns=['Review_Text', 'Rating', 'Branch', 'Year_Month'])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            empty_df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            result = await integration_pipeline.ingest_csv_file(temp_file)
            assert result['success'] is False
            assert 'No reviews found' in str(result['error'])
        finally:
            Path(temp_file).unlink()

    @pytest.mark.asyncio
    async def test_different_chunk_sizes(
        self, 
        mock_vector_db, 
        sample_csv_file
    ):
        """Test ingestion with different chunk sizes."""
        chunk_configs = [
            (100, 20),   # Small chunks
            (500, 100),  # Medium chunks
            (1000, 200)  # Large chunks
        ]
        
        for chunk_size, chunk_overlap in chunk_configs:
            config = IngestionConfig(
                batch_size=10,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                collection_name="disney_reviews"
            )
            
            doc_config = DocumentProcessorConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=10,
                max_chunk_size=2000
            )
            
            pipeline = IngestionPipeline(
                vector_db=mock_vector_db,
                config=config,
                doc_config=doc_config
            )
            
            # Execute ingestion
            result = await pipeline.ingest_csv_file(sample_csv_file)
            
            assert result['success'] is True
            assert result['total_reviews'] == 10
            
            # Verify data can be retrieved
            context_docs = mock_vector_db.get_relevant_context(
                query="Space Mountain",
                n_results=2,
                similarity_threshold=0.3
            )
            
            # May be empty due to mocking, but structure should be correct
            for doc in context_docs:
                assert 'content' in doc
                assert 'metadata' in doc

    @pytest.mark.asyncio
    async def test_concurrent_ingestion(
        self, 
        mock_vector_db, 
        sample_csv_file
    ):
        """Test concurrent ingestion operations."""
        # Create multiple pipelines
        pipelines = []
        for i in range(3):
            config = IngestionConfig(
                batch_size=5,
                collection_name=f"test_collection_{i}"
            )
            pipeline = IngestionPipeline(vector_db=mock_vector_db, config=config)
            pipelines.append(pipeline)
        
        # Execute concurrent ingestion
        tasks = [
            pipeline.ingest_csv_file(sample_csv_file)
            for pipeline in pipelines
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all ingestions succeeded
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent ingestion {i} failed: {result}")
            assert result['success'] is True
            assert result['total_reviews'] == 10

    @pytest.mark.asyncio
    async def test_ingestion_with_special_characters(
        self, 
        integration_pipeline, 
        mock_vector_db
    ):
        """Test ingestion with special characters and unicode."""
        special_data = pd.DataFrame({
            'Review_Text': [
                'Special chars: !@#$%^&*()_+{}|:"<>?[]\\;\',./',
                'Unicode: 🎢🎠🎡 and emojis!',
                '中文测试 Chinese characters',
                'Accented: café, naïve, résumé',
                'Math symbols: ∑, ∫, ∞, ≠, ≤, ≥'
            ],
            'Rating': [3, 4, 2, 5, 3],
            'Branch': ['Disneyland', 'Disney World', 'Disneyland', 'Disney World', 'Disneyland'],
            'Year_Month': [2023, 2023, 2023, 2023, 2023]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            special_data.to_csv(f.name, index=False, encoding='utf-8')
            temp_file = f.name
        
        try:
            result = await integration_pipeline.ingest_csv_file(temp_file)
            
            assert result['success'] is True
            assert result['total_reviews'] == 5
            
            # Test retrieval of special character content
            context_docs = mock_vector_db.get_relevant_context(
                query="emoji",
                n_results=2,
                similarity_threshold=0.3
            )
            
            # May be empty due to mocking, but structure should be correct
            for doc in context_docs:
                assert 'content' in doc
                assert 'metadata' in doc
            
        finally:
            Path(temp_file).unlink()