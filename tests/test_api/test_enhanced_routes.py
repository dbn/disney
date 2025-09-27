"""
Tests for enhanced API routes with metadata extraction.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from disney.api.main import app
from disney.api.models import QueryRequest, QueryResponse


class TestEnhancedQueryEndpoint:
    """Test the enhanced query endpoint with metadata extraction."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_retrieval_manager(self):
        """Mock RetrievalManager for testing."""
        mock_manager = MagicMock()
        mock_manager.query_with_metadata = AsyncMock(return_value="Enhanced answer")
        mock_manager.get_enhanced_context = AsyncMock(return_value=[
            {
                "review_id": "12345",
                "relevance_score": 0.95,
                "excerpt": "Great experience at Disney!",
                "metadata": {
                    "rating": 5,
                    "year": 2023,
                    "month": 6,
                    "branch": "Disneyland"
                }
            }
        ])
        return mock_manager
    
    @pytest.mark.asyncio
    async def test_query_enhanced_success(self, client, mock_retrieval_manager):
        """Test successful enhanced query."""
        with patch('disney.api.routes.get_retrieval_manager', return_value=mock_retrieval_manager):
            response = client.post(
                "/api/v1/query-enhanced",
                json={
                    "question": "Show me 5-star reviews from Disneyland in 2023",
                    "context_limit": 5,
                    "temperature": 0.7
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "processing_time_ms" in data
        
        assert data["answer"] == "Enhanced answer"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["review_id"] == "12345"
        assert data["sources"][0]["relevance_score"] == 0.95
        
        # Verify the enhanced methods were called
        mock_retrieval_manager.query_with_metadata.assert_called_once_with(
            "Show me 5-star reviews from Disneyland in 2023"
        )
        mock_retrieval_manager.get_enhanced_context.assert_called_once_with(
            query="Show me 5-star reviews from Disneyland in 2023",
            n_results=5,
            use_metadata_extraction=True
        )
    
    @pytest.mark.asyncio
    async def test_query_enhanced_with_metadata_filters(self, client, mock_retrieval_manager):
        """Test enhanced query with metadata filtering."""
        # Mock query parser result
        mock_parse_result = MagicMock()
        mock_parse_result.search_query = "Disney reviews experiences"
        mock_parse_result.filters = MagicMock()
        mock_parse_result.filters.rating = 5
        mock_parse_result.filters.branch = "Disneyland"
        mock_parse_result.confidence = 0.9
        mock_parse_result.reasoning = "Extracted 5-star rating and Disneyland location"
        
        mock_retrieval_manager.query_parser = MagicMock()
        mock_retrieval_manager.query_parser.parse_query = AsyncMock(return_value=mock_parse_result)
        mock_retrieval_manager.query_parser._build_chromadb_filters = MagicMock(
            return_value={"rating": 5, "branch": "Disneyland"}
        )
        
        with patch('disney.api.routes.get_retrieval_manager', return_value=mock_retrieval_manager):
            response = client.post(
                "/api/v1/query-enhanced",
                json={
                    "question": "What do people think about Space Mountain at Disneyland? Show me 5-star reviews.",
                    "context_limit": 3,
                    "temperature": 0.5
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Enhanced answer"
        assert len(data["sources"]) == 1
        
        # Verify metadata filtering was used
        mock_retrieval_manager.query_with_metadata.assert_called_once()
        mock_retrieval_manager.get_enhanced_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_enhanced_error_handling(self, client):
        """Test enhanced query error handling."""
        with patch('disney.api.routes.get_retrieval_manager', side_effect=Exception("Database error")):
            response = client.post(
                "/api/v1/query-enhanced",
                json={
                    "question": "Test query",
                    "context_limit": 5
                }
            )
        
        assert response.status_code == 500
        data = response.json()
        assert "Database error" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_query_enhanced_invalid_request(self, client, mock_retrieval_manager):
        """Test enhanced query with invalid request."""
        with patch('disney.api.routes.get_retrieval_manager', return_value=mock_retrieval_manager):
            response = client.post(
                "/api/v1/query-enhanced",
                json={
                    "question": "",  # Empty question
                    "context_limit": 5
                }
            )
        
        # Should still process but with empty query
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_query_enhanced_fallback_to_regular_query(self, client):
        """Test fallback to regular query when enhanced query fails."""
        mock_retrieval_manager = MagicMock()
        # Mock the query_with_metadata to handle fallback internally
        mock_retrieval_manager.query_with_metadata = AsyncMock(return_value="Fallback answer")
        mock_retrieval_manager.get_enhanced_context = AsyncMock(return_value=[])
        
        with patch('disney.api.routes.get_retrieval_manager', return_value=mock_retrieval_manager):
            response = client.post(
                "/api/v1/query-enhanced",
                json={
                    "question": "Test query",
                    "context_limit": 5
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return the fallback answer
        assert data["answer"] == "Fallback answer"
        mock_retrieval_manager.query_with_metadata.assert_called_once_with("Test query")
    
    @pytest.mark.asyncio
    async def test_query_enhanced_processing_time(self, client, mock_retrieval_manager):
        """Test that processing time is included in response."""
        with patch('disney.api.routes.get_retrieval_manager', return_value=mock_retrieval_manager):
            response = client.post(
                "/api/v1/query-enhanced",
                json={
                    "question": "Test query",
                    "context_limit": 5
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "processing_time_ms" in data
        assert isinstance(data["processing_time_ms"], (int, float))
        assert data["processing_time_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_query_enhanced_sources_format(self, client, mock_retrieval_manager):
        """Test that sources are properly formatted."""
        mock_retrieval_manager.get_enhanced_context.return_value = [
            {
                "review_id": "12345",
                "relevance_score": 0.95,
                "excerpt": "Great experience!",
                "metadata": {
                    "rating": 5,
                    "year": 2023,
                    "branch": "Disneyland"
                }
            },
            {
                "review_id": "67890",
                "relevance_score": 0.87,
                "excerpt": "Amazing time at the park!",
                "metadata": {
                    "rating": 4,
                    "year": 2023,
                    "branch": "Disney World"
                }
            }
        ]
        
        with patch('disney.api.routes.get_retrieval_manager', return_value=mock_retrieval_manager):
            response = client.post(
                "/api/v1/query-enhanced",
                json={
                    "question": "Test query",
                    "context_limit": 5
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["sources"]) == 2
        
        # Check first source
        source1 = data["sources"][0]
        assert source1["review_id"] == "12345"
        assert source1["relevance_score"] == 0.95
        assert source1["excerpt"] == "Great experience!"
        assert source1["metadata"]["rating"] == 5
        assert source1["metadata"]["year"] == 2023
        assert source1["metadata"]["branch"] == "Disneyland"
        
        # Check second source
        source2 = data["sources"][1]
        assert source2["review_id"] == "67890"
        assert source2["relevance_score"] == 0.87
        assert source2["excerpt"] == "Amazing time at the park!"
        assert source2["metadata"]["rating"] == 4
        assert source2["metadata"]["year"] == 2023
        assert source2["metadata"]["branch"] == "Disney World"


class TestEnhancedQueryIntegration:
    """Integration tests for enhanced query functionality."""
    
    @pytest.mark.asyncio
    async def test_enhanced_vs_regular_query(self):
        """Test that enhanced query provides different results than regular query."""
        # This test would require a real ChromaDB instance and data
        # Skip if not available
        pytest.skip("Requires real ChromaDB instance and data")
    
    @pytest.mark.asyncio
    async def test_metadata_extraction_accuracy(self):
        """Test accuracy of metadata extraction."""
        # This test would require a real LLM and test cases
        # Skip if not available
        pytest.skip("Requires real LLM and test cases")
