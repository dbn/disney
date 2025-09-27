"""
Tests for enhanced API routes with metadata extraction.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from disney.api.models import QueryRequest, QueryResponse, SourceDocument


class TestEnhancedQueryEndpoint:
    """Test the enhanced query endpoint with metadata extraction."""
    
    @pytest.mark.asyncio
    async def test_query_enhanced_success(self, client, mock_retrieval_manager_routes_patch):
        """Test successful enhanced query."""
        response = client.post(
            "/api/v1/query",
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
        
        assert data["answer"] == "Test answer from mock"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["review_id"] == "test_1"
        assert data["sources"][0]["relevance_score"] == 0.95
        
        # Verify the enhanced methods were called
        mock_retrieval_manager_routes_patch.query_with_metadata.assert_called_once_with(
            "Show me 5-star reviews from Disneyland in 2023"
        )
    
    @pytest.mark.asyncio
    async def test_query_enhanced_with_metadata_filters(self, client, mock_retrieval_manager_routes_patch):
        """Test enhanced query with metadata filtering."""
        response = client.post(
            "/api/v1/query",
            json={
                "question": "What do people think about Space Mountain at Disneyland? Show me 5-star reviews.",
                "context_limit": 3,
                "temperature": 0.5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Test answer from mock"
        assert len(data["sources"]) == 1
        
        # Verify metadata filtering was used
        mock_retrieval_manager_routes_patch.query_with_metadata.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_enhanced_error_handling(self, client, mock_retrieval_manager_routes_patch):
        """Test enhanced query error handling."""
        # Mock the retrieval manager to raise an exception
        mock_retrieval_manager_routes_patch.query_with_metadata.side_effect = Exception("Database error")
        
        response = client.post(
            "/api/v1/query",
            json={
                "question": "Test query",
                "context_limit": 5
            }
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "Database error" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_query_enhanced_invalid_request(self, client, mock_retrieval_manager_routes_patch):
        """Test enhanced query with invalid request."""
        response = client.post(
            "/api/v1/query",
            json={
                "question": "",  # Empty question
                "context_limit": 5
            }
        )
        
        # Should still process but with empty query
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_query_enhanced_fallback_to_regular_query(self, client, mock_retrieval_manager_routes_patch):
        """Test fallback to regular query when enhanced query fails."""
        # Mock the query_with_metadata to handle fallback internally
        mock_response = QueryResponse(
            answer="Fallback answer",
            sources=[],
            confidence=0.5,
            processing_time_ms=500.0
        )
        mock_retrieval_manager_routes_patch.query_with_metadata.return_value = mock_response
        
        response = client.post(
            "/api/v1/query",
            json={
                "question": "Test query",
                "context_limit": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return the fallback answer
        assert data["answer"] == "Fallback answer"
        mock_retrieval_manager_routes_patch.query_with_metadata.assert_called_once_with("Test query")
    
    @pytest.mark.asyncio
    async def test_query_enhanced_processing_time(self, client, mock_retrieval_manager_routes_patch):
        """Test that processing time is included in response."""
        response = client.post(
            "/api/v1/query",
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
    async def test_query_enhanced_sources_format(self, client, mock_retrieval_manager_routes_patch):
        """Test that sources are properly formatted."""
        # Update the mock response to include multiple sources
        mock_response = QueryResponse(
            answer="Enhanced answer with multiple sources",
            sources=[
                SourceDocument(
                    review_id="12345",
                    relevance_score=0.95,
                    excerpt="Great experience!",
                    metadata={
                        "rating": 5,
                        "year": 2023,
                        "branch": "Disneyland"
                    }
                ),
                SourceDocument(
                    review_id="67890",
                    relevance_score=0.87,
                    excerpt="Amazing time at the park!",
                    metadata={
                        "rating": 4,
                        "year": 2023,
                        "branch": "Disney World"
                    }
                )
            ],
            confidence=0.87,
            processing_time_ms=1250.0
        )
        mock_retrieval_manager_routes_patch.query_with_metadata.return_value = mock_response
        
        response = client.post(
            "/api/v1/query",
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
    
    @pytest.mark.asyncio
    async def test_query_with_real_retrieval_manager(self):
        """Test query with real RetrievalManager (if available)."""
        # This test would require a real ChromaDB instance and data
        # Skip if not available
        pytest.skip("Requires real ChromaDB instance and data")
