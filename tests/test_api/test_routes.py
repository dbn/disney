"""Tests for Customer Experience Assessment API routes."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from disney.api.models import QueryResponse, SourceDocument


def test_query_endpoint_success(client, mock_retrieval_manager_routes_patch):
    """Test successful query processing."""
    query_data = {
        "question": "What do customers say about Space Mountain?",
        "context_limit": 5,
        "temperature": 0.7
    }
    
    response = client.post("/api/v1/query", json=query_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
    assert "processing_time_ms" in data
    assert len(data["sources"]) == 1
    assert data["sources"][0]["review_id"] == "test_1"
    assert data["answer"] == "Test answer from mock"
    assert data["confidence"] == 0.87
    assert data["processing_time_ms"] == 1000.0


def test_query_endpoint_no_context(client, mock_retrieval_manager_routes_patch):
    """Test query when no context is found."""
    # Mock empty response
    empty_response = QueryResponse(
        answer="I couldn't find any relevant information for your question.",
        sources=[],
        confidence=0.5,
        processing_time_ms=500.0
    )
    mock_retrieval_manager_routes_patch.query_with_metadata.return_value = empty_response
    
    query_data = {
        "question": "What do customers say about Space Mountain?",
        "context_limit": 5,
        "temperature": 0.7
    }
    
    response = client.post("/api/v1/query", json=query_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "I couldn't find any relevant information" in data["answer"]
    assert data["confidence"] == 0.5
    assert len(data["sources"]) == 0


def test_query_endpoint_invalid_data(client, mock_retrieval_manager_routes_patch):
    """Test query endpoint with invalid data."""
    invalid_data = {
        "question": "",  # Empty question should fail validation
        "context_limit": 0,  # Invalid limit
        "temperature": 3.0  # Invalid temperature
    }
    
    response = client.post("/api/v1/query", json=invalid_data)
    assert response.status_code == 422  # Validation error


def test_health_endpoint_success(client, mock_retrieval_manager_routes_patch):
    """Test health endpoint with successful dependency checks."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "dependencies" in data
    assert data["dependencies"]["chromadb"] == "healthy"
    assert data["dependencies"]["llm_service"] == "healthy"


def test_status_endpoint(client, mock_retrieval_manager_routes_patch):
    """Test status endpoint."""
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    
    data = response.json()
    assert data["service"] == "customer-experience-api"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"
    assert "components" in data


def test_query_endpoint_missing_question(client, mock_retrieval_manager_routes_patch):
    """Test query endpoint with missing question field."""
    invalid_data = {
        "context_limit": 5,
        "temperature": 0.7
    }
    
    response = client.post("/api/v1/query", json=invalid_data)
    assert response.status_code == 422  # Validation error


def test_query_endpoint_negative_context_limit(client, mock_retrieval_manager_routes_patch):
    """Test query endpoint with negative context limit."""
    query_data = {
        "question": "What do customers say about Space Mountain?",
        "context_limit": -1,
        "temperature": 0.7
    }
    
    response = client.post("/api/v1/query", json=query_data)
    assert response.status_code == 422  # Validation error


def test_query_endpoint_temperature_out_of_range(client, mock_retrieval_manager_routes_patch):
    """Test query endpoint with temperature out of range."""
    query_data = {
        "question": "What do customers say about Space Mountain?",
        "context_limit": 5,
        "temperature": 2.5  # Should be between 0 and 2
    }
    
    response = client.post("/api/v1/query", json=query_data)
    assert response.status_code == 422  # Validation error


def test_health_endpoint_chromadb_unavailable(client, mock_retrieval_manager_routes_patch):
    """Test health endpoint when ChromaDB is unavailable."""
    # Mock vector store manager to raise an exception
    mock_retrieval_manager_routes_patch.get_collection_stats.side_effect = Exception("ChromaDB connection failed")
    
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "degraded"
    assert data["dependencies"]["chromadb"] == "unhealthy"


def test_query_endpoint_processing_error(client, mock_retrieval_manager_routes_patch):
    """Test query endpoint when processing fails."""
    # Mock vector manager to raise an exception
    mock_retrieval_manager_routes_patch.query_with_metadata.side_effect = Exception("Chain processing failed")
    
    query_data = {
        "question": "What do customers say about Space Mountain?",
        "context_limit": 5,
        "temperature": 0.7
    }
    
    response = client.post("/api/v1/query", json=query_data)
    assert response.status_code == 500
    
    data = response.json()
    assert "detail" in data
    assert "Chain processing failed" in data["detail"]


def test_cache_status_endpoint(client, mock_retrieval_manager_routes_patch):
    """Test cache status endpoint."""
    # Mock the stats function to return cached status
    with patch('disney.api.routes.get_retrieval_manager_stats', return_value={
        "is_cached": True,
        "instance_type": "RetrievalManager"
    }):
        response = client.get("/api/v1/cache-status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["cache_status"] == "active"
        assert data["retrieval_manager_cached"] is True
        assert data["instance_type"] == "RetrievalManager"
