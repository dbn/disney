"""Tests for Customer Experience Assessment API routes."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock

from src.disney.api.main import app


@pytest.fixture
def client():
    """Test client for the API."""
    return TestClient(app)


@pytest.fixture
def mock_rag_components():
    """Mock RAG components for testing."""
    with patch('src.disney.api.routes.get_retrieval_manager') as mock_vector_manager:
        
        # Mock vector store manager
        mock_vector_instance = MagicMock()
        mock_vector_instance.query = AsyncMock(return_value="Based on customer reviews, Space Mountain is highly rated with customers saying the wait is worth it.")
        mock_vector_instance.get_relevant_context.return_value = [
            {
                "id": "test_review_1",
                "content": "Space Mountain was amazing! The wait was worth it.",
                "metadata": {"rating": 5, "branch": "Disneyland"},
                "relevance_score": 0.95,
                "distance": 0.05
            }
        ]
        mock_vector_manager.return_value = mock_vector_instance
        
        yield mock_vector_instance


def test_query_endpoint_success(client, mock_rag_components):
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
    assert data["sources"][0]["review_id"] == "test_review_1"


def test_query_endpoint_no_context(client, mock_rag_components):
    """Test query when no context is found."""
    # Mock empty context
    mock_rag_components.get_relevant_context.return_value = []
    
    query_data = {
        "question": "What do customers say about Space Mountain?",
        "context_limit": 5,
        "temperature": 0.7
    }
    
    response = client.post("/api/v1/query", json=query_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "Based on customer reviews" in data["answer"]  # Chain still returns answer
    assert data["confidence"] == 0.8  # Default confidence for chain-based approach
    assert len(data["sources"]) == 0


def test_query_endpoint_invalid_data(client):
    """Test query endpoint with invalid data."""
    invalid_data = {
        "question": "",  # Empty question should fail validation
        "context_limit": 0,  # Invalid limit
        "temperature": 3.0  # Invalid temperature
    }
    
    response = client.post("/api/v1/query", json=invalid_data)
    assert response.status_code == 422  # Validation error


@patch('src.disney.api.routes.get_retrieval_manager')
def test_health_endpoint_success(mock_vector_manager, client):
    """Test health endpoint with successful dependency checks."""
    # Mock vector store manager
    mock_vector_instance = AsyncMock()
    mock_vector_instance.get_collection_stats.return_value = {
        "collection_name": "disney_reviews",
        "document_count": 100,
        "last_updated": "2023-01-01T00:00:00",
        "embedding_model": "all-MiniLM-L6-v2"
    }
    mock_vector_manager.return_value = mock_vector_instance
    
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "dependencies" in data
    assert data["dependencies"]["chromadb"] == "healthy"
    assert data["dependencies"]["llm_service"] == "healthy"


def test_status_endpoint(client):
    """Test status endpoint."""
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    
    data = response.json()
    assert data["service"] == "customer-experience-api"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"
    assert "components" in data


def test_query_endpoint_missing_question(client):
    """Test query endpoint with missing question field."""
    invalid_data = {
        "context_limit": 5,
        "temperature": 0.7
    }
    
    response = client.post("/api/v1/query", json=invalid_data)
    assert response.status_code == 422  # Validation error


def test_query_endpoint_negative_context_limit(client):
    """Test query endpoint with negative context limit."""
    query_data = {
        "question": "What do customers say about Space Mountain?",
        "context_limit": -1,
        "temperature": 0.7
    }
    
    response = client.post("/api/v1/query", json=query_data)
    assert response.status_code == 422  # Validation error


def test_query_endpoint_temperature_out_of_range(client):
    """Test query endpoint with temperature out of range."""
    query_data = {
        "question": "What do customers say about Space Mountain?",
        "context_limit": 5,
        "temperature": 2.5  # Should be between 0 and 2
    }
    
    response = client.post("/api/v1/query", json=query_data)
    assert response.status_code == 422  # Validation error


@patch('src.disney.api.routes.get_retrieval_manager')
def test_health_endpoint_chromadb_unavailable(mock_vector_manager, client):
    """Test health endpoint when ChromaDB is unavailable."""
    # Mock vector store manager to raise an exception
    mock_vector_instance = AsyncMock()
    mock_vector_instance.get_collection_stats.side_effect = Exception("ChromaDB connection failed")
    mock_vector_manager.return_value = mock_vector_instance
    
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "degraded"
    assert data["dependencies"]["chromadb"] == "unhealthy"


def test_query_endpoint_processing_error(client, mock_rag_components):
    """Test query endpoint when processing fails."""
    # Mock vector manager to raise an exception
    mock_rag_components.query.side_effect = Exception("Chain processing failed")
    
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
