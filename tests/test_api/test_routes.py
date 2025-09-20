"""Tests for Customer Experience Assessment API routes."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from src.disney.api.main import app


@pytest.fixture
def client():
    """Test client for the API."""
    return TestClient(app)


@pytest.fixture
def mock_rag_components():
    """Mock RAG components for testing."""
    with patch('src.disney.api.routes.get_retriever') as mock_retriever, \
         patch('src.disney.api.routes.get_generator') as mock_generator:
        
        # Mock retriever
        mock_retriever_instance = AsyncMock()
        mock_retriever_instance.get_relevant_context.return_value = [
            {
                "id": "test_review_1",
                "content": "Space Mountain was amazing! The wait was worth it.",
                "metadata": {"rating": 5, "branch": "Disneyland"},
                "relevance_score": 0.95,
                "distance": 0.05
            }
        ]
        mock_retriever.return_value = mock_retriever_instance
        
        # Mock generator
        mock_generator_instance = AsyncMock()
        mock_generator_instance.generate_answer.return_value = {
            "answer": "Based on customer reviews, Space Mountain is highly rated with customers saying the wait is worth it.",
            "confidence": 0.87,
            "context_used": 1,
            "context_length": 100
        }
        mock_generator.return_value = mock_generator_instance
        
        yield mock_retriever_instance, mock_generator_instance


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
    mock_retriever, mock_generator = mock_rag_components
    mock_retriever.get_relevant_context.return_value = []
    
    query_data = {
        "question": "What do customers say about Space Mountain?",
        "context_limit": 5,
        "temperature": 0.7
    }
    
    response = client.post("/api/v1/query", json=query_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "I apologize, but I couldn't find any relevant information" in data["answer"]
    assert data["confidence"] == 0.0
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


@patch('src.disney.api.routes.get_http_client')
def test_health_endpoint_success(mock_http_client, client):
    """Test health endpoint with successful dependency checks."""
    # Mock HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value.status_code = 200
    mock_http_client.return_value.__aenter__.return_value = mock_client
    
    # Mock generator
    with patch('src.disney.api.routes.get_generator') as mock_generator:
        mock_generator.return_value = AsyncMock()
        
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "dependencies" in data
        assert data["dependencies"]["context_service"] == "healthy"
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