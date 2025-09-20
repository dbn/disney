"""Tests for Customer Experience Assessment API routes."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.disney.api.main import app


@pytest.fixture
def client():
    """Test client for the API."""
    return TestClient(app)


def test_query_endpoint_placeholder(client):
    """Test query endpoint with placeholder response."""
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


def test_query_endpoint_invalid_data(client):
    """Test query endpoint with invalid data."""
    invalid_data = {
        "question": "",  # Empty question should fail validation
        "context_limit": 0,  # Invalid limit
        "temperature": 3.0  # Invalid temperature
    }
    
    response = client.post("/api/v1/query", json=invalid_data)
    assert response.status_code == 422  # Validation error


def test_health_endpoint(client):
    """Test health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "dependencies" in data
