"""Tests for Customer Experience Assessment API main module."""

import pytest
from fastapi.testclient import TestClient

from src.disney.api.main import app


@pytest.fixture
def client():
    """Test client for the API."""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "customer-experience-api"


def test_api_docs_available(client):
    """Test that API documentation is available."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_redoc_available(client):
    """Test that ReDoc documentation is available."""
    response = client.get("/redoc")
    assert response.status_code == 200
