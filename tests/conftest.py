"""Pytest configuration and fixtures for Disney API tests."""

import pytest
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Set test environment variables before importing the app
os.environ["CHROMA_HOST"] = "localhost"
os.environ["CHROMA_PORT"] = "8000"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["QUERY_PARSER_ENABLED"] = "false"

from disney.api.main import app
from disney.rag.retrieval_manager import get_in_memory_retrieval_manager
from disney.api.dependencies import get_retrieval_manager


@pytest.fixture(scope="session")
def client():
    """Create test client with mocked RetrievalManager."""
    # We'll override the dependency in individual test fixtures
    return TestClient(app)


@pytest.fixture(scope="function")
def mock_retrieval_manager():
    """Mock RetrievalManager for testing."""
    from disney.api.models import QueryResponse, SourceDocument
    from unittest.mock import AsyncMock
    
    # Create a completely mocked RetrievalManager
    manager = MagicMock()
    
    mock_response = QueryResponse(
        answer="Test answer from mock",
        sources=[
            SourceDocument(
                review_id="test_1",
                relevance_score=0.95,
                excerpt="Test excerpt",
                metadata={"rating": 5, "branch": "Disneyland"}
            )
        ],
        confidence=0.87,
        processing_time_ms=1000.0
    )
    
    # Mock all the methods we need
    manager.query_with_metadata = AsyncMock(return_value=mock_response)
    manager.query = AsyncMock(return_value="Test answer from mock")
    manager.get_collection_stats = MagicMock(return_value={
        "collection_name": "disney_reviews",
        "document_count": 100,
        "last_updated": "2023-01-01T00:00:00",
        "embedding_model": "all-MiniLM-L6-v2"
    })
    manager.is_cache_enabled = MagicMock(return_value=True)
    
    return manager


@pytest.fixture(scope="function")
def mock_retrieval_manager_with_patch(client, mock_retrieval_manager):
    """Patch the get_retrieval_manager dependency with mock."""
    # Override the dependency in the app
    app.dependency_overrides[get_retrieval_manager] = lambda: mock_retrieval_manager
    yield mock_retrieval_manager
    # Clean up after test
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def mock_retrieval_manager_routes_patch(client, mock_retrieval_manager):
    """Patch the get_retrieval_manager dependency in routes with mock."""
    # Override the dependency in the app
    app.dependency_overrides[get_retrieval_manager] = lambda: mock_retrieval_manager
    yield mock_retrieval_manager
    # Clean up after test
    app.dependency_overrides.clear()