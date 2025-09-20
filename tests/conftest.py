"""Pytest configuration and fixtures."""

import pytest
import asyncio
from typing import AsyncGenerator
import httpx
from unittest.mock import Mock, AsyncMock

from src.disney.shared.config import settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for testing."""
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
def mock_context_service():
    """Mock Context Retrieval Service responses."""
    mock_service = Mock()
    mock_service.search.return_value = {
        "results": [
            {
                "id": "test_doc_1",
                "document": "This is a test Disney review about Space Mountain.",
                "metadata": {"rating": 5, "branch": "Disneyland"},
                "distance": 0.1,
                "score": 0.9
            }
        ],
        "query": "test query",
        "total_results": 1
    }
    mock_service.index.return_value = {
        "success": True,
        "indexed_count": 1,
        "message": "Documents indexed successfully"
    }
    return mock_service


@pytest.fixture
def sample_review_data():
    """Sample Disney review data for testing."""
    return {
        "Review_Text": "Space Mountain was amazing! The wait was worth it.",
        "Rating": 5,
        "Year_Month": "2023-01",
        "Branch": "Disneyland"
    }


@pytest.fixture
def sample_query_request():
    """Sample query request for testing."""
    return {
        "question": "What do customers say about Space Mountain?",
        "context_limit": 5,
        "temperature": 0.7
    }
