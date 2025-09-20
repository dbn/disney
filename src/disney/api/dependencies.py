"""Dependency injection for Customer Experience Assessment API."""

from typing import Generator
import httpx

from ..shared.config import settings
from ..shared.logging import setup_logging

logger = setup_logging("customer-experience-api")


def get_http_client() -> Generator[httpx.AsyncClient, None, None]:
    """Get HTTP client for external service communication."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


def get_context_service_url() -> str:
    """Get Context Retrieval Service URL."""
    return settings.context_service_url
