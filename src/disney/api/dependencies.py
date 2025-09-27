"""Dependency injection for Customer Experience Assessment API."""

from typing import Generator, Optional
import httpx

from ..shared.config import settings
from ..shared.logging import setup_logging
from ..rag.retrieval_manager import RetrievalManager

logger = setup_logging("customer-experience-api")

# Global cache for RetrievalManager instances
_retrieval_manager_cache: Optional[RetrievalManager] = None


async def get_http_client() -> Generator[httpx.AsyncClient, None, None]:
    """Get HTTP client for external service communication."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


def get_chroma_host() -> str:
    """Get ChromaDB host."""
    return settings.chroma_host


def get_chroma_port() -> int:
    """Get ChromaDB port."""
    return settings.chroma_port


def get_retrieval_manager() -> RetrievalManager:
    """Get cached RetrievalManager instance.
    
    This dependency provides a singleton RetrievalManager instance
    that is cached and reused across all requests, avoiding expensive
    re-initialization of ChromaDB clients, embeddings, and vector stores.
    
    Returns:
        Cached RetrievalManager instance
    """
    global _retrieval_manager_cache
    
    if _retrieval_manager_cache is None:
        logger.info("Creating new RetrievalManager instance (first request)")
        _retrieval_manager_cache = RetrievalManager(
            chroma_host=settings.chroma_host,
            chroma_port=settings.chroma_port
        )
        logger.info("RetrievalManager instance created and cached")
    else:
        logger.debug("Using cached RetrievalManager instance")
    
    return _retrieval_manager_cache


def reset_retrieval_manager():
    """Reset the cached RetrievalManager instance.
    
    This is useful for testing or when you need to force
    re-initialization of the RetrievalManager.
    """
    global _retrieval_manager_cache
    _retrieval_manager_cache = None
    logger.info("RetrievalManager cache reset")


def get_retrieval_manager_stats() -> dict:
    """Get RetrievalManager cache statistics.
    
    Returns:
        Dictionary with cache statistics
    """
    return {
        "is_cached": _retrieval_manager_cache is not None,
        "instance_type": type(_retrieval_manager_cache).__name__ if _retrieval_manager_cache else None
    }
