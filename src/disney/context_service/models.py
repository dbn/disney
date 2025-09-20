"""Pydantic models for Context Retrieval Service."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    """Request model for indexing documents."""
    
    documents: List[str] = Field(..., description="List of documents to index")
    metadatas: Optional[List[Dict[str, Any]]] = Field(None, description="Metadata for each document")
    ids: Optional[List[str]] = Field(None, description="Unique IDs for each document")


class IndexResponse(BaseModel):
    """Response model for indexing operation."""
    
    success: bool = Field(..., description="Whether indexing was successful")
    indexed_count: int = Field(..., description="Number of documents indexed")
    message: str = Field(..., description="Status message")


class SearchRequest(BaseModel):
    """Request model for vector search."""
    
    query: str = Field(..., description="Search query")
    n_results: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity threshold")


class SearchResult(BaseModel):
    """Individual search result."""
    
    id: str = Field(..., description="Document ID")
    document: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    distance: float = Field(..., description="Similarity distance")
    score: float = Field(..., description="Similarity score")


class SearchResponse(BaseModel):
    """Response model for search operation."""
    
    results: List[SearchResult] = Field(..., description="Search results")
    query: str = Field(..., description="Original query")
    total_results: int = Field(..., description="Total number of results found")


class StatsResponse(BaseModel):
    """Database statistics response."""
    
    total_documents: int = Field(..., description="Total number of documents")
    collection_name: str = Field(..., description="Collection name")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    database_status: str = Field(..., description="Database connection status")
