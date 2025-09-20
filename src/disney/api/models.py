"""Pydantic models for Customer Experience Assessment API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    
    question: str = Field(..., description="Natural language question about Disney reviews")
    context_limit: int = Field(default=5, ge=1, le=20, description="Maximum number of context documents")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature setting")


class SourceDocument(BaseModel):
    """Source document model."""
    
    review_id: str = Field(..., description="Unique identifier for the review")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    excerpt: str = Field(..., description="Relevant text excerpt")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(..., description="Source documents used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    dependencies: dict = Field(..., description="Dependency status")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: float = Field(..., description="Error timestamp")
