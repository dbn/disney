"""API routes for Customer Experience Assessment Service."""

import time
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from .models import QueryRequest, QueryResponse, HealthResponse, ErrorResponse, SourceDocument
from ..shared.logging import setup_logging
from ..rag.retrieval_manager import RetrievalManager
from .dependencies import get_http_client, get_retrieval_manager, get_retrieval_manager_stats

# Set up logging
logger = setup_logging("customer-experience-api")

# Create router
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_reviews(
    request: QueryRequest,
    vector_manager: RetrievalManager = Depends(get_retrieval_manager)
):
    """Submit a question and get LLM result back."""
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {request.question[:100]}...")
        
        # Use the cached RetrievalManager instance
        result = await vector_manager.query(request.question)
        
        # Format response
        sources = []
        for doc in result["context"][:request.context_limit]:
            sources.append(SourceDocument(
                review_id=doc.get("id", "unknown"),
                relevance_score=doc.get("relevance_score", 0.0),
                excerpt=doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", ""),
                metadata=doc.get("metadata", {})
            ))
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Query processed successfully in {processing_time_ms:.2f}ms")
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            confidence=0.8,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(
    vector_manager: RetrievalManager = Depends(get_retrieval_manager)
):
    """Health check endpoint."""
    try:
        dependencies = {}
        
        # Check ChromaDB Service using cached instance
        try:
            stats = vector_manager.get_collection_stats()
            if stats and stats.get("document_count", 0) >= 0:
                dependencies["chromadb"] = "healthy"
            else:
                dependencies["chromadb"] = "unhealthy"
        except Exception as e:
            logger.warning(f"ChromaDB health check failed: {str(e)}")
            dependencies["chromadb"] = "unhealthy"
        
        # Check LLM service (already initialized in RetrievalManager)
        dependencies["llm_service"] = "healthy" if vector_manager else "unhealthy"
        
        # Overall health status
        overall_status = "healthy" if all(
            status == "healthy" for status in dependencies.values()
        ) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            version="1.0.0",
            dependencies=dependencies
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            dependencies={"error": str(e)}
        )

@router.get("/status")
async def get_status():
    """Get detailed service status and metrics."""
    try:
        # Get basic service info
        status_info = {
            "service": "customer-experience-api",
            "version": "1.0.0",
            "status": "running",
            "timestamp": time.time(),
            "components": {
                "rag_retriever": "available",
                "rag_generator": "available",
                "chromadb_client": "available"
            }
        }
        
        return JSONResponse(content=status_info)
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )

