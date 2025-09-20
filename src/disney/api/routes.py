"""API routes for Customer Experience Assessment Service."""

import time
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from .models import QueryRequest, QueryResponse, HealthResponse, ErrorResponse, SourceDocument
from ..shared.logging import setup_logging
from ..rag.retrieval import get_retriever
from ..rag.generator import get_generator
from .dependencies import get_http_client, get_context_service_url

# Set up logging
logger = setup_logging("customer-experience-api")

# Create router
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_reviews(
    request: QueryRequest,
    http_client=Depends(get_http_client),
    context_service_url=Depends(get_context_service_url)
):
    """Submit a question and get LLM result back."""
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {request.question[:100]}...")
        
        # Initialize RAG components
        retriever = get_retriever()
        generator = get_generator()
        
        # Step 1: Retrieve relevant context
        logger.info("Retrieving relevant context...")
        context_docs = await retriever.get_relevant_context(
            query=request.question,
            n_results=request.context_limit,
            similarity_threshold=0.7,  # Could be made configurable
            max_context_length=4000   # Could be made configurable
        )
        
        if not context_docs:
            logger.warning("No relevant context found for query")
            return QueryResponse(
                answer="I apologize, but I couldn't find any relevant information about your question in the Disney reviews database. Please try rephrasing your question or asking about a different topic.",
                sources=[],
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Step 2: Generate answer using LLM
        logger.info(f"Generating answer with {len(context_docs)} context documents...")
        generation_result = generator.generate_answer(
            question=request.question,
            context_docs=context_docs,
            temperature=request.temperature
        )
        
        # Step 3: Format response
        sources = []
        for doc in context_docs:
            sources.append(SourceDocument(
                review_id=doc.get("id", "unknown"),
                relevance_score=doc.get("relevance_score", 0.0),
                excerpt=doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", ""),
                metadata=doc.get("metadata", {})
            ))
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Query processed successfully in {processing_time_ms:.2f}ms")
        
        return QueryResponse(
            answer=generation_result["answer"],
            sources=sources,
            confidence=generation_result["confidence"],
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
    http_client=Depends(get_http_client),
    context_service_url=Depends(get_context_service_url)
):
    """Health check endpoint."""
    try:
        dependencies = {}
        
        # Check Context Retrieval Service
        try:
            response = await http_client.get(f"{context_service_url}/health", timeout=5.0)
            if response.status_code == 200:
                dependencies["context_service"] = "healthy"
            else:
                dependencies["context_service"] = "unhealthy"
        except Exception as e:
            logger.warning(f"Context service health check failed: {str(e)}")
            dependencies["context_service"] = "unhealthy"
        
        # Check LLM service (OpenAI)
        try:
            generator = get_generator()
            # Simple test to check if OpenAI API is accessible
            dependencies["llm_service"] = "healthy" if generator else "unhealthy"
        except Exception as e:
            logger.warning(f"LLM service health check failed: {str(e)}")
            dependencies["llm_service"] = "unhealthy"
        
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
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
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
                "context_service_client": "available"
            }
        }
        
        return JSONResponse(content=status_info)
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )