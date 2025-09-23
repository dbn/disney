"""API routes for Customer Experience Assessment Service."""

import time
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from .models import QueryRequest, QueryResponse, HealthResponse, ErrorResponse, SourceDocument
from ..shared.logging import setup_logging
from ..rag.vector_store_manager import get_vector_store_manager
from ..rag.generator import get_generator
from .dependencies import get_http_client, get_chroma_host, get_chroma_port

# Set up logging
logger = setup_logging("customer-experience-api")

# Create router
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_reviews(
    request: QueryRequest,
    
    chroma_host=Depends(get_chroma_host),
    chroma_port=Depends(get_chroma_port)
):
    """Submit a question and get LLM result back."""
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {request.question[:100]}...")
        
        # Initialize RAG components
        vector_manager = get_vector_store_manager(chroma_host, chroma_port)
        generator = get_generator()
        
        # Step 1: Retrieve relevant context
        logger.info("Retrieving relevant context...")
        context_docs = await vector_manager.get_relevant_context(
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
    chroma_host=Depends(get_chroma_host),
    chroma_port=Depends(get_chroma_port)
):
    """Health check endpoint."""
    try:
        dependencies = {}
        
        # Check ChromaDB Service (direct connection)
        try:
            vector_manager = get_vector_store_manager(chroma_host, chroma_port)
            stats = await vector_manager.get_collection_stats()
            if stats and stats.get("document_count", 0) >= 0:
                dependencies["chromadb"] = "healthy"
            else:
                dependencies["chromadb"] = "unhealthy"
        except Exception as e:
            logger.warning(f"ChromaDB health check failed: {str(e)}")
            dependencies["chromadb"] = "unhealthy"
        
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