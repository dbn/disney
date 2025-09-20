"""RAG retrieval submodule for finding relevant context."""

from typing import List, Dict, Any, Optional
import httpx

from ..shared.config import settings
from ..shared.logging import setup_logging
from .embedder import get_embedder

logger = setup_logging("rag-retrieval")


class RAGRetriever:
    """RAG retrieval component for finding relevant context."""
    
    def __init__(self, context_service_url: Optional[str] = None):
        """Initialize the RAG retriever.
        
        Args:
            context_service_url: URL of the Context Retrieval Service
        """
        self.context_service_url = context_service_url or settings.context_service_url
        self.embedder = get_embedder()
    
    async def retrieve_context(
        self, 
        query: str, 
        n_results: int = 5,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a query.
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of relevant context documents
        """
        try:
            logger.info(f"Retrieving context for query: {query[:100]}...")
            
            # Use Context Retrieval Service for vector search
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.context_service_url}/api/v1/search",
                    json={
                        "query": query,
                        "n_results": n_results,
                        "similarity_threshold": similarity_threshold
                    }
                )
                response.raise_for_status()
                
                search_data = response.json()
                results = search_data.get("results", [])
                
                # Format results for RAG pipeline
                context_docs = []
                for result in results:
                    context_docs.append({
                        "id": result["id"],
                        "content": result["document"],
                        "metadata": result.get("metadata", {}),
                        "relevance_score": result["score"],
                        "distance": result["distance"]
                    })
                
                logger.info(f"Retrieved {len(context_docs)} context documents")
                return context_docs
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error retrieving context: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def rank_context(
        self, 
        context_docs: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Rank context documents by relevance.
        
        Args:
            context_docs: List of context documents
            query: Original query
            
        Returns:
            Ranked list of context documents
        """
        try:
            # Sort by relevance score (descending)
            ranked_docs = sorted(
                context_docs, 
                key=lambda x: x.get("relevance_score", 0), 
                reverse=True
            )
            
            logger.info(f"Ranked {len(ranked_docs)} context documents")
            return ranked_docs
            
        except Exception as e:
            logger.error(f"Error ranking context: {str(e)}")
            return context_docs
    
    def filter_context(
        self, 
        context_docs: List[Dict[str, Any]], 
        max_length: int = 4000
    ) -> List[Dict[str, Any]]:
        """Filter context documents to fit within length limits.
        
        Args:
            context_docs: List of context documents
            max_length: Maximum total context length
            
        Returns:
            Filtered list of context documents
        """
        try:
            filtered_docs = []
            current_length = 0
            
            for doc in context_docs:
                doc_length = len(doc.get("content", ""))
                
                if current_length + doc_length <= max_length:
                    filtered_docs.append(doc)
                    current_length += doc_length
                else:
                    # Truncate the last document if needed
                    remaining_length = max_length - current_length
                    if remaining_length > 100:  # Only add if meaningful length
                        truncated_doc = doc.copy()
                        truncated_doc["content"] = doc["content"][:remaining_length]
                        filtered_docs.append(truncated_doc)
                    break
            
            logger.info(f"Filtered to {len(filtered_docs)} context documents ({current_length} chars)")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error filtering context: {str(e)}")
            return context_docs
    
    async def get_relevant_context(
        self, 
        query: str, 
        n_results: int = 5,
        similarity_threshold: Optional[float] = None,
        max_context_length: int = 4000
    ) -> List[Dict[str, Any]]:
        """Get relevant context for a query with full pipeline.
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            similarity_threshold: Minimum similarity threshold
            max_context_length: Maximum total context length
            
        Returns:
            List of relevant context documents
        """
        try:
            # Retrieve context
            context_docs = await self.retrieve_context(
                query=query,
                n_results=n_results,
                similarity_threshold=similarity_threshold
            )
            
            # Rank by relevance
            ranked_docs = self.rank_context(context_docs, query)
            
            # Filter by length
            filtered_docs = self.filter_context(ranked_docs, max_context_length)
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return []


# Global retriever instance
_retriever = None


def get_retriever() -> RAGRetriever:
    """Get or create retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
