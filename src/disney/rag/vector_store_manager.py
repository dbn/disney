"""Vector store management submodule for document indexing and retrieval using LangChain and ChromaDB."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from ..shared.config import settings
from ..shared.logging import setup_logging

logger = setup_logging("vector-store-manager")


class VectorStoreManager:
    """Vector store management component for document indexing and retrieval using LangChain and ChromaDB."""
    
    def __init__(self, chroma_host: Optional[str] = None, chroma_port: Optional[int] = None):
        """Initialize the vector store manager.
        
        Args:
            chroma_host: ChromaDB host
            chroma_port: ChromaDB port
        """
        self.chroma_host = chroma_host or settings.chroma_host
        self.chroma_port = chroma_port or settings.chroma_port
        self.collection_name = "disney_reviews"
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.HttpClient(
            host=self.chroma_host,
            port=self.chroma_port,
            settings=ChromaSettings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )
        
        # Initialize LangChain Chroma vector store
        self.vector_store = None
        self._initialize_vector_store()
        
        logger.info(f"VectorStoreManager initialized with ChromaDB at {self.chroma_host}:{self.chroma_port}")
    
    def _initialize_vector_store(self):
        """Initialize the LangChain Chroma vector store."""
        try:
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            logger.info(f"Vector store initialized with collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            # Create a new collection if it doesn't exist
            try:
                self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Disney reviews collection"}
                )
                self.vector_store = Chroma(
                    client=self.chroma_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings
                )
                logger.info(f"Created new collection: {self.collection_name}")
            except Exception as create_error:
                logger.error(f"Failed to create collection: {str(create_error)}")
                raise
    
    async def get_relevant_context(
        self, 
        query: str, 
        n_results: int = 5,
        similarity_threshold: float = 0.7,
        max_context_length: int = 4000
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a query.
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            similarity_threshold: Minimum similarity threshold
            max_context_length: Maximum context length
            
        Returns:
            List of relevant context documents
        """
        try:
            logger.info(f"Retrieving context for query: {query[:100]}...")
            
            if self.vector_store is None:
                logger.error(f"Vector store not initialized: {self.vector_store}")
                return []
            
            # Use LangChain similarity search with score
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, 
                k=n_results
            )
            
            # Filter by similarity threshold and format results
            relevant_docs = []
            total_length = 0
            
            for doc, score in docs_with_scores:
                # Convert similarity score to relevance score (higher is better)
                relevance_score = 1 - score if score <= 1 else 1 / (1 + score)
                
                if relevance_score >= similarity_threshold:
                    # Check if adding this document would exceed max context length
                    doc_length = len(doc.page_content)
                    if total_length + doc_length <= max_context_length:
                        relevant_docs.append({
                            "id": doc.metadata.get("id", "unknown"),
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "relevance_score": relevance_score,
                            "similarity_score": score
                        })
                        total_length += doc_length
                    else:
                        # Truncate the last document if needed
                        remaining_length = max_context_length - total_length
                        if remaining_length > 100:  # Only add if there's meaningful content
                            truncated_content = doc.page_content[:remaining_length] + "..."
                            relevant_docs.append({
                                "id": doc.metadata.get("id", "unknown"),
                                "content": truncated_content,
                                "metadata": doc.metadata,
                                "relevance_score": relevance_score,
                                "similarity_score": score
                            })
                        break
            
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store.
        
        Args:
            documents: List of LangChain documents to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.vector_store is None:
                logger.error(f"Vector store not initialized: {self.vector_store}")
                return False
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            if not self.chroma_client:
                return {"error": "ChromaDB client not initialized"}
            
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "last_updated": datetime.now().isoformat(),
                "embedding_model": settings.embedding_model
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
    
    async def search_similar(
        self, 
        query: str, 
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents with optional metadata filtering.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar documents
        """
        try:
            if self.vector_store is None:
                logger.error(f"Vector store not initialized: {self.vector_store}")
                return []
            
            # Perform similarity search
            if filter_metadata:
                docs = self.vector_store.similarity_search(
                    query, 
                    k=n_results,
                    filter=filter_metadata
                )
            else:
                docs = self.vector_store.similarity_search(query, k=n_results)
            
            # Format results
            results = []
            for doc in docs:
                results.append({
                    "id": doc.metadata.get("id", "unknown"),
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []


# Global vector store manager instance
_vector_store_manager = None


def get_vector_store_manager(chroma_host: Optional[str] = None, chroma_port: Optional[int] = None) -> VectorStoreManager:
    """Get or create vector store manager instance."""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager(chroma_host, chroma_port)
    return _vector_store_manager


def reset_vector_store_manager():
    """Reset the global vector store manager instance."""
    global _vector_store_manager
    _vector_store_manager = None


# Backward compatibility aliases
def get_retriever(chroma_host: Optional[str] = None, chroma_port: Optional[int] = None) -> VectorStoreManager:
    """Get vector store manager (backward compatibility)."""
    return get_vector_store_manager(chroma_host, chroma_port)


def reset_retriever():
    """Reset vector store manager (backward compatibility)."""
    reset_vector_store_manager()
