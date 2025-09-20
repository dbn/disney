"""ChromaDB operations for Context Retrieval Service."""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from ..shared.config import settings
from ..shared.logging import setup_logging

logger = setup_logging("context-service")


class VectorDatabase:
    """ChromaDB wrapper for vector operations."""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="disney_reviews",
                metadata={"description": "Disney park customer reviews"}
            )
            
            logger.info("ChromaDB client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise
    
    def add_documents(
        self, 
        documents: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> bool:
        """Add documents to the vector database.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.collection:
                raise Exception("Collection not initialized")
            
            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Add documents to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas or [{}] * len(documents),
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def search_documents(
        self, 
        query: str, 
        n_results: int = 5,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        try:
            if not self.collection:
                raise Exception("Collection not initialized")
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i, (doc, metadata, distance, id) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0],
                results['ids'][0]
            )):
                # Calculate similarity score (1 - distance)
                score = 1 - distance
                
                # Apply similarity threshold if provided
                if similarity_threshold and score < similarity_threshold:
                    continue
                
                formatted_results.append({
                    'id': id,
                    'document': doc,
                    'metadata': metadata,
                    'distance': distance,
                    'score': score
                })
            
            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            if not self.collection:
                raise Exception("Collection not initialized")
            
            # Get collection count
            count = self.collection.count()
            
            return {
                'total_documents': count,
                'collection_name': self.collection.name,
                'last_updated': None  # ChromaDB doesn't provide this directly
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                'total_documents': 0,
                'collection_name': 'unknown',
                'last_updated': None
            }


# Global vector database instance
_vector_db = None


def get_vector_db() -> VectorDatabase:
    """Get or create vector database instance."""
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDatabase()
    return _vector_db
