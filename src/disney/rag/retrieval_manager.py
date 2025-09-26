"""Chain-based vector store management using LangChain chains."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from ..shared.config import settings
from ..shared.logging import setup_logging
from .prompt_template import get_prompt_template

logger = setup_logging("vector-store-manager")


class RetrievalManager:
    """Chain-based retrieval management using LangChain chains."""
    
    def __init__(self, chroma_host: Optional[str] = None, chroma_port: Optional[int] = None):
        """Initialize the chain-based vector store manager.
        
        Args:
            chroma_host: ChromaDB host
            chroma_port: ChromaDB port
        """
        self.chroma_host = chroma_host or settings.chroma_host
        self.chroma_port = chroma_port or settings.chroma_port
        self.collection_name = "disney_reviews"
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_llm()
        self._initialize_vector_store()
        self._initialize_chain()
        
        logger.info(f"RetrievalManager initialized with collection: {self.collection_name}")
    
    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        logger.info(f"Initialized embeddings: {settings.embedding_model}")
    
    def _initialize_llm(self):
        """Initialize the language model."""
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key
        )
        logger.info(f"Initialized LLM: {settings.llm_model}")
    
    def _initialize_vector_store(self):
        """Initialize the ChromaDB vector store."""
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
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            client=self.chroma_client
        )
        
        logger.info(f"Initialized vector store: {self.collection_name}")
    
    def _initialize_chain(self):
        """Initialize the RAG chain."""
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type=settings.retriever_search_type,
            search_kwargs={
                "k": settings.retriever_k,
                "score_threshold": settings.retriever_score_threshold
            }
        )
        
        # Create prompt template
        prompt = get_prompt_template()
        
        # Create the RAG chain
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("Initialized RAG chain")
    
    # New simple query method
    async def query(self, question: str) -> str:
        """Query the vector store using the RAG chain.
        
        Args:
            question: The question to ask
            
        Returns:
            The answer as a string
        """
        try:
            return await self.rag_chain.ainvoke(question)
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            return "I apologize, but I encountered an error while processing your question."
    
    # Document management methods
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store.
        
        Args:
            documents: List of LangChain documents to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> bool:
        """Add texts to the vector store.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vectorstore.add_texts(texts, metadatas)
            logger.info(f"Added {len(texts)} texts to vector store")
            return True
        except Exception as e:
            logger.error(f"Error adding texts: {str(e)}")
            return False
    
    # Utility methods
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "last_updated": datetime.now().isoformat(),
                "embedding_model": settings.embedding_model,
                "llm_model": settings.llm_model,
                "retriever_k": settings.retriever_k,
                "retriever_score_threshold": settings.retriever_score_threshold
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "error": str(e)
            }
    
    def search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search for similar documents with scores.
        
        Args:
            query: Query string
            k: Number of documents to return
            
        Returns:
            List of (document, score) tuples
        """
        try:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"Error in similarity search with score: {str(e)}")
            return []
    
    def delete_collection(self) -> bool:
        """Delete the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.chroma_client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset the collection (delete and recreate).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.delete_collection()
            self._initialize_vector_store()
            self._initialize_chain()
            logger.info(f"Reset collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            return False
    
    # Backward compatibility methods
    def get_relevant_context(
        self, 
        query: str, 
        n_results: int = 5, 
        similarity_threshold: float = 0.7,
        max_context_length: int = 4000
    ) -> List[Dict[str, Any]]:
        """Get relevant context (backward compatibility method).
        
        Args:
            query: Query string
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score
            max_context_length: Maximum context length
            
        Returns:
            List of relevant context dictionaries
        """
        try:
            # Use the retriever to get documents
            docs = self.retriever.get_relevant_documents(query)
            
            # Convert to old format
            results = []
            for i, doc in enumerate(docs[:n_results]):
                results.append({
                    "id": f"doc_{i}",
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": 1.0 - (i * 0.1),  # Approximate score
                    "distance": i * 0.1
                })
            
            return results
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return []
    
    def _get_vector_store(self):
        """Get the vector store (backward compatibility)."""
        return self.vectorstore
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get information about the chain and configuration.
        
        Returns:
            Dictionary with chain and configuration information
        """
        return {
            "collection_name": self.collection_name,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "retriever_k": settings.retriever_k,
            "retriever_score_threshold": settings.retriever_score_threshold,
            "retriever_search_type": settings.retriever_search_type,
            "llm_temperature": settings.llm_temperature,
            "llm_max_tokens": settings.llm_max_tokens
        }


# Factory functions for backward compatibility
def get_retrieval_manager(chroma_host: Optional[str] = None, chroma_port: Optional[int] = None) -> RetrievalManager:
    """Get a RetrievalManager instance."""
    return RetrievalManager(chroma_host, chroma_port)


def reset_retrieval_manager():
    """Reset the global RetrievalManager instance."""
    # This would reset a global instance if we had one
    pass