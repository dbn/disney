"""Chain-based vector store management using LangChain chains."""

from functools import cache
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from urllib3 import response

from disney.api.models import QueryResponse

from ..shared.config import settings
from ..shared.logging import setup_logging
from .prompt_template import get_prompt_template
from .query_parser import QueryParser, QueryParserConfig
from operator import itemgetter
from disney.api.models import SourceDocument

logger = setup_logging("retrieval-manager")

def to_source_document(doc:Document) -> SourceDocument:
    """
    Convert a langchain Document (or similar object) to the API SourceDocument model.

    Args:
        doc: A langchain Document or object with .page_content and .metadata

    Returns:
        SourceDocument: Pydantic model for API response
    """
    
    review_id = doc.metadata.get("review_id", "")
    # If score is present (e.g., from similarity_search_with_score), use it; else default to 1.0
    relevance_score = doc.metadata.get("relevance_score", 1.0)
    # If the score is a distance, convert to relevance (assuming 1.0 - score)
    if "score" in doc.metadata:
        try:
            relevance_score = max(0.0, min(1.0, 1.0 - float(doc.metadata["score"])))
        except Exception:
            relevance_score = 1.0
    excerpt = getattr(doc, "page_content", "")
    metadata = doc.metadata if hasattr(doc, "metadata") else {}

    return SourceDocument(
        review_id=review_id,
        relevance_score=relevance_score,
        excerpt=excerpt,
        metadata=metadata
    )

class RetrievalManager:
    """Chain-based retrieval management using LangChain chains."""
    
    def __init__(
        self, 
        chroma_host: Optional[str] = None, 
        chroma_port: Optional[int] = None,
        chroma_client: Optional[chromadb.ClientAPI] = None,
        chroma_settings: Optional[ChromaSettings] = None,
        collection_name: str = "disney_reviews",
        query_parser: Optional[QueryParser] = None
    ):
        """Initialize the chain-based vector store manager.
        
        Args:
            chroma_host: ChromaDB host (ignored if chroma_client provided)
            chroma_port: ChromaDB port (ignored if chroma_client provided)
            chroma_client: Optional ChromaDB client (HttpClient or in-memory Client)
            chroma_settings: Optional ChromaDB settings (ignored if chroma_client provided)
            collection_name: Name of the ChromaDB collection to use
            query_parser: Optional QueryParser for metadata extraction
        """
        self.chroma_host = chroma_host or settings.chroma_host
        self.chroma_port = chroma_port or settings.chroma_port
        self.chroma_client = chroma_client
        self.chroma_settings = chroma_settings
        self.collection_name = collection_name
        
        # Initialize query parser
        if query_parser is None and settings.query_parser_enabled:
            query_parser_config = QueryParserConfig(
                llm_model=settings.query_parser_llm_model,
                temperature=settings.query_parser_temperature,
                max_tokens=settings.query_parser_max_tokens,
                confidence_threshold=settings.query_parser_confidence_threshold
            )
            self.query_parser = QueryParser(query_parser_config)
        else:
            self.query_parser = query_parser
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_llm()
        self._initialize_vector_store()
        self._initialize_chain()
        
        # Log client type for debugging
        client_type = self._detect_client_type(self.chroma_client) if self.chroma_client else "http"
        logger.info(f"RetrievalManager initialized with collection: {self.collection_name} (client: {client_type})")
    
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
        """Initialize the ChromaDB vector store with optional client injection."""
        if self.chroma_client is not None:
            # Use injected client (for testing or custom scenarios)
            logger.info("Using injected ChromaDB client")
        else:
            # Create default HttpClient for external server
            self.chroma_client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port,
                settings=self.chroma_settings or ChromaSettings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
            logger.info(f"Created HttpClient for {self.chroma_host}:{self.chroma_port}")
        
        # Initialize LangChain Chroma vector store
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            client=self.chroma_client
        )
        
        logger.info(f"Initialized vector store: {self.collection_name}")
    
    def _detect_client_type(self, client: chromadb.ClientAPI) -> str:
        """Detect if client is HttpClient or in-memory Client."""
        if hasattr(client, 'get_tenant'):
            return "http"  # HttpClient
        else:
            return "memory"  # In-memory Client
    
   
    def _initialize_chain(self):
        """
        Initialize a chain that takes context and a question, and returns an answer.
        This chain DOES NOT perform retrieval.
        """
        logger.info("Initializing generation chain...")

        # Keep the retriever initialization as is. We'll use it manually later.
        self.retriever = self.vectorstore.as_retriever(
            search_type=settings.retriever_search_type,
            search_kwargs={"k": settings.retriever_k}
        )
        
        # Get your existing prompt template.
        # It must accept "context" and "question" as input variables.
        prompt = get_prompt_template()
        
        # Create the generation chain
        self.generation_chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("Initialized generation chain.")
    
    


    async def query(self, question: str, metadata_filter: dict | None = None) -> QueryResponse:
        """
        Query the vector store with a question and an optional metadata filter.
        
        Args:
            question: The question to ask.
            metadata_filter: A dictionary for filtering metadata, e.g., {"Branch": "Disneyland_Paris"}.
            
        Returns:
            The answer as a string.
        """
        logger.info(f"Received query: '{question}' with filter: {metadata_filter}")
        start_time = time.time()
        try:
            # 1. Define search arguments for the retriever
            search_kwargs = {"k": settings.retriever_k}
            if metadata_filter:
                retrieved_docs = await self.retriever.aget_relevant_documents(
                    question, 
                    filter=metadata_filter
                )
            else:
                retrieved_docs = await self.retriever.aget_relevant_documents(
                    question
                )
            
            # (Your debug function can be called here if you want)
            # inspect_retrieved_docs(retrieved_docs)

            if not retrieved_docs:
                logger.warning("No documents retrieved after filtering.")
                answer = "I couldn't find any relevant information for your question with the provided filters."
                context = "No reviews found for the given question."
            else:            
                context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            answer = await self.generation_chain.ainvoke({
                "context": context,
                "question": question
            })

            response = QueryResponse(
                answer=answer,
                sources=[to_source_document(doc) for doc in retrieved_docs],
                confidence=0.85,  # Could be extracted from parse result
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            return response

        except Exception as e:
            logger.error(f"Error in query: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error while processing your question."



    # Document management methods
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store.
        
        Args:
            documents: List of LangChain documents to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.vectorstore.aadd_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
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
                "retriever_score_threshold": settings.retriever_similarity_threshold
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
    

    async def query_with_metadata(self, query: str) -> QueryResponse:
        """Enhanced query method with metadata extraction.
        
        Args:
            query: User's natural language query
            
        Returns:
            Generated answer using metadata-filtered search
        """
        if not self.query_parser:
            logger.warning("Query parser not available, falling back to regular query")
            return await self.query(query)
        
        try:
            # Parse query for metadata
            parse_result = await self.query_parser.parse_query(query)
            logger.debug(f"Parsed query: {parse_result.search_query}, filters: {parse_result.filters}")
            
            # Build ChromaDB filters from structured result
            chromadb_filters = self.query_parser._build_chromadb_filters(parse_result.filters)
            logger.debug(f"Built ChromaDB filters: {chromadb_filters}")
            
            # Use metadata filters if available
            if chromadb_filters:               
                return await self.query( parse_result.search_query, chromadb_filters )
            else:
                return await self.query(query)
        except Exception as e:
            logger.error(f"Error processing with RAG chain: {str(e)}")
            return "I apologize, but I encountered an error while processing your question."

    async def _process_with_rag_chain(self, docs: List[tuple], query: str) -> str:
        """Process retrieved documents with the RAG chain.
        
        Args:
            docs: List of (document, score) tuples from similarity search
            query: User's query string
            
        Returns:
            Generated answer from the RAG chain
        """
        try:
            # Extract documents from (doc, score) tuples
            formatted_docs = [doc for doc, score in docs]
            
            # Prepare input for the RAG chain
            chain_input = {"context": formatted_docs, "question": query}
            
            # Invoke the RAG chain
            answer = await self.rag_chain.ainvoke(chain_input)
            
            logger.debug(f"RAG chain processed {len(formatted_docs)} documents")
            return answer
            
        except Exception as e:
            logger.error(f"Error processing with RAG chain: {str(e)}")
            return "I apologize, but I encountered an error while processing your question."

    async def get_enhanced_context(
        self, 
        query: str, 
        n_results: int = 5,
        use_metadata_extraction: bool = True
    ) -> List[Dict[str, Any]]:
        """Enhanced context retrieval with metadata filtering.
        
        Args:
            query: User's natural language query
            n_results: Number of results to return
            use_metadata_extraction: Whether to use metadata extraction
            
        Returns:
            List of context documents with metadata filtering
        """
        if not use_metadata_extraction or not self.query_parser:
            return self.get_relevant_context(query, n_results)
        
        try:
            # Parse query for metadata
            parse_result = await self.query_parser.parse_query(query)
            chromadb_filters = self.query_parser._build_chromadb_filters(parse_result.filters)
            
            # Use filtered search if metadata available
            if chromadb_filters:
                logger.info(f"Using metadata filters for context: {chromadb_filters}")
                docs = self.vectorstore.similarity_search_with_score(
                    parse_result.search_query,
                    k=n_results,
                    filter=chromadb_filters
                )
            else:
                docs = self.vectorstore.similarity_search_with_score(
                    parse_result.search_query,
                    k=n_results
                )
            
            return self._format_context_documents(docs, n_results)
            
        except Exception as e:
            logger.error(f"Enhanced context retrieval failed: {e}")
            # Fallback to regular context retrieval
            return self.get_relevant_context(query, n_results)

    def _format_context_documents(self, docs: List[tuple], n_results: int) -> List[Dict[str, Any]]:
        """Format documents for context response.
        
        Args:
            docs: List of (document, score) tuples
            n_results: Maximum number of results to return
            
        Returns:
            List of formatted context documents
        """
        try:
            results = []
            for i, (doc, score) in enumerate(docs[:n_results]):
                results.append({
                    "review_id": doc.metadata.get("review_id", f"doc_{i}"),
                    "relevance_score": 1.0 - score,  # Convert distance to relevance
                    "excerpt": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return results
        except Exception as e:
            logger.error(f"Error formatting context documents: {str(e)}")
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
            "retriever_score_threshold": settings.retriever_similarity_threshold,
            "retriever_search_type": settings.retriever_search_type,
            "llm_temperature": settings.llm_temperature,
            "llm_max_tokens": settings.llm_max_tokens
        }


def get_retrieval_manager(
    chroma_host: Optional[str] = None, 
    chroma_port: Optional[int] = None,
    chroma_client: Optional[chromadb.ClientAPI] = None,
    chroma_settings: Optional[ChromaSettings] = None,
    collection_name: str = "disney_reviews"
) -> RetrievalManager:
    """Get a RetrievalManager instance with optional client injection.
    
    Args:
        chroma_host: ChromaDB host (ignored if chroma_client provided)
        chroma_port: ChromaDB port (ignored if chroma_client provided)
        chroma_client: Optional ChromaDB client (HttpClient or in-memory Client)
        chroma_settings: Optional ChromaDB settings (ignored if chroma_client provided)
        collection_name: Name of the ChromaDB collection to use
    """
    return RetrievalManager(chroma_host, chroma_port, chroma_client, chroma_settings, collection_name)


def get_in_memory_retrieval_manager(collection_name: str = "disney_reviews") -> RetrievalManager:
    """Get RetrievalManager with in-memory ChromaDB for testing.
    
    Args:
        collection_name: Name of the ChromaDB collection to use
    
    Returns:
        RetrievalManager instance with in-memory ChromaDB client
    """
    import chromadb
    client = chromadb.Client()
    return RetrievalManager(chroma_client=client, collection_name=collection_name)


def reset_retrieval_manager():
    """Reset the global RetrievalManager instance."""
    # This would reset a global instance if we had one
    pass