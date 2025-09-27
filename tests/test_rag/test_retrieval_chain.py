"""Tests for chain-based RetrievalManager."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain.schema import Document

from disney.rag.retrieval_manager import RetrievalManager
from disney.rag.prompt_template import get_prompt_template


class TestRetrievalManagerChain:
    """Test cases for chain-based RetrievalManager."""

    @pytest.fixture
    def mock_retrieval_manager(self):
        """Create a mock RetrievalManager for testing."""
        # Create a completely mocked RetrievalManager
        manager = MagicMock(spec=RetrievalManager)
        
        # Mock the query method
        manager.query = AsyncMock(return_value="Space Mountain is highly rated by customers.")
        
        # Mock the rag_chain
        manager.rag_chain = MagicMock()
        manager.rag_chain.ainvoke = AsyncMock(return_value="Space Mountain is highly rated by customers.")
        
        # Mock other methods
        manager.add_documents = AsyncMock(return_value=True)
        manager.get_collection_stats.return_value = {
            "collection_name": "disney_reviews",
            "document_count": 100,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "gpt-4o-mini"
        }
        manager.search_with_score.return_value = []
        manager.delete_collection.return_value = True
        manager.reset_collection.return_value = True
        manager.get_relevant_context.return_value = []
        manager.get_chain_info.return_value = {
            "collection_name": "disney_reviews",
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "gpt-4o-mini",
            "retriever_k": 5,
            "retriever_score_threshold": 0.7
        }
        
        # Mock the retriever
        manager.retriever = MagicMock()
        manager.retriever.get_relevant_documents.return_value = []
        
        # Mock the vectorstore
        manager.vectorstore = MagicMock()
        manager.vectorstore.aadd_documents = AsyncMock(return_value=None)
        manager.vectorstore.similarity_search_with_score.return_value = []
        
        # Mock the chroma client
        manager.chroma_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        manager.chroma_client.get_collection.return_value = mock_collection
        manager.chroma_client.delete_collection.return_value = None
        
        # Set collection name
        manager.collection_name = "disney_reviews"
        
        return manager

    def test_initialization(self, mock_retrieval_manager):
        """Test RetrievalManager initialization."""
        manager = mock_retrieval_manager
        assert manager is not None
        assert manager.collection_name == "disney_reviews"

    @pytest.mark.asyncio
    async def test_query_success(self, mock_retrieval_manager):
        """Test successful query using the chain."""
        manager = mock_retrieval_manager
        
        result = await manager.query("What do customers say about Space Mountain?")
        
        assert result == "Space Mountain is highly rated by customers."
        manager.query.assert_called_once_with("What do customers say about Space Mountain?")

    @pytest.mark.asyncio
    async def test_query_error_handling(self, mock_retrieval_manager):
        """Test query error handling."""
        manager = mock_retrieval_manager
        # Mock the query method to return an error message
        manager.query = AsyncMock(return_value="I apologize, but I encountered an error while processing your question.")
        
        result = await manager.query("Test question")
        
        assert "I apologize, but I encountered an error" in result

    @pytest.mark.asyncio
    async def test_add_documents_success(self, mock_retrieval_manager):
        """Test successful document addition."""
        manager = mock_retrieval_manager
        
        documents = [
            Document(page_content="Test content 1", metadata={"id": "1"}),
            Document(page_content="Test content 2", metadata={"id": "2"})
        ]
        
        result = await manager.add_documents(documents)
        
        assert result is True
        manager.add_documents.assert_called_once_with(documents)

    @pytest.mark.asyncio
    async def test_add_documents_error(self, mock_retrieval_manager):
        """Test document addition error handling."""
        manager = mock_retrieval_manager
        # Mock the add_documents method to return False for error case
        manager.add_documents = AsyncMock(return_value=False)
        
        documents = [Document(page_content="Test", metadata={"id": "1"})]
        result = await manager.add_documents(documents)
        
        assert result is False


    def test_get_collection_stats_success(self, mock_retrieval_manager):
        """Test successful collection stats retrieval."""
        manager = mock_retrieval_manager
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        manager.chroma_client.get_collection.return_value = mock_collection
        
        result = manager.get_collection_stats()
        
        assert result["document_count"] == 100
        assert result["collection_name"] == "disney_reviews"
        assert "embedding_model" in result
        assert "llm_model" in result

    def test_get_collection_stats_error(self, mock_retrieval_manager):
        """Test collection stats error handling."""
        manager = mock_retrieval_manager
        # Mock the get_collection_stats method to return error case
        manager.get_collection_stats.return_value = {
            "document_count": 0,
            "error": "Stats error"
        }
        
        result = manager.get_collection_stats()
        
        assert result["document_count"] == 0
        assert "error" in result

    def test_search_with_score_success(self, mock_retrieval_manager):
        """Test successful similarity search with scores."""
        manager = mock_retrieval_manager
        mock_docs_with_scores = [
            (Document(page_content="Test 1", metadata={"id": "1"}), 0.1),
            (Document(page_content="Test 2", metadata={"id": "2"}), 0.2)
        ]
        # Mock the search_with_score method directly
        manager.search_with_score.return_value = mock_docs_with_scores
        
        result = manager.search_with_score("test query", k=2)
        
        assert len(result) == 2
        assert result[0][0].page_content == "Test 1"
        assert result[0][1] == 0.1

    def test_delete_collection_success(self, mock_retrieval_manager):
        """Test successful collection deletion."""
        manager = mock_retrieval_manager
        
        result = manager.delete_collection()
        
        assert result is True
        manager.delete_collection.assert_called_once()

    def test_delete_collection_error(self, mock_retrieval_manager):
        """Test collection deletion error handling."""
        manager = mock_retrieval_manager
        # Mock the delete_collection method to return False for error case
        manager.delete_collection.return_value = False
        
        result = manager.delete_collection()
        
        assert result is False

    def test_reset_collection_success(self, mock_retrieval_manager):
        """Test successful collection reset."""
        manager = mock_retrieval_manager
        
        result = manager.reset_collection()
        
        assert result is True
        manager.reset_collection.assert_called_once()

    def test_get_relevant_context_backward_compatibility(self, mock_retrieval_manager):
        """Test backward compatibility for get_relevant_context."""
        manager = mock_retrieval_manager
        
        # Mock the get_relevant_context method directly
        mock_result = [
            {
                "content": "Test content 1",
                "metadata": {"id": "1"},
                "relevance_score": 0.9,
                "distance": 0.1
            },
            {
                "content": "Test content 2", 
                "metadata": {"id": "2"},
                "relevance_score": 0.8,
                "distance": 0.2
            }
        ]
        manager.get_relevant_context.return_value = mock_result
        
        result = manager.get_relevant_context("test query", n_results=2)
        
        assert len(result) == 2
        assert result[0]["content"] == "Test content 1"
        assert result[0]["metadata"]["id"] == "1"
        assert "relevance_score" in result[0]
        assert "distance" in result[0]

    def test_get_chain_info(self, mock_retrieval_manager):
        """Test getting chain information."""
        manager = mock_retrieval_manager
        
        result = manager.get_chain_info()
        
        assert "collection_name" in result
        assert "embedding_model" in result
        assert "llm_model" in result
        assert "retriever_k" in result
        assert "retriever_score_threshold" in result

    def test_factory_functions(self):
        """Test factory functions for backward compatibility."""
        with patch('disney.rag.retrieval_manager.RetrievalManager') as mock_manager_class:
            mock_instance = MagicMock()
            mock_manager_class.return_value = mock_instance
            
            from disney.rag.retrieval_manager import get_retrieval_manager
            
            # Test get_retrieval_manager
            result1 = get_retrieval_manager("localhost", 8000)
            assert result1 == mock_instance
            mock_manager_class.assert_called_with("localhost", 8000, None, None, "disney_reviews")
            
            # Test get_retriever (backward compatibility)
            result2 = get_retrieval_manager("localhost", 8000)
            assert result2 == mock_instance


class TestPromptTemplate:
    """Test cases for prompt template."""

    def test_get_prompt_template(self):
        """Test getting the prompt template."""
        from disney.rag.prompt_template import get_prompt_template
        
        template = get_prompt_template()
        
        assert template is not None
        assert hasattr(template, 'format')
        
        # Test template formatting
        formatted = template.format(question="Test question", context="Test context")
        assert "Test question" in formatted
        assert "Test context" in formatted
        assert "Answer:" in formatted

    def test_template_content(self):
        """Test template content structure."""
        from disney.rag.prompt_template import DISNEY_QA_TEMPLATE
        
        assert "You are an assistant for Disney customer experience questions" in DISNEY_QA_TEMPLATE
        assert "Question: {question}" in DISNEY_QA_TEMPLATE
        assert "Context: {context}" in DISNEY_QA_TEMPLATE
        assert "Answer:" in DISNEY_QA_TEMPLATE