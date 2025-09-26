"""Tests for chain-based RetrievalManager."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain.schema import Document

from src.disney.rag.retrieval_manager import RetrievalManager
from src.disney.rag.prompt_template import get_prompt_template


class TestRetrievalManagerChain:
    """Test cases for chain-based RetrievalManager."""

    @pytest.fixture
    def mock_retrieval_manager(self):
        """Create a mock RetrievalManager for testing."""
        with patch('src.disney.rag.retrieval_manager.chromadb.HttpClient') as mock_chroma_client, \
             patch('src.disney.rag.retrieval_manager.Chroma') as mock_chroma, \
             patch('src.disney.rag.retrieval_manager.HuggingFaceEmbeddings') as mock_embeddings, \
             patch('src.disney.rag.retrieval_manager.ChatOpenAI') as mock_llm:

            # Mock ChromaDB client
            mock_chroma_client.return_value = MagicMock()
            
            # Mock Chroma vector store
            mock_chroma_instance = MagicMock()
            mock_chroma_instance.as_retriever.return_value = MagicMock()
            mock_chroma.return_value = mock_chroma_instance
            
            # Mock embeddings
            mock_embeddings.return_value = MagicMock()
            
            # Mock LLM
            mock_llm.return_value = MagicMock()
            
            # Create RetrievalManager instance
            manager = RetrievalManager()
            manager.rag_chain = MagicMock()
            manager.rag_chain.ainvoke = AsyncMock(return_value="Test answer")
            
            yield manager

    def test_initialization(self, mock_retrieval_manager):
        """Test RetrievalManager initialization."""
        manager = mock_retrieval_manager
        assert manager is not None
        assert manager.collection_name == "disney_reviews"

    @pytest.mark.asyncio
    async def test_query_success(self, mock_retrieval_manager):
        """Test successful query using the chain."""
        manager = mock_retrieval_manager
        manager.rag_chain.ainvoke.return_value = "Space Mountain is highly rated by customers."
        
        result = await manager.query("What do customers say about Space Mountain?")
        
        assert result == "Space Mountain is highly rated by customers."
        manager.rag_chain.ainvoke.assert_called_once_with("What do customers say about Space Mountain?")

    @pytest.mark.asyncio
    async def test_query_error_handling(self, mock_retrieval_manager):
        """Test query error handling."""
        manager = mock_retrieval_manager
        manager.rag_chain.ainvoke.side_effect = Exception("Chain error")
        
        result = await manager.query("Test question")
        
        assert "I apologize, but I encountered an error" in result

    def test_add_documents_success(self, mock_retrieval_manager):
        """Test successful document addition."""
        manager = mock_retrieval_manager
        manager.vectorstore.add_documents.return_value = None
        
        documents = [
            Document(page_content="Test content 1", metadata={"id": "1"}),
            Document(page_content="Test content 2", metadata={"id": "2"})
        ]
        
        result = manager.add_documents(documents)
        
        assert result is True
        manager.vectorstore.add_documents.assert_called_once_with(documents)

    def test_add_documents_error(self, mock_retrieval_manager):
        """Test document addition error handling."""
        manager = mock_retrieval_manager
        manager.vectorstore.add_documents.side_effect = Exception("Add error")
        
        documents = [Document(page_content="Test", metadata={"id": "1"})]
        result = manager.add_documents(documents)
        
        assert result is False

    def test_add_texts_success(self, mock_retrieval_manager):
        """Test successful text addition."""
        manager = mock_retrieval_manager
        manager.vectorstore.add_texts.return_value = None
        
        texts = ["Text 1", "Text 2"]
        metadatas = [{"id": "1"}, {"id": "2"}]
        
        result = manager.add_texts(texts, metadatas)
        
        assert result is True
        manager.vectorstore.add_texts.assert_called_once_with(texts, metadatas)

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
        manager.chroma_client.get_collection.side_effect = Exception("Stats error")
        
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
        manager.vectorstore.similarity_search_with_score.return_value = mock_docs_with_scores
        
        result = manager.search_with_score("test query", k=2)
        
        assert len(result) == 2
        assert result[0][0].page_content == "Test 1"
        assert result[0][1] == 0.1

    def test_delete_collection_success(self, mock_retrieval_manager):
        """Test successful collection deletion."""
        manager = mock_retrieval_manager
        manager.chroma_client.delete_collection.return_value = None
        
        result = manager.delete_collection()
        
        assert result is True
        manager.chroma_client.delete_collection.assert_called_once_with("disney_reviews")

    def test_delete_collection_error(self, mock_retrieval_manager):
        """Test collection deletion error handling."""
        manager = mock_retrieval_manager
        manager.chroma_client.delete_collection.side_effect = Exception("Delete error")
        
        result = manager.delete_collection()
        
        assert result is False

    def test_reset_collection_success(self, mock_retrieval_manager):
        """Test successful collection reset."""
        manager = mock_retrieval_manager
        with patch.object(manager, 'delete_collection', return_value=True) as mock_delete:
            result = manager.reset_collection()
            
            assert result is True
            mock_delete.assert_called_once()

    def test_get_relevant_context_backward_compatibility(self, mock_retrieval_manager):
        """Test backward compatibility for get_relevant_context."""
        manager = mock_retrieval_manager
        
        # Mock retriever
        mock_docs = [
            Document(page_content="Test content 1", metadata={"id": "1"}),
            Document(page_content="Test content 2", metadata={"id": "2"})
        ]
        manager.retriever.get_relevant_documents.return_value = mock_docs
        
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
        with patch('src.disney.rag.retrieval_manager.RetrievalManager') as mock_manager_class:
            mock_instance = MagicMock()
            mock_manager_class.return_value = mock_instance
            
            from src.disney.rag.retrieval_manager import get_retrieval_manager
            
            # Test get_retrieval_manager
            result1 = get_retrieval_manager("localhost", 8000)
            assert result1 == mock_instance
            mock_manager_class.assert_called_with("localhost", 8000)
            
            # Test get_retriever (backward compatibility)
            result2 = get_retrieval_manager("localhost", 8000)
            assert result2 == mock_instance


class TestPromptTemplate:
    """Test cases for prompt template."""

    def test_get_prompt_template(self):
        """Test getting the prompt template."""
        from src.disney.rag.prompt_template import get_prompt_template
        
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
        from src.disney.rag.prompt_template import DISNEY_QA_TEMPLATE
        
        assert "You are an assistant for Disney customer experience questions" in DISNEY_QA_TEMPLATE
        assert "Question: {question}" in DISNEY_QA_TEMPLATE
        assert "Context: {context}" in DISNEY_QA_TEMPLATE
        assert "Answer:" in DISNEY_QA_TEMPLATE