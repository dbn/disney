"""Integration tests for the RAG pipeline."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx

from src.disney.rag.retrieval_manager import RetrievalManager
from src.disney.rag.generator import AnswerGenerator


def test_answer_generator_with_mock_llm():
    """Test answer generator with mocked LLM."""
    # Mock settings to avoid API key requirement
    with patch('src.disney.rag.generator.settings') as mock_settings:
        mock_settings.openai_api_key = "test-key"
        mock_settings.model_name = "gpt-3.5-turbo"
        
        # Mock LangChain components
        with patch('src.disney.rag.generator.ChatOpenAI') as mock_openai, \
             patch('src.disney.rag.generator.PromptTemplate') as mock_prompt, \
             patch('src.disney.rag.generator.LLMChain') as mock_chain:
            
            # Setup mocks
            mock_llm = MagicMock()
            mock_openai.return_value = mock_llm
            
            mock_prompt_template = MagicMock()
            mock_prompt.from_template.return_value = mock_prompt_template
            
            mock_chain_instance = MagicMock()
            mock_chain_instance.run.return_value = "Space Mountain is highly rated by customers with an average rating of 4.5/5. Most customers mention the wait times are reasonable and the ride is worth it."
            mock_chain.return_value = mock_chain_instance
            
            # Test Answer Generator
            generator = AnswerGenerator()
            
            context_docs = [
                {
                    "id": "review_1",
                    "content": "Space Mountain was amazing!",
                    "metadata": {"rating": 5}
                }
            ]
            
            result = generator.generate_answer(
                question="What do customers say about Space Mountain?",
                context_docs=context_docs,
                temperature=0.7
            )
            
            assert "answer" in result
            assert "confidence" in result
            assert result["confidence"] > 0.0




def test_retrieval_manager_initialization():
    """Test RetrievalManager initialization with mocked ChromaDB."""
    with patch('src.disney.rag.retrieval_manager.chromadb.HttpClient') as mock_chroma_client, \
         patch('src.disney.rag.retrieval_manager.Chroma') as mock_chroma, \
         patch('src.disney.rag.retrieval_manager.HuggingFaceEmbeddings') as mock_embeddings:
        
        # Mock ChromaDB client
        mock_client_instance = MagicMock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock Chroma vector store
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance
        
        # Mock embeddings
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Test RetrievalManager initialization
        vector_manager = RetrievalManager()
        
        # Verify that the client was created
        mock_chroma_client.assert_called_once()
        mock_chroma.assert_called_once()
        mock_embeddings.assert_called_once()
        
        # Verify basic properties
        assert vector_manager.chroma_host == "localhost"
        assert vector_manager.chroma_port == 8000
        assert vector_manager.collection_name == "disney_reviews"


def test_rag_components_integration():
    """Test that RAG components can be imported and initialized."""
    # Test that we can import all components
    from src.disney.rag.retrieval_manager import RetrievalManager
    from src.disney.rag.generator import AnswerGenerator
    
    # Test that classes exist and can be referenced
    assert RetrievalManager is not None
    assert AnswerGenerator is not None




def test_answer_generator_config():
    """Test AnswerGenerator configuration."""
    with patch('src.disney.rag.generator.settings') as mock_settings:
        mock_settings.openai_api_key = "test-key"
        mock_settings.model_name = "gpt-3.5-turbo"
        
        # Test that we can create an instance with custom parameters
        generator = AnswerGenerator(api_key="custom-key", model_name="gpt-4")
        
        # Verify the generator was created
        assert generator is not None
        assert generator.api_key == "custom-key"
        assert generator.model_name == "gpt-4"


def test_retrieval_manager_config():
    """Test RetrievalManager configuration."""
    with patch('src.disney.rag.retrieval_manager.chromadb.HttpClient') as mock_chroma_client, \
         patch('src.disney.rag.retrieval_manager.Chroma') as mock_chroma, \
         patch('src.disney.rag.retrieval_manager.HuggingFaceEmbeddings') as mock_embeddings:
        
        # Mock all dependencies
        mock_chroma_client.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()
        mock_embeddings.return_value = MagicMock()
        
        # Test with custom configuration (using actual constructor parameters)
        vector_manager = RetrievalManager(
            chroma_host="custom-host",
            chroma_port=9000
        )
        
        # Verify configuration
        assert vector_manager.chroma_host == "custom-host"
        assert vector_manager.chroma_port == 9000
        assert vector_manager.collection_name == "disney_reviews"  # This is set internally
