"""Integration tests for the RAG pipeline."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx

from src.disney.rag.retrieval_manager import RetrievalManager
from src.disney.rag.generator import AnswerGenerator
from src.disney.rag.document_processor import DocumentProcessor, DocumentProcessorConfig


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


def test_document_processor():
    """Test document processing functionality."""
    config = DocumentProcessorConfig(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=10,
        max_chunk_size=200
    )
    
    processor = DocumentProcessor(config)
    
    # Test with sample review data
    sample_reviews = [
        {
            "id": "review_1",
            "content": "This is a sample Disney review about Space Mountain. The ride was amazing and the wait time was reasonable.",
            "metadata": {"rating": 5, "branch": "Disneyland"}
        }
    ]
    
    documents = processor.process_reviews_batch(sample_reviews)
    
    assert len(documents) > 0
    assert all(hasattr(doc, 'page_content') for doc in documents)
    assert all(hasattr(doc, 'metadata') for doc in documents)


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
    from src.disney.rag.document_processor import DocumentProcessor, DocumentProcessorConfig
    
    # Test that classes exist and can be referenced
    assert RetrievalManager is not None
    assert AnswerGenerator is not None
    assert DocumentProcessor is not None
    assert DocumentProcessorConfig is not None


def test_document_processor_config():
    """Test DocumentProcessorConfig validation."""
    # Test valid config
    config = DocumentProcessorConfig(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=10,
        max_chunk_size=200
    )
    
    assert config.chunk_size == 100
    assert config.chunk_overlap == 20
    assert config.min_chunk_size == 10
    assert config.max_chunk_size == 200
    
    # Test default values (using actual defaults from the class)
    config_default = DocumentProcessorConfig()
    assert config_default.chunk_size == 1000
    assert config_default.chunk_overlap == 200
    assert config_default.min_chunk_size == 100  # Updated to match actual default
    assert config_default.max_chunk_size == 2000


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
