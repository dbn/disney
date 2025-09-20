"""Integration tests for the RAG pipeline."""

import pytest
from unittest.mock import patch, AsyncMock
import httpx

from src.disney.rag.retrieval import RAGRetriever
from src.disney.rag.generator import AnswerGenerator


@pytest.mark.asyncio
async def test_rag_pipeline_integration():
    """Test the complete RAG pipeline integration."""
    
    # Mock Context Retrieval Service response
    mock_search_response = {
        "results": [
            {
                "id": "review_1",
                "document": "Space Mountain was amazing! The wait was about 45 minutes but totally worth it.",
                "metadata": {"rating": 5, "branch": "Disneyland"},
                "distance": 0.1,
                "score": 0.9
            },
            {
                "id": "review_2", 
                "document": "Space Mountain had long lines but the ride was incredible. Highly recommend!",
                "metadata": {"rating": 4, "branch": "Disneyland"},
                "distance": 0.15,
                "score": 0.85
            }
        ],
        "query": "What do customers say about Space Mountain wait times?",
        "total_results": 2
    }
    
    # Mock HTTP client
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json.return_value = mock_search_response
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client
        
        # Test RAG Retriever
        retriever = RAGRetriever()
        context_docs = await retriever.get_relevant_context(
            query="What do customers say about Space Mountain wait times?",
            n_results=2
        )
        
        assert len(context_docs) == 2
        assert context_docs[0]["id"] == "review_1"
        assert context_docs[0]["relevance_score"] == 0.9
        assert "Space Mountain was amazing" in context_docs[0]["content"]


def test_answer_generator_with_mock_llm():
    """Test answer generator with mocked LLM."""
    
    # Mock LangChain components
    with patch('src.disney.rag.generator.OpenAI') as mock_openai, \
         patch('src.disney.rag.generator.PromptTemplate') as mock_prompt, \
         patch('src.disney.rag.generator.LLMChain') as mock_chain:
        
        # Mock LLM response
        mock_llm_instance = AsyncMock()
        mock_llm_instance.temperature = 0.7
        mock_openai.return_value = mock_llm_instance
        
        # Mock prompt template
        mock_prompt_instance = AsyncMock()
        mock_prompt.return_value = mock_prompt_instance
        
        # Mock LLM chain
        mock_chain_instance = AsyncMock()
        mock_chain_instance.run.return_value = "Based on customer reviews, Space Mountain has wait times around 45 minutes but customers say it's worth it."
        mock_chain.return_value = mock_chain_instance
        
        # Test generator
        generator = AnswerGenerator(api_key="test-key")
        
        context_docs = [
            {
                "id": "review_1",
                "content": "Space Mountain was amazing! The wait was about 45 minutes but totally worth it.",
                "relevance_score": 0.9
            }
        ]
        
        result = generator.generate_answer(
            question="What do customers say about Space Mountain wait times?",
            context_docs=context_docs,
            temperature=0.7
        )
        
        assert "wait times" in result["answer"].lower()
        assert result["confidence"] > 0
        assert result["context_used"] == 1


def test_context_ranking():
    """Test context document ranking."""
    retriever = RAGRetriever()
    
    context_docs = [
        {"id": "doc1", "content": "Content 1", "relevance_score": 0.7},
        {"id": "doc2", "content": "Content 2", "relevance_score": 0.9},
        {"id": "doc3", "content": "Content 3", "relevance_score": 0.5}
    ]
    
    ranked_docs = retriever.rank_context(context_docs, "test query")
    
    # Should be sorted by relevance score (descending)
    assert ranked_docs[0]["relevance_score"] == 0.9
    assert ranked_docs[1]["relevance_score"] == 0.7
    assert ranked_docs[2]["relevance_score"] == 0.5


def test_context_filtering():
    """Test context document filtering by length."""
    retriever = RAGRetriever()
    
    context_docs = [
        {"id": "doc1", "content": "Short content", "relevance_score": 0.9},
        {"id": "doc2", "content": "A" * 1000, "relevance_score": 0.8},  # 1000 chars
        {"id": "doc3", "content": "B" * 1000, "relevance_score": 0.7},  # 1000 chars
        {"id": "doc4", "content": "C" * 1000, "relevance_score": 0.6},  # 1000 chars
    ]
    
    # Filter with max length of 2500 chars
    filtered_docs = retriever.filter_context(context_docs, max_length=2500)
    
    # Should include first 3 docs (14 + 1000 + 1000 = 2014 chars)
    assert len(filtered_docs) == 3
    assert filtered_docs[0]["id"] == "doc1"  # Highest relevance
    assert filtered_docs[1]["id"] == "doc2"
    assert filtered_docs[2]["id"] == "doc3"
