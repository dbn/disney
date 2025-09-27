"""
Tests for the query parser component.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError

from disney.rag.query_parser import QueryParser, QueryParserConfig, QueryFilters, QueryParseResult


class TestQueryFilters:
    """Test QueryFilters Pydantic model."""
    
    def test_query_filters_valid_data(self):
        """Test QueryFilters with valid data."""
        filters = QueryFilters(
            rating=5,
            year=2023,
            month=6,
            branch="Disneyland",
            reviewer_location="USA",
            review_id="12345"
        )
        
        assert filters.rating == 5
        assert filters.year == 2023
        assert filters.month == 6
        assert filters.branch == "Disneyland"
        assert filters.reviewer_location == "USA"
        assert filters.review_id == "12345"
    
    def test_query_filters_partial_data(self):
        """Test QueryFilters with partial data."""
        filters = QueryFilters(rating=4, year=2022)
        
        assert filters.rating == 4
        assert filters.year == 2022
        assert filters.month is None
        assert filters.branch is None
        assert filters.reviewer_location is None
        assert filters.review_id is None
    
    def test_query_filters_validation_errors(self):
        """Test QueryFilters validation errors."""
        # Invalid rating
        with pytest.raises(ValidationError):
            QueryFilters(rating=6)
        
        with pytest.raises(ValidationError):
            QueryFilters(rating=0)
        
        # Invalid year
        with pytest.raises(ValidationError):
            QueryFilters(year=1999)
        
        with pytest.raises(ValidationError):
            QueryFilters(year=2031)
        
        # Invalid month
        with pytest.raises(ValidationError):
            QueryFilters(month=0)
        
        with pytest.raises(ValidationError):
            QueryFilters(month=13)


class TestQueryParseResult:
    """Test QueryParseResult Pydantic model."""
    
    def test_query_parse_result_valid(self):
        """Test QueryParseResult with valid data."""
        filters = QueryFilters(rating=5, year=2023)
        result = QueryParseResult(
            search_query="Disney reviews",
            filters=filters,
            confidence=0.9,
            reasoning="Extracted rating and year"
        )
        
        assert result.search_query == "Disney reviews"
        assert result.filters.rating == 5
        assert result.filters.year == 2023
        assert result.confidence == 0.9
        assert result.reasoning == "Extracted rating and year"
    
    def test_query_parse_result_validation_errors(self):
        """Test QueryParseResult validation errors."""
        # Invalid confidence
        with pytest.raises(ValidationError):
            QueryParseResult(
                search_query="test",
                filters=QueryFilters(),
                confidence=1.5,  # Invalid confidence
                reasoning="test"
            )
        
        with pytest.raises(ValidationError):
            QueryParseResult(
                search_query="test",
                filters=QueryFilters(),
                confidence=-0.1,  # Invalid confidence
                reasoning="test"
            )


class TestQueryParserConfig:
    """Test QueryParserConfig Pydantic model."""
    
    def test_query_parser_config_defaults(self):
        """Test QueryParserConfig with default values."""
        config = QueryParserConfig()
        
        assert config.llm_model == "gpt-4o-mini"
        assert config.temperature == 0.1
        assert config.max_tokens == 500
        assert config.confidence_threshold == 0.7
        assert config.enable_metadata_extraction is True
        assert config.enable_query_rewriting is True
    
    def test_query_parser_config_custom(self):
        """Test QueryParserConfig with custom values."""
        config = QueryParserConfig(
            llm_model="gpt-4",
            temperature=0.2,
            max_tokens=1000,
            confidence_threshold=0.8,
            enable_metadata_extraction=False
        )
        
        assert config.llm_model == "gpt-4"
        assert config.temperature == 0.2
        assert config.max_tokens == 1000
        assert config.confidence_threshold == 0.8
        assert config.enable_metadata_extraction is False


class TestQueryParser:
    """Test QueryParser class."""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        mock_llm = AsyncMock()
        return mock_llm
    
    @pytest.fixture
    def mock_chain(self):
        """Mock chain for testing."""
        mock_chain = AsyncMock()
        return mock_chain
    
    @pytest.fixture
    def query_parser(self, mock_llm, mock_chain):
        """Create QueryParser with mocked dependencies."""
        with patch('disney.rag.query_parser.ChatOpenAI', return_value=mock_llm), \
             patch('disney.rag.query_parser.get_metadata_extraction_template'), \
             patch('disney.rag.query_parser.PydanticOutputParser'):
            
            config = QueryParserConfig()
            parser = QueryParser(config)
            # Mock the chain after initialization
            parser.chain = mock_chain
            return parser
    
    def test_query_parser_initialization(self, query_parser):
        """Test QueryParser initialization."""
        assert query_parser.config is not None
        assert query_parser.config.llm_model == "gpt-4o-mini"
    
    @pytest.mark.asyncio
    async def test_parse_query_success(self, query_parser, mock_chain):
        """Test successful query parsing."""
        # Mock successful parsing result
        expected_result = QueryParseResult(
            search_query="Disney reviews about Space Mountain",
            filters=QueryFilters(rating=5, branch="Disneyland"),
            confidence=0.9,
            reasoning="Extracted 5-star rating and Disneyland location"
        )
        mock_chain.ainvoke.return_value = expected_result
        
        result = await query_parser.parse_query("Show me 5-star reviews from Disneyland about Space Mountain")
        
        assert result.search_query == "Disney reviews about Space Mountain"
        assert result.filters.rating == 5
        assert result.filters.branch == "Disneyland"
        assert result.confidence == 0.9
        mock_chain.ainvoke.assert_called_once_with({"query": "Show me 5-star reviews from Disneyland about Space Mountain"})
    
    @pytest.mark.asyncio
    async def test_parse_query_empty(self, query_parser):
        """Test parsing empty query."""
        result = await query_parser.parse_query("")
        
        assert result.search_query == ""
        assert result.filters.rating is None
        assert result.confidence == 0.0
        assert result.reasoning == "Empty query provided"
    
    @pytest.mark.asyncio
    async def test_parse_query_whitespace(self, query_parser):
        """Test parsing whitespace-only query."""
        result = await query_parser.parse_query("   ")
        
        assert result.search_query == ""
        assert result.filters.rating is None
        assert result.confidence == 0.0
        assert result.reasoning == "Empty query provided"
    
    @pytest.mark.asyncio
    async def test_parse_query_validation_error(self, query_parser, mock_chain):
        """Test parsing with validation error."""
        from pydantic import ValidationError
        
        # Create a proper ValidationError
        try:
            QueryParseResult(
                search_query="test",
                filters=QueryFilters(),
                confidence=1.5,  # Invalid confidence
                reasoning="test"
            )
        except ValidationError as e:
            mock_chain.ainvoke.side_effect = e
        
        result = await query_parser.parse_query("test query")
        
        assert result.search_query == "test query"
        assert result.confidence == 0.0
        assert "Validation error" in result.reasoning
    
    @pytest.mark.asyncio
    async def test_parse_query_general_error(self, query_parser, mock_chain):
        """Test parsing with general error."""
        mock_chain.ainvoke.side_effect = Exception("LLM error")
        
        result = await query_parser.parse_query("test query")
        
        assert result.search_query == "test query"
        assert result.confidence == 0.0
        assert "Parsing failed" in result.reasoning
    
    def test_build_chromadb_filters(self, query_parser):
        """Test building ChromaDB filters from QueryFilters."""
        filters = QueryFilters(
            rating=5,
            year=2023,
            month=6,
            branch="Disneyland",
            reviewer_location="USA",
            review_id="12345"
        )
        
        chromadb_filters = query_parser._build_chromadb_filters(filters)
        
        expected = {
            "$and": [
                {"rating": 5},
                {"year": 2023},
                {"month": 6},
                {"branch": "Disneyland"},
                {"reviewer_location": "USA"},
                {"review_id": "12345"}
            ]
        }
        assert chromadb_filters == expected
    
    def test_build_chromadb_filters_partial(self, query_parser):
        """Test building ChromaDB filters with partial data."""
        filters = QueryFilters(rating=4, year=2022)
        
        chromadb_filters = query_parser._build_chromadb_filters(filters)
        
        expected = {
            "$and": [
                {"rating": 4},
                {"year": 2022}
            ]
        }
        assert chromadb_filters == expected
    
    def test_build_chromadb_filters_empty(self, query_parser):
        """Test building ChromaDB filters with empty data."""
        filters = QueryFilters()
        
        chromadb_filters = query_parser._build_chromadb_filters(filters)
        
        assert chromadb_filters == {}
    
    def test_handle_validation_error(self, query_parser):
        """Test handling validation errors."""
        result = query_parser._handle_validation_error("test query", "validation error")
        
        assert result.search_query == "test query"
        assert result.confidence == 0.0
        assert "Validation error" in result.reasoning
    
    def test_handle_parsing_error(self, query_parser):
        """Test handling parsing errors."""
        result = query_parser._handle_parsing_error("test query", "parsing error")
        
        assert result.search_query == "test query"
        assert result.confidence == 0.0
        assert "Parsing failed" in result.reasoning


class TestQueryParserIntegration:
    """Integration tests for QueryParser."""
    
    @pytest.mark.asyncio
    async def test_query_parser_with_real_llm(self):
        """Test QueryParser with real LLM (requires OpenAI API key)."""
        # Skip if no API key or if using test key
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "test-key":
            pytest.skip("OpenAI API key not available or using test key")
        
        config = QueryParserConfig(
            llm_model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=200
        )
        
        parser = QueryParser(config)
        
        # Test simple query
        result = await parser.parse_query("Show me 5-star reviews from Disneyland")
        
        assert result.search_query is not None
        assert result.confidence > 0
        assert result.reasoning is not None
        
        # Should extract rating and branch
        if result.filters.rating is not None:
            assert result.filters.rating == 5
        if result.filters.branch is not None:
            assert result.filters.branch == "Disneyland"
    
    @pytest.mark.asyncio
    async def test_query_parser_complex_query(self):
        """Test QueryParser with complex query."""
        # Skip if no API key or if using test key
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "test-key":
            pytest.skip("OpenAI API key not available or using test key")
        
        config = QueryParserConfig()
        parser = QueryParser(config)
        
        # Test complex query with multiple metadata
        result = await parser.parse_query(
            "What do customers say about Space Mountain at Disney World in December 2023? Show me reviews from US visitors."
        )
        
        assert result.search_query is not None
        assert result.confidence > 0
        
        # Should extract multiple metadata fields
        if result.filters.year is not None:
            assert result.filters.year == 2023
        if result.filters.month is not None:
            assert result.filters.month == 12
        if result.filters.branch is not None:
            assert result.filters.branch == "Disney World"
        if result.filters.reviewer_location is not None:
            assert "US" in result.filters.reviewer_location or "USA" in result.filters.reviewer_location
