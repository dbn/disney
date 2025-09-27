"""
Query parser component for metadata extraction from user queries.

This module provides LLM-based metadata extraction and query rewriting capabilities
for the Disney customer review analysis system.
"""

from enum import Enum
import logging
from typing import Optional, Dict, Any
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

from ..shared.config import settings
from .prompt_template import get_metadata_extraction_template
from disney.rag import prompt_template

logger = logging.getLogger(__name__)

class BranchEnum(str, Enum):
    HONG_KONG = 'Disneyland_HongKong'
    CALIFORNIA = 'Disneyland_California'
    PARIS = 'Disneyland_Paris'

class QueryFilters(BaseModel):
    """ChromaDB filters extracted from user query."""
    rating: Optional[int] = Field(None, ge=1, le=5, description="Customer rating (1-5)")
    year: Optional[int] = Field(None, ge=2000, le=2030, description="Review year")
    month: Optional[int] = Field(None, ge=1, le=12, description="Review month (1-12)")
    branch: Optional[str] = Field(None, description="Disney site location")
    reviewer_location: Optional[str] = Field(None, description="Reviewer's location")
    review_id: Optional[str] = Field(None, description="Specific review ID")


class QueryParseResult(BaseModel):
    """Structured result from query parsing."""
    search_query: str = Field(..., description="Rewritten semantic query for better search")
    filters: QueryFilters = Field(default_factory=QueryFilters, description="Extracted metadata filters")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in extraction (0.0-1.0)")
    reasoning: str = Field(..., description="Brief explanation of extraction")


class QueryParserConfig(BaseModel):
    """Configuration for query parser."""
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=500, ge=100, le=2000, description="Maximum tokens")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    enable_metadata_extraction: bool = Field(default=True, description="Enable metadata extraction")
    enable_query_rewriting: bool = Field(default=True, description="Enable query rewriting")


class QueryParser:
    """Query parser with LLM-based metadata extraction and structured output."""
    
    def __init__(self, config: Optional[QueryParserConfig] = None):
        """Initialize the query parser.
        
        Args:
            config: Query parser configuration. If None, uses default config.
        """
        self.config = config or QueryParserConfig()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=settings.openai_api_key
        )
        
        # Use PydanticOutputParser for structured output
        self.output_parser = PydanticOutputParser(pydantic_object=QueryParseResult)

        print(f' branches "{BranchEnum.HONG_KONG.value}", "{BranchEnum.CALIFORNIA.value}", "{BranchEnum.PARIS.value}"')

        # Create prompt template
        # self.prompt_template = get_metadata_extraction_template()
        self.prompt_template = PromptTemplate(
            template=prompt_template.METADATA_EXTRACTION_TEMPLATE,
            input_variables=["query"],
            # The PydanticOutputParser automatically provides the formatting instructions
            partial_variables={"format_instructions": self.output_parser.get_format_instructions(), 
            "branch_enum":  f'"{BranchEnum.HONG_KONG.value}", "{BranchEnum.CALIFORNIA.value}", "{BranchEnum.PARIS.value}"'}

        )
        
        # Create the chain with structured output
        self.chain = (
            self.prompt_template 
            | self.llm 
            | self.output_parser
        )
        
        logger.info(f"QueryParser initialized with model: {self.config.llm_model}")
    
    async def parse_query(self, query: str) -> QueryParseResult:
        """Parse user query and extract metadata using structured output.
        
        Args:
            query: User's natural language query
            
        Returns:
            QueryParseResult with extracted metadata and rewritten query
        """
        if not query or not query.strip():
            return QueryParseResult(
                search_query="",
                filters=QueryFilters(),
                confidence=0.0,
                reasoning="Empty query provided"
            )
        
        try:
            # Use structured output parsing
            result = await self.chain.ainvoke({"query": query.strip()})
            
            # Validate confidence threshold
            if result.confidence < self.config.confidence_threshold:
                logger.warning(f"Low confidence parsing: {result.confidence}")
                # Still return result but with warning
            
            logger.debug(f"Parsed query: {query} -> {result.search_query}")
            return result
            
        except ValidationError as e:
            logger.error(f"Pydantic validation error: {e}")
            return self._handle_validation_error(query, str(e))
        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
            return self._handle_parsing_error(query, str(e))
    
    def _build_chromadb_filters(self, filters: QueryFilters) -> Dict[str, Any]:
        """Convert Pydantic filters to ChromaDB where clause format.
        
        Args:
            filters: Pydantic QueryFilters object
            
        Returns:
            Dictionary suitable for ChromaDB where clause
        """
        chromadb_filters = {}
        
        if filters.rating is not None:
            chromadb_filters["rating"] = filters.rating
        if filters.year is not None:
            chromadb_filters["year"] = filters.year
        if filters.month is not None:
            chromadb_filters["month"] = filters.month
        if filters.branch is not None:
            chromadb_filters["branch"] = filters.branch
        if filters.reviewer_location is not None:
            chromadb_filters["reviewer_location"] = filters.reviewer_location
        if filters.review_id is not None:
            chromadb_filters["review_id"] = filters.review_id
            
        return chromadb_filters
    
    def _handle_validation_error(self, query: str, error: str) -> QueryParseResult:
        """Handle Pydantic validation errors.
        
        Args:
            query: Original query
            error: Validation error message
            
        Returns:
            QueryParseResult with error information
        """
        return QueryParseResult(
            search_query=query,
            filters=QueryFilters(),
            confidence=0.0,
            reasoning=f"Validation error: {error}"
        )
    
    def _handle_parsing_error(self, query: str, error: str) -> QueryParseResult:
        """Handle general parsing errors.
        
        Args:
            query: Original query
            error: Error message
            
        Returns:
            QueryParseResult with error information
        """
        return QueryParseResult(
            search_query=query,
            filters=QueryFilters(),
            confidence=0.0,
            reasoning=f"Parsing failed: {error}"
        )


