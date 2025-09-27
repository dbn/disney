#!/usr/bin/env python3
"""
Query Parser Demo

This script demonstrates the query parser functionality for extracting metadata
from natural language queries and converting them to ChromaDB filters.
"""

import asyncio
import os
from disney.rag.query_parser import QueryParser, QueryParserConfig


async def demo_query_parser():
    """Demonstrate query parser functionality."""
    print("🔍 Query Parser Demo")
    print("=" * 50)
    
    # Initialize query parser
    config = QueryParserConfig(
        llm_model="gpt-4o-mini",
        temperature=0.1,
        confidence_threshold=0.7
    )
    
    parser = QueryParser(config)
    
    # Example queries
    test_queries = [
        # "Show me 5-star reviews from Disneyland in June 2023",
        # "What do customers say about Space Mountain at Disney World?",
        # "Find reviews about food from US visitors in December",
        # "Show me recent complaints about wait times",
        # "What do people think about the new attraction in March 2024?",
        # "Find review 12345",
        # "Show me 4-star reviews from Disneyland-HongKong in summer",
        "What do customers from Genmrany say in general?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 40)
        
        try:
            # Parse the query
            result = await parser.parse_query(query)
            
            print(f"🔍 Search Query: {result.search_query}")
            print(f"🎯 Confidence: {result.confidence:.2f}")
            print(f"💭 Reasoning: {result.reasoning}")
            
            # Show extracted filters
            filters = result.filters
            print("📊 Extracted Metadata:")
            if filters.rating:
                print(f"  ⭐ Rating: {filters.rating}")
            if filters.year:
                print(f"  📅 Year: {filters.year}")
            if filters.month:
                print(f"  📆 Month: {filters.month}")
            if filters.branch:
                print(f"  🏰 Branch: {filters.branch}")
            if filters.reviewer_location:
                print(f"  🌍 Location: {filters.reviewer_location}")
            if filters.review_id:
                print(f"  🆔 Review ID: {filters.review_id}")
            
            # Show ChromaDB filters
            chromadb_filters = parser._build_chromadb_filters(filters)
            if chromadb_filters:
                print(f"🗃️  ChromaDB Filters: {chromadb_filters}")
            else:
                print("🗃️  ChromaDB Filters: None (no metadata extracted)")
                
        except Exception as e:
            print(f"❌ Error parsing query: {e}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("   The demo will use mocked responses for testing.")
        print("   Set OPENAI_API_KEY to test with real LLM responses.")
        print()
    
    # Run the demo
    asyncio.run(demo_query_parser())
