#!/usr/bin/env python3
"""Test script for the Customer Experience Assessment API."""

import asyncio
import httpx
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from disney.shared.logging import setup_logging

logger = setup_logging("api-tester")


async def test_api_endpoints():
    """Test the API endpoints."""
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Test simple health endpoint (from main.py)
            logger.info("Testing simple health endpoint...")
            response = await client.get(f"{base_url}/health")
            print(f"Simple Health Status: {response.status_code}")
            if response.status_code == 200:
                health_data = response.json()
                print(f"Simple Health Response: {json.dumps(health_data, indent=2)}")
            else:
                print(f"Simple Health Error: {response.text}")
            
            # Test comprehensive health endpoint (from routes.py)
            logger.info("Testing comprehensive health endpoint...")
            response = await client.get(f"{base_url}/api/v1/health")
            print(f"Comprehensive Health Status: {response.status_code}")
            if response.status_code == 200:
                health_data = response.json()
                print(f"Comprehensive Health Response: {json.dumps(health_data, indent=2)}")
            else:
                print(f"Comprehensive Health Error: {response.text}")
            
            # Test status endpoint
            logger.info("Testing status endpoint...")
            response = await client.get(f"{base_url}/api/v1/status")
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                status_data = response.json()
                print(f"Status Response: {json.dumps(status_data, indent=2)}")
            else:
                print(f"Status Error: {response.text}")
            
            # Test cache status endpoint
            logger.info("Testing cache status endpoint...")
            response = await client.get(f"{base_url}/api/v1/cache-status")
            print(f"Cache Status Code: {response.status_code}")
            if response.status_code == 200:
                cache_data = response.json()
                print(f"Cache Status Response: {json.dumps(cache_data, indent=2)}")
            else:
                print(f"Cache Status Error: {response.text}")
            
            # Test query endpoint with multiple questions
            test_questions = [
                "What do customers say about Space Mountain?",
                "What are the most common complaints about Disneyland?",
                "How do customers rate the food at Disneyland?",
                "What do people think about the wait times?",
                "What are the best attractions according to reviews?"
            ]
            
            for i, question in enumerate(test_questions, 1):
                logger.info(f"Testing query {i}: {question}")
                query_data = {
                    "question": question,
                    "context_limit": 3,
                    "temperature": 0.7
                }
                
                response = await client.post(
                    f"{base_url}/api/v1/query",
                    json=query_data
                )
                print(f"\n--- Query {i} ---")
                print(f"Question: {question}")
                print(f"Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Answer: {result.get('answer', 'No answer')}")
                    print(f"Confidence: {result.get('confidence', 0):.2f}")
                    print(f"Sources: {len(result.get('sources', []))}")
                    print(f"Processing Time: {result.get('processing_time_ms', 0):.2f}ms")
                    
                    # Show source details
                    if result.get('sources'):
                        print("Source Documents:")
                        for j, source in enumerate(result['sources'][:2], 1):
                            print(f"  {j}. {source.get('excerpt', '')[:100]}...")
                            print(f"     Rating: {source.get('metadata', {}).get('rating', 'N/A')}")
                            print(f"     Relevance: {source.get('relevance_score', 0):.2f}")
                else:
                    print(f"Query Error: {response.text}")
                
                print("-" * 50)
                
        except httpx.ConnectError:
            logger.error("Could not connect to API. Make sure the service is running.")
            print("Error: Could not connect to API. Make sure the service is running.")
            print("\nTo start the API server, run:")
            print("  python -m src.disney.api.main")
            print("  or")
            print("  uvicorn src.disney.api.main:app --host 0.0.0.0 --port 8000")
        except Exception as e:
            logger.error(f"Error testing API: {str(e)}")
            print(f"Error: {str(e)}")


async def test_chromadb_connection():
    """Test ChromaDB connection through the API."""
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Test a simple query to check if ChromaDB has data
            logger.info("Testing ChromaDB connection...")
            query_data = {
                "question": "Disney",
                "context_limit": 1,
                "temperature": 0.7
            }
            
            response = await client.post(
                f"{base_url}/api/v1/query",
                json=query_data
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('sources'):
                    print("✅ ChromaDB connection successful - data found")
                    print(f"Found {len(result['sources'])} source documents")
                else:
                    print("⚠️ ChromaDB connected but no data found")
            else:
                print(f"❌ ChromaDB connection failed: {response.text}")
                
        except Exception as e:
            logger.error(f"Error testing ChromaDB: {str(e)}")
            print(f"Error testing ChromaDB: {str(e)}")


async def test_api_docs():
    """Test API documentation endpoints."""
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Test OpenAPI docs
            logger.info("Testing API documentation...")
            response = await client.get(f"{base_url}/docs")
            print(f"API Docs Status: {response.status_code}")
            if response.status_code == 200:
                print("✅ API documentation is accessible at /docs")
            else:
                print(f"❌ API documentation error: {response.text}")
            
            # Test ReDoc
            response = await client.get(f"{base_url}/redoc")
            print(f"ReDoc Status: {response.status_code}")
            if response.status_code == 200:
                print("✅ ReDoc documentation is accessible at /redoc")
            else:
                print(f"❌ ReDoc documentation error: {response.text}")
                
        except Exception as e:
            logger.error(f"Error testing API docs: {str(e)}")
            print(f"Error testing API docs: {str(e)}")


async def main():
    """Main test function."""
    print("🧪 Testing Disney AI Customer Experience Assessment API")
    print("=" * 60)
    
    print("\n1. Testing API Health and Status...")
    await test_api_endpoints()
    
    print("\n2. Testing ChromaDB Connection...")
    await test_chromadb_connection()
    
    print("\n3. Testing API Documentation...")
    await test_api_docs()
    
    print("\n✅ Testing complete!")
    print("\nTo test manually, you can use:")
    print("curl -X POST 'http://localhost:8000/api/v1/query' \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"question\": \"What do customers say about Space Mountain?\", \"context_limit\": 3}'")
    print("\nTo start the API server:")
    print("  python -m src.disney.api.main")
    print("  or")
    print("  uvicorn src.disney.api.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    asyncio.run(main())