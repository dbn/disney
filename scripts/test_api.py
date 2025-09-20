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
            # Test health endpoint
            logger.info("Testing health endpoint...")
            response = await client.get(f"{base_url}/health")
            print(f"Health Status: {response.status_code}")
            if response.status_code == 200:
                print(f"Health Response: {response.json()}")
            else:
                print(f"Health Error: {response.text}")
            
            # Test status endpoint
            logger.info("Testing status endpoint...")
            response = await client.get(f"{base_url}/api/v1/status")
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print(f"Status Response: {response.json()}")
            else:
                print(f"Status Error: {response.text}")
            
            # Test query endpoint
            logger.info("Testing query endpoint...")
            query_data = {
                "question": "What do customers say about Space Mountain?",
                "context_limit": 3,
                "temperature": 0.7
            }
            
            response = await client.post(
                f"{base_url}/api/v1/query",
                json=query_data
            )
            print(f"Query Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Answer: {result.get('answer', 'No answer')}")
                print(f"Confidence: {result.get('confidence', 0)}")
                print(f"Sources: {len(result.get('sources', []))}")
                print(f"Processing Time: {result.get('processing_time_ms', 0)}ms")
            else:
                print(f"Query Error: {response.text}")
                
        except httpx.ConnectError:
            logger.error("Could not connect to API. Make sure the service is running.")
            print("Error: Could not connect to API. Make sure the service is running.")
        except Exception as e:
            logger.error(f"Error testing API: {str(e)}")
            print(f"Error: {str(e)}")


async def test_context_service():
    """Test the Context Retrieval Service."""
    base_url = "http://localhost:8001"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Test health endpoint
            logger.info("Testing Context Service health...")
            response = await client.get(f"{base_url}/health")
            print(f"Context Service Health: {response.status_code}")
            if response.status_code == 200:
                print(f"Context Service Response: {response.json()}")
            else:
                print(f"Context Service Error: {response.text}")
            
            # Test search endpoint
            logger.info("Testing search endpoint...")
            search_data = {
                "query": "Space Mountain wait times",
                "n_results": 3
            }
            
            response = await client.post(
                f"{base_url}/api/v1/search",
                json=search_data
            )
            print(f"Search Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Found {len(result.get('results', []))} results")
                for i, doc in enumerate(result.get('results', [])[:2]):
                    print(f"  Result {i+1}: {doc.get('document', '')[:100]}...")
            else:
                print(f"Search Error: {response.text}")
                
        except httpx.ConnectError:
            logger.error("Could not connect to Context Service. Make sure it's running.")
            print("Error: Could not connect to Context Service. Make sure it's running.")
        except Exception as e:
            logger.error(f"Error testing Context Service: {str(e)}")
            print(f"Error: {str(e)}")


async def main():
    """Main test function."""
    print("🧪 Testing Disney AI Customer Experience Assessment API")
    print("=" * 60)
    
    print("\n1. Testing Context Retrieval Service...")
    await test_context_service()
    
    print("\n2. Testing Customer Experience Assessment API...")
    await test_api_endpoints()
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    asyncio.run(main())
