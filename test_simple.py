#!/usr/bin/env python3
"""Simple test script to verify imports and basic functionality."""

import os
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Set environment variables
os.environ.setdefault("LOG_LEVEL", "DEBUG")
os.environ.setdefault("CHROMA_HOST", "localhost")
os.environ.setdefault("CHROMA_PORT", "8001")

def test_basic_imports():
    """Test basic imports."""
    print("🔧 Testing basic imports...")
    
    try:
        import uvicorn
        print("✅ uvicorn")
        
        from fastapi import FastAPI
        print("✅ FastAPI")
        
        from disney.shared.config import settings
        print("✅ settings")
        
        from disney.api.main import app
        print("✅ app")
        
        print("🎉 All basic imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_retrieval_manager():
    """Test RetrievalManager with in-memory client."""
    print("\n🔧 Testing RetrievalManager...")
    
    try:
        from disney.rag.retrieval_manager_fixed import RetrievalManager
        import chromadb
        
        # Use in-memory client to avoid connection issues
        client = chromadb.Client()
        
        manager = RetrievalManager(
            chroma_client=client,
            collection_name="test_collection"
        )
        
        print("✅ RetrievalManager created successfully")
        print(f"  - Collection: {manager.collection_name}")
        print(f"  - Chain info: {manager.get_chain_info()}")
        
        return True
        
    except Exception as e:
        print(f"❌ RetrievalManager failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting simple test...")
    
    if test_basic_imports() and test_retrieval_manager():
        print("\n🎉 All tests passed! You can now run the API.")
        print("\nTo start the API, run:")
        print("python debug_api_fixed.py")
    else:
        print("\n❌ Tests failed. Check the errors above.")
        sys.exit(1)
