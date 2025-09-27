#!/usr/bin/env python3
"""Fixed debug script for running the API locally with comprehensive error handling."""

import os
import sys
import logging
import traceback
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Set environment variables
os.environ.setdefault("LOG_LEVEL", "DEBUG")
os.environ.setdefault("CHROMA_HOST", "localhost")
os.environ.setdefault("CHROMA_PORT", "8001")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_imports():
    """Test each import step by step."""
    print("🔧 Testing imports step by step...")
    
    try:
        print("  - Testing basic imports...")
        import uvicorn
        from fastapi import FastAPI
        print("  ✅ Basic imports successful")
        
        print("  - Testing shared modules...")
        from disney.shared.config import settings
        print("  ✅ Settings imported")
        
        print("  - Testing logging...")
        from disney.shared.logging import setup_logging
        print("  ✅ Logging imported")
        
        print("  - Testing API models...")
        from disney.api.models import QueryRequest, QueryResponse
        print("  ✅ API models imported")
        
        print("  - Testing dependencies...")
        from disney.api.dependencies import get_retrieval_manager
        print("  ✅ Dependencies imported")
        
        print("  - Testing RetrievalManager...")
        from disney.rag.retrieval_manager import RetrievalManager
        print("  ✅ RetrievalManager imported")
        
        print("  - Testing API routes...")
        from disney.api.routes import router
        print("  ✅ Routes imported")
        
        print("  - Testing API main...")
        from disney.api.main import app
        print("  ✅ App imported")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_retrieval_manager():
    """Test RetrievalManager creation."""
    print("🔧 Testing RetrievalManager creation...")
    
    try:
        from disney.rag.retrieval_manager import RetrievalManager
        
        # Test with in-memory client to avoid ChromaDB connection issues
        import chromadb
        client = chromadb.Client()
        
        manager = RetrievalManager(
            chroma_client=client,
            collection_name="test_collection"
        )
        print("  ✅ RetrievalManager created successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ RetrievalManager creation failed: {e}")
        traceback.print_exc()
        return False

def test_app_creation():
    """Test app creation."""
    print("🔧 Testing app creation...")
    
    try:
        from disney.api.main import app
        print(f"  ✅ App created: {app.title}")
        print(f"  - Version: {app.version}")
        print(f"  - Routes: {len(app.routes)}")
        return True
        
    except Exception as e:
        print(f"  ❌ App creation failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting comprehensive API debug session...")
    print("=" * 50)
    
    # Step 1: Test imports
    if not test_imports():
        print("❌ Import test failed. Exiting.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    
    # Step 2: Test RetrievalManager
    if not test_retrieval_manager():
        print("❌ RetrievalManager test failed. Exiting.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    
    # Step 3: Test app creation
    if not test_app_creation():
        print("❌ App creation test failed. Exiting.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! Starting server...")
    
    # Step 4: Start server
    try:
        import uvicorn
        
        print("🔧 Starting uvicorn server...")
        uvicorn.run(
            "disney.api.main:app",  # ✅ Import string instead of app object
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="debug",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        traceback.print_exc()
