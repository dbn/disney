#!/usr/bin/env python3
"""Setup script for debug environment."""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("🔧 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.11+")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔧 Checking dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "chromadb",
        "langchain",
        "langchain-community",
        "langchain-openai",
        "pandas",
        "httpx",
        "python-dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies available")
    return True

def create_env_file():
    """Create .env file if it doesn't exist."""
    print("🔧 Checking .env file...")
    
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file exists")
        return True
    
    print("📝 Creating .env file...")
    env_content = """# Disney API Environment Variables
OPENAI_API_KEY=your_openai_api_key_here
CHROMA_HOST=localhost
CHROMA_PORT=8001
LOG_LEVEL=DEBUG

# Optional: Set these if you want to use external ChromaDB
# CHROMA_HOST=localhost
# CHROMA_PORT=8001
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ .env file created")
    print("⚠️  Please update OPENAI_API_KEY in .env file")
    return True

def check_project_structure():
    """Check if project structure is correct."""
    print("🔧 Checking project structure...")
    
    required_paths = [
        "src/disney/api/main.py",
        "src/disney/api/routes.py",
        "src/disney/api/dependencies.py",
        "src/disney/rag/retrieval_manager.py",
        "src/disney/shared/config.py"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not Path(path).exists():
            missing_paths.append(path)
        else:
            print(f"✅ {path}")
    
    if missing_paths:
        print(f"❌ Missing files: {', '.join(missing_paths)}")
        return False
    
    print("✅ Project structure is correct")
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up debug environment...")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_project_structure(),
        check_dependencies(),
        create_env_file()
    ]
    
    if all(checks):
        print("\n🎉 Environment setup complete!")
        print("\nNext steps:")
        print("1. Update OPENAI_API_KEY in .env file")
        print("2. Run: python test_simple.py")
        print("3. If tests pass, run: python debug_api_fixed.py")
    else:
        print("\n❌ Setup failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
