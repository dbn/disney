#!/usr/bin/env python3
"""Development environment setup script."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def main():
    """Set up development environment."""
    print("🚀 Setting up Disney AI development environment...")
    
    # Check if uv is installed
    if not run_command("uv --version", "Checking uv installation"):
        print("❌ uv is not installed. Please install uv first:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    
    # Install all dependencies
    if not run_command("uv sync --all-extras", "Installing all dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Install pre-commit hooks
    if not run_command("uv run pre-commit install", "Installing pre-commit hooks"):
        print("⚠️  Pre-commit hooks installation failed (optional)")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file from template...")
        env_content = """# Disney AI Configuration

# Service URLs
CONTEXT_SERVICE_URL=http://localhost:8001

# API Keys
OPENAI_API_KEY=your_openai_api_key_here

# Database Settings
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Model Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_CONTEXT_LENGTH=4000
SIMILARITY_THRESHOLD=0.7

# Logging
LOG_LEVEL=INFO

# Data Pipeline
DATA_PATH=data/DisneylandReviews.csv
"""
        env_file.write_text(env_content)
        print("✅ Created .env file. Please update with your actual values.")
    
    print("\n🎉 Development environment setup complete!")
    print("\nNext steps:")
    print("1. Update .env file with your API keys and configuration")
    print("2. Start services with: docker-compose up -d")
    print("3. Run tests with: uv run pytest")
    print("4. Start Jupyter with: uv run --extra notebook jupyter lab")


if __name__ == "__main__":
    main()
