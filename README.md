# Disney AI Customer Review Analysis

A production-oriented prototype that enables Disney customer experience teams to query customer reviews in natural language using AI-powered RAG (Retrieval-Augmented Generation).

## 🏗️ Architecture

The system consists of several microservices:

- **Customer Experience Assessment Service** - Main FastAPI service for natural language queries with integrated RAG pipeline
- **ChromaDB** - Standalone vector database for storing and retrieving Disney review embeddings
- **Data Pipeline** - Periodic ingestion of Disney reviews into the vector database
- **Jupyter Notebook** - Data exploration and experimentation environment

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose

### Setup

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd disney
   python scripts/setup_dev.py
   ```

2. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Start services with Docker:**
   ```bash
   docker-compose up -d
   ```

4. **Or run services individually:**
   ```bash
   # Start ChromaDB (required first)
   docker run -d --name chromadb -p 8000:8000 chromadb/chroma:latest
   
   # Customer Experience Service
   uv run --extra api python -m src.disney.api.main
   
   # Data Pipeline
   uv run --extra pipeline python scripts/run_pipeline.py
   
   # Jupyter Notebook
   uv run --extra notebook jupyter lab
   ```

## 🤖 RAG Implementation

The system implements a complete RAG (Retrieval-Augmented Generation) pipeline using LangChain's chain-based approach:

### **Query Processing Flow:**
1. **Question Input** - User submits natural language question via API
2. **RAG Chain Processing** - LangChain chain handles retrieval and generation in one step
3. **Context Retrieval** - System searches ChromaDB for relevant reviews using embeddings
4. **Answer Generation** - LLM generates response using retrieved context and prompt template
5. **Response Formatting** - Structured response with sources and confidence scores

### **Key Components:**

#### **RetrievalManager** (`src/disney/rag/retrieval_manager.py`)
- **Purpose**: Central component for all retrieval and RAG operations
- **Features**: 
  - LangChain chain-based query processing
  - ChromaDB integration for vector storage
  - HuggingFace embeddings for text similarity
  - OpenAI LLM for answer generation
  - Async query processing with `rag_chain.ainvoke()`

#### **Document Processing** (`src/disney/rag/document_processor.py`)
- **Purpose**: Process and chunk Disney review data
- **Features**:
  - CSV loading and preprocessing
  - Text chunking with configurable parameters
  - Metadata extraction and enrichment
  - Progress tracking with tqdm

#### **Prompt Template** (`src/disney/rag/prompt_template.py`)
- **Purpose**: Centralized prompt management for RAG queries
- **Features**:
  - Disney-specific prompt template
  - Configurable via environment variables
  - Optimized for customer review analysis

### **API Endpoints:**
- `POST /api/v1/query` - Submit questions and get AI-generated answers
- `GET /api/v1/health` - Service health and dependency status
- `GET /api/v1/status` - Detailed service metrics

### **Example Query:**
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What do customers say about Space Mountain wait times?",
    "context_limit": 5,
    "temperature": 0.7
  }'
```

### **Response Format:**
```json
{
  "answer": "Based on customer reviews, Space Mountain wait times...",
  "sources": [
    {
      "review_id": "123",
      "relevance_score": 0.95,
      "excerpt": "The wait was about 45 minutes but totally worth it...",
      "metadata": {"rating": 5, "branch": "Disneyland"}
    }
  ],
  "confidence": 0.87,
  "processing_time_ms": 1250
}
```

## 📊 Data Pipeline

The data pipeline processes Disney reviews and indexes them into the vector database:

```bash
# Run data migration and validation
uv run python scripts/migrate_data.py validate data/DisneylandReviews.csv
uv run python scripts/migrate_data.py analyze data/DisneylandReviews.csv

# Run the pipeline
uv run --extra pipeline python scripts/run_pipeline.py
```

## 🧪 Testing

The project includes comprehensive testing across multiple layers:

### **Test Categories:**
- **Unit Tests**: Individual component testing with mocks
- **Integration Tests**: End-to-end testing with mocked external services
- **API Tests**: FastAPI endpoint testing with request/response validation
- **RAG Tests**: RetrievalManager and chain-based processing tests

### **Running Tests:**
```bash
# Run all tests
uv run --extra dev pytest

# Run specific test categories
uv run --extra dev pytest tests/test_api/          # API endpoint tests
uv run --extra dev pytest tests/test_rag/          # RAG component tests
uv run --extra dev pytest tests/test_integration.py # Integration tests

# Run with coverage
uv run --extra dev pytest --cov=src/disney

# Test API endpoints manually
uv run --extra api python scripts/test_api.py
```

### **Test Coverage:**
- **RetrievalManager**: Chain-based query processing, document management, collection operations
- **API Routes**: Query processing, health checks, error handling
- **Document Processing**: CSV loading, chunking, metadata extraction
- **Integration**: End-to-end ingestion and retrieval workflows

## 🔧 Development

### Dependencies

The project uses `uv` for dependency management with separate groups:

- `api` - Customer Experience Assessment Service (includes ChromaDB integration)
- `rag` - RAG components (RetrievalManager, document processing, embeddings)
- `pipeline` - Data pipeline
- `notebook` - Jupyter notebook
- `dev` - Development tools

### uv Commands

#### **Installation & Setup**
```bash
# Install all dependencies
uv sync --all-extras

# Install specific service dependencies
uv sync --extra api          # Customer Experience Service
uv sync --extra rag          # RAG components
uv sync --extra pipeline     # Data pipeline
uv sync --extra notebook     # Jupyter notebook
uv sync --extra dev          # Development tools

# Install with frozen lockfile (production)
uv sync --frozen

# Add new dependency to specific group
uv add --extra api httpx
uv add --extra dev pytest-cov
```

#### **Running Services**
```bash
# Customer Experience Assessment Service
uv run --extra api python -m src.disney.api.main

# Data Pipeline
uv run --extra pipeline python scripts/run_pipeline.py

# Jupyter Notebook
uv run --extra notebook jupyter lab

# Run with specific Python version
uv run --python 3.11 --extra api python -m src.disney.api.main
```

#### **Development Workflow**
```bash
# Run tests
uv run pytest
uv run pytest tests/test_api/
uv run pytest --cov=src/disney

# Code formatting
uv run black .
uv run isort .

# Linting
uv run flake8 src/
uv run mypy src/

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files

# Data migration and validation
uv run python scripts/migrate_data.py validate data/DisneylandReviews.csv
uv run python scripts/migrate_data.py analyze data/DisneylandReviews.csv
```

#### **Dependency Management**
```bash
# Show installed packages
uv pip list

# Show dependency tree
uv pip show --tree

# Update dependencies
uv sync --upgrade

# Update specific dependency
uv add --upgrade fastapi

# Remove dependency
uv remove package-name

# Show outdated packages
uv pip list --outdated
```

#### **Environment Management**
```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Deactivate virtual environment
deactivate

# Show environment info
uv pip show --verbose

# Export requirements
uv pip freeze > requirements.txt
```

#### **Project Management**
```bash
# Initialize new project (if starting fresh)
uv init

# Add project dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Lock dependencies
uv lock

# Sync with lockfile
uv sync --frozen

# Show project info
uv show
```

### Code Quality

```bash
# Format code
uv run black .
uv run isort .

# Lint code
uv run flake8 src/
uv run mypy src/

# Run pre-commit hooks
uv run pre-commit run --all-files
```

## 📚 API Documentation

Once services are running, visit:

- Customer Experience API: http://localhost:8000/docs
- Jupyter Notebook: http://localhost:8888

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| customer-experience-api | 8000 | Main API service with integrated RAG |
| chromadb | 8000 | Vector database (standalone) |
| data-pipeline | - | Data ingestion service |
| jupyter | 8888 | Notebook environment |

## 💾 Persistent Storage

ChromaDB data is persisted in the `./chroma/` directory:
- Vector embeddings and metadata are stored locally
- Data persists across container restarts
- Directory is automatically created on first run
- Add `chroma/` to `.gitignore` to avoid committing data files

## 📁 Project Structure

```
disney/
├── src/disney/           # Main source code
│   ├── api/              # Customer Experience Service
│   ├── rag/              # RAG components (RetrievalManager, document processing)
│   ├── pipeline/         # Data pipeline
│   └── shared/           # Shared utilities
├── tests/                # Test suite
│   ├── test_api/         # API tests
│   ├── test_rag/         # RAG component tests
│   └── test_integration.py # Integration tests
├── scripts/              # Utility scripts
├── docker/               # Docker configurations
├── notebooks/            # Jupyter notebooks
├── chroma/               # ChromaDB persistent storage (auto-created)
└── data/                 # Sample data files
```

## 🔑 Environment Variables

See `env.example` for all available configuration options.

## 🚀 Recent Updates

### **Architecture Simplification:**
- **Removed**: Context Retrieval Service (simplified architecture)
- **Added**: Direct ChromaDB integration in main API service
- **Improved**: LangChain chain-based RAG implementation

### **Component Updates:**
- **Renamed**: `VectorStoreManager` → `RetrievalManager` (better reflects purpose)
- **Enhanced**: Async query processing with `rag_chain.ainvoke()`
- **Added**: Persistent ChromaDB storage in `./chroma/` directory
- **Improved**: Comprehensive test coverage across all components

### **Performance Improvements:**
- **Faster**: Chain-based processing reduces overhead
- **Reliable**: Persistent storage prevents data loss
- **Scalable**: Direct ChromaDB integration supports higher throughput

## 📝 License

[Add your license here]