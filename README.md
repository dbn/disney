# Disney AI Customer Review Analysis

A production-oriented prototype that enables Disney customer experience teams to query customer reviews in natural language using AI-powered RAG (Retrieval-Augmented Generation).

## 🏗️ Architecture

The system consists of several microservices:

- **Customer Experience Assessment Service** - Main FastAPI service for natural language queries
- **Context Retrieval Service** - FastAPI service for vector search and indexing using ChromaDB
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
   # Customer Experience Service
   uv run --extra api python -m src.disney.api.main
   
   # Context Retrieval Service
   uv run --extra context python -m src.disney.context_service.main
   
   # Data Pipeline
   uv run --extra pipeline python scripts/run_pipeline.py
   
   # Jupyter Notebook
   uv run --extra notebook jupyter lab
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

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/test_api/
uv run pytest tests/test_context_service/
uv run pytest tests/test_rag/

# Run with coverage
uv run pytest --cov=src/disney
```

## 🔧 Development

### Dependencies

The project uses `uv` for dependency management with separate groups:

- `api` - Customer Experience Assessment Service
- `context` - Context Retrieval Service  
- `rag` - RAG components (shared)
- `pipeline` - Data pipeline
- `notebook` - Jupyter notebook
- `dev` - Development tools

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
- Context Retrieval API: http://localhost:8001/docs
- Jupyter Notebook: http://localhost:8888

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| customer-experience-api | 8000 | Main API service |
| context-service | 8001 | Vector search service |
| chromadb | 8002 | Vector database |
| jupyter | 8888 | Notebook environment |

## 📁 Project Structure

```
disney/
├── src/disney/           # Main source code
│   ├── api/              # Customer Experience Service
│   ├── context_service/  # Context Retrieval Service
│   ├── rag/              # RAG components
│   ├── pipeline/         # Data pipeline
│   └── shared/           # Shared utilities
├── tests/                # Test suite
├── scripts/              # Utility scripts
├── docker/               # Docker configurations
└── notebooks/            # Jupyter notebooks
```

## 🔑 Environment Variables

See `env.example` for all available configuration options.

## 📝 License

[Add your license here]
Disney customer experience management system
