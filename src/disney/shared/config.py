"""Configuration management for Disney AI services."""

import os
from typing import Optional
from pydantic import  Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # Service URLs
    context_service_url: str = Field(
        default="http://localhost:8001",
        env="CONTEXT_SERVICE_URL"
    )
    
    # API Keys
    openai_api_key: Optional[str] = Field(
        default=None,
        env="OPENAI_API_KEY"
    )
    
    # Database Settings
    chroma_host: str = Field(
        default="localhost",
        env="CHROMA_HOST"
    )
    chroma_port: int = Field(
        default=8000,
        env="CHROMA_PORT"
    )
    
    # Model Settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    max_context_length: int = Field(
        default=4000,
        env="MAX_CONTEXT_LENGTH"
    )
    similarity_threshold: float = Field(
        default=0.7,
        env="SIMILARITY_THRESHOLD"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL"
    )
    
    # Data Pipeline
    data_path: str = Field(
        default="data/DisneylandReviews.csv",
        env="DATA_PATH"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
