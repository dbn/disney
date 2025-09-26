"""Configuration management for Disney AI services."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # ChromaDB Settings
    chroma_host: str = Field(
        default="localhost",
        env="CHROMA_HOST"
    )
    
    chroma_port: int = Field(
        default=8000,
        env="CHROMA_PORT"
    )

    # API Keys
    openai_api_key: Optional[str] = Field(
        default=None,
        env="OPENAI_API_KEY"
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
    
    # Chain Configuration
    retriever_k: int = Field(
        default=5,
        env="RETRIEVER_K"
    )
    retriever_score_threshold: float = Field(
        default=0.7,
        env="RETRIEVER_SCORE_THRESHOLD"
    )
    retriever_search_type: str = Field(
        default="similarity",
        env="RETRIEVER_SEARCH_TYPE"
    )
    llm_temperature: float = Field(
        default=0.7,
        env="LLM_TEMPERATURE"
    )
    llm_max_tokens: int = Field(
        default=1000,
        env="LLM_MAX_TOKENS"
    )
    llm_model: str = Field(
        default="4o-mini",
        env="LLM_MODEL"
    )
    enable_streaming: bool = Field(
        default=False,
        env="ENABLE_STREAMING"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()