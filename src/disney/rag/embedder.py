"""Text embedding module for RAG pipeline."""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from ..shared.config import settings
from ..shared.logging import setup_logging

logger = setup_logging("rag-embedder")


class TextEmbedder:
    """Text embedding using sentence-transformers."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the embedder.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name or settings.embedding_model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector
        """
        try:
            if not self.model:
                raise Exception("Model not loaded")
            
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self.model.encode(cleaned_text)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            Array of embedding vectors
        """
        try:
            if not self.model:
                raise Exception("Model not loaded")
            
            # Clean and preprocess texts
            cleaned_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(cleaned_texts)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Basic preprocessing
        text = text.strip()
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Truncate if too long (sentence-transformers has limits)
        max_length = 512  # Typical limit for most models
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text truncated to {max_length} characters")
        
        return text
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        if not self.model:
            raise Exception("Model not loaded")
        
        return self.model.get_sentence_embedding_dimension()


# Global embedder instance
_embedder = None


def get_embedder() -> TextEmbedder:
    """Get or create embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedder()
    return _embedder
