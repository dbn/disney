"""Data ingestion logic for Disney reviews pipeline using RAG-based processing."""

import asyncio
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from ..shared.config import settings
from ..shared.logging import setup_logging
from ..rag.retrieval_manager import get_retrieval_manager

logger = setup_logging("data-pipeline")


class DataIngester:
    """Data ingestion component for processing Disney reviews using RAG pipeline."""

    def __init__(self, chroma_host: Optional[str] = None, chroma_port: Optional[int] = None):
        """Initialize the data ingester.

        Args:
            chroma_host: ChromaDB host
            chroma_port: ChromaDB port
        """
        self.chroma_host = chroma_host or settings.chroma_host
        self.chroma_port = chroma_port or settings.chroma_port
        self.data_path = settings.data_path
        self.retrieval_manager = None

    def load_reviews_data(self) -> pd.DataFrame:
        """Load Disney reviews data from CSV file.

        Returns:
            DataFrame with reviews data
        """
        try:
            logger.info(f"Loading reviews data from: {self.data_path}")

            # Load CSV data with latin-1 encoding to handle special characters
            df = pd.read_csv(self.data_path, encoding="latin-1")

            logger.info(f"Loaded {len(df)} reviews")
            return df

        except Exception as e:
            logger.error(f"Error loading reviews data: {str(e)}")
            raise

    def preprocess_reviews(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Preprocess reviews data for indexing.

        Args:
            df: Reviews DataFrame

        Returns:
            List of preprocessed review dictionaries
        """
        try:
            documents = []
            
            # Create progress bar for preprocessing
            with tqdm(total=len(df), desc="Preprocessing reviews", unit="review") as pbar:
                for idx, row in df.iterrows():
                    # Extract review text and clean it
                    review_text = str(row.get('Review_Text', '')).strip()
                    if not review_text or review_text == 'nan':
                        pbar.update(1)
                        continue
                    
                    # Extract metadata
                    rating = row.get('Rating', 0)
                    year_month = str(row.get('Year_Month', ''))
                    year = year_month.split('-')[0] if year_month and year_month != 'nan' else None
                    branch = str(row.get('Branch', 'Unknown'))
                    
                    # Create document
                    doc = {
                        'id': f"review_{idx}",
                        'content': review_text,
                        'metadata': {
                            'rating': int(rating) if pd.notna(rating) else 0,
                            'year': year,
                            'branch': branch,
                            'original_index': idx
                        }
                    }
                    
                    documents.append(doc)
                    pbar.update(1)
            
            logger.info(f"Preprocessed {len(documents)} review documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error preprocessing reviews: {str(e)}")
            raise

    async def index_documents(
        self, 
        documents: List[Dict[str, Any]], 
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Index documents using direct ChromaDB integration.

        Args:
            documents: List of document dictionaries
            batch_size: Batch size for processing

        Returns:
            Dictionary with indexing results
        """
        try:
            logger.info(f"Indexing {len(documents)} documents in batches of {batch_size}")
            
            # Initialize vector store manager if not already done
            if not self.retrieval_manager:
                self.retrieval_manager = get_retrieval_manager(self.chroma_host, self.chroma_port)
            
            # Convert documents to LangChain format
            from langchain.schema import Document
            langchain_docs = []
            
            # Create progress bar for document conversion
            with tqdm(total=len(documents), desc="Converting documents", unit="doc") as pbar:
                for doc in documents:
                    langchain_doc = Document(
                        page_content=doc['content'],
                        metadata=doc['metadata']
                    )
                    langchain_docs.append(langchain_doc)
                    pbar.update(1)
            
            # Add documents to ChromaDB with progress bar
            with tqdm(total=len(langchain_docs), desc="Indexing to ChromaDB", unit="doc") as pbar:
                # Process in batches for better progress tracking
                for i in range(0, len(langchain_docs), batch_size):
                    batch = langchain_docs[i:i + batch_size]
                    
                    # Add batch to vector store
                    success = await self.retrieval_manager.add_documents(batch)
                    
                    if not success:
                        raise Exception(f"Failed to add batch {i//batch_size + 1} to ChromaDB")
                    
                    pbar.update(len(batch))
            
            result = {
                'success': True,
                'total_documents': len(documents),
                'indexed_documents': len(documents),
                'batch_size': batch_size
            }
            logger.info(f"Successfully indexed {len(documents)} documents")
            return result
                
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise

    async def run_ingestion_pipeline(
        self, 
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Run the complete ingestion pipeline.

        Args:
            batch_size: Batch size for processing

        Returns:
            Dictionary with pipeline results
        """
        try:
            logger.info("Starting data ingestion pipeline")
            
            # Step 1: Load data
            df = self.load_reviews_data()
            
            # Step 2: Preprocess data
            documents = self.preprocess_reviews(df)
            
            if not documents:
                raise ValueError("No documents to process")
            
            # Step 3: Index documents
            result = await self.index_documents(documents, batch_size)
            
            logger.info("Data ingestion pipeline completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {str(e)}")
            raise


# Global ingester instance
_ingester = None


def get_ingester(chroma_host: Optional[str] = None, chroma_port: Optional[int] = None) -> DataIngester:
    """Get or create ingester instance."""
    global _ingester
    if _ingester is None:
        _ingester = DataIngester(chroma_host, chroma_port)
    return _ingester
