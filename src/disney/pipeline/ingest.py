"""Data ingestion logic for Disney reviews pipeline."""

import pandas as pd
from typing import List, Dict, Any, Optional
import httpx

from ..shared.config import settings
from ..shared.logging import setup_logging

logger = setup_logging("data-pipeline")


class DataIngester:
    """Data ingestion component for processing Disney reviews."""
    
    def __init__(self, context_service_url: Optional[str] = None):
        """Initialize the data ingester.
        
        Args:
            context_service_url: URL of the Context Retrieval Service
        """
        self.context_service_url = context_service_url or settings.context_service_url
        self.data_path = settings.data_path
    
    def load_reviews_data(self) -> pd.DataFrame:
        """Load Disney reviews data from CSV file.
        
        Returns:
            DataFrame with reviews data
        """
        try:
            logger.info(f"Loading reviews data from: {self.data_path}")
            
            # Load CSV data
            df = pd.read_csv(self.data_path)
            
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
            List of processed review documents
        """
        try:
            logger.info("Preprocessing reviews data")
            
            documents = []
            
            for idx, row in df.iterrows():
                # Extract relevant fields
                review_text = str(row.get('Review_Text', ''))
                rating = row.get('Rating', 0)
                year = row.get('Year_Month', '').split('-')[0] if pd.notna(row.get('Year_Month')) else 'Unknown'
                branch = row.get('Branch', 'Unknown')
                
                # Skip empty reviews
                if not review_text or review_text.strip() == '':
                    continue
                
                # Create document
                doc = {
                    'id': f"review_{idx}",
                    'content': review_text.strip(),
                    'metadata': {
                        'rating': int(rating) if pd.notna(rating) else 0,
                        'year': year,
                        'branch': branch,
                        'original_index': idx
                    }
                }
                
                documents.append(doc)
            
            logger.info(f"Preprocessed {len(documents)} review documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error preprocessing reviews: {str(e)}")
            raise
    
    async def index_documents(
        self, 
        documents: List[Dict[str, Any]], 
        batch_size: int = 100
    ) -> bool:
        """Index documents into the vector database.
        
        Args:
            documents: List of document dictionaries
            batch_size: Number of documents to process per batch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Indexing {len(documents)} documents in batches of {batch_size}")
            
            total_indexed = 0
            
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Prepare batch data
                batch_docs = [doc['content'] for doc in batch]
                batch_metadatas = [doc['metadata'] for doc in batch]
                batch_ids = [doc['id'] for doc in batch]
                
                # Send to Context Retrieval Service
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self.context_service_url}/api/v1/index",
                        json={
                            "documents": batch_docs,
                            "metadatas": batch_metadatas,
                            "ids": batch_ids
                        }
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    if result.get('success'):
                        total_indexed += len(batch)
                        logger.info(f"Indexed batch {i//batch_size + 1}: {len(batch)} documents")
                    else:
                        logger.error(f"Failed to index batch {i//batch_size + 1}")
                        return False
            
            logger.info(f"Successfully indexed {total_indexed} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return False
    
    async def run_ingestion(self) -> bool:
        """Run the complete data ingestion pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting data ingestion pipeline")
            
            # Load data
            df = self.load_reviews_data()
            
            # Preprocess data
            documents = self.preprocess_reviews(df)
            
            # Index documents
            success = await self.index_documents(documents)
            
            if success:
                logger.info("Data ingestion pipeline completed successfully")
            else:
                logger.error("Data ingestion pipeline failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {str(e)}")
            return False


# Global ingester instance
_ingester = None


def get_ingester() -> DataIngester:
    """Get or create ingester instance."""
    global _ingester
    if _ingester is None:
        _ingester = DataIngester()
    return _ingester
