"""
Ingestion pipeline for Disney reviews using LangChain document processing.

This module provides the main ingestion orchestrator that processes Disney reviews
CSV data and indexes it into the vector database using the document processor.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field

from .document_processor import DocumentProcessor, DocumentProcessorConfig
from ..shared.config import settings

logger = logging.getLogger(__name__)


class IngestionConfig(BaseModel):
    """Configuration for the ingestion pipeline."""
    
    batch_size: int = Field(default=100, description="Batch size for processing")
    max_workers: int = Field(default=4, description="Maximum number of worker processes")
    chunk_size: int = Field(default=1000, description="Document chunk size")
    chunk_overlap: int = Field(default=200, description="Document chunk overlap")
    text_column: str = Field(default="Review_Text", description="CSV column containing review text")
    collection_name: str = Field(default="disney_reviews", description="ChromaDB collection name")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold for search")
    max_context_length: int = Field(default=4000, description="Maximum context length")


class CSVLoader:
    """CSV loader for Disney reviews dataset."""
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        logger.info("CSVLoader initialized")
    
    def load_reviews(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load Disney reviews from CSV file."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            logger.info(f"Loading reviews from: {file_path}")
            
            # Load CSV with pandas for better control
            df = pd.read_csv(file_path)
            
            if self.config.text_column not in df.columns:
                available_columns = list(df.columns)
                raise ValueError(
                    f"Column '{self.config.text_column}' not found. "
                    f"Available columns: {available_columns}"
                )
            
            # Convert to list of dictionaries
            reviews = []
            for idx, row in df.iterrows():
                # Skip rows with empty review text
                if pd.isna(row[self.config.text_column]) or not str(row[self.config.text_column]).strip():
                    continue
                
                review_data = {
                    'id': f"review_{idx}",
                    'content': str(row[self.config.text_column]).strip(),
                    'rating': self._extract_rating(row),
                    'branch': self._extract_branch(row),
                    'year': self._extract_year(row),
                    'source_file': str(file_path),
                    'row_index': idx,
                }
                reviews.append(review_data)
            
            logger.info(f"Loaded {len(reviews)} reviews from CSV")
            return reviews
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            raise
    
    def _extract_rating(self, row: pd.Series) -> Optional[int]:
        """Extract rating from row data."""
        # Try different possible rating columns
        rating_columns = ['Rating', 'rating', 'stars', 'Stars']
        for col in rating_columns:
            if col in row and pd.notna(row[col]):
                try:
                    rating = int(float(row[col]))
                    return min(max(rating, 1), 5)  # Clamp between 1-5
                except (ValueError, TypeError):
                    continue
        return None
    
    def _extract_branch(self, row: pd.Series) -> Optional[str]:
        """Extract Disney branch from row data."""
        # Try different possible branch columns
        branch_columns = ['Branch', 'branch', 'Location', 'location', 'Park', 'park']
        for col in branch_columns:
            if col in row and pd.notna(row[col]):
                branch = str(row[col]).strip().lower()
                if 'disneyland' in branch or 'california' in branch:
                    return 'Disneyland'
                elif 'disney world' in branch or 'florida' in branch:
                    return 'Disney World'
        return None
    
    def _extract_year(self, row: pd.Series) -> Optional[int]:
        """Extract year from row data."""
        # Try different possible year columns
        year_columns = ['Year_Month', 'year_month', 'Year', 'year', 'Date', 'date']
        for col in year_columns:
            if col in row and pd.notna(row[col]):
                try:
                    year_str = str(row[col])
                    # Extract year from various formats
                    if '/' in year_str:
                        year_str = year_str.split('/')[-1]  # Take last part after /
                    elif '-' in year_str:
                        year_str = year_str.split('-')[0]  # Take first part before -
                    
                    year = int(year_str)
                    current_year = datetime.now().year
                    if 2000 <= year <= current_year:
                        return year
                except (ValueError, TypeError):
                    continue
        return None


class BatchIndexer:
    """Efficient batch indexing to vector database."""
    
    def __init__(self, vector_db, config: IngestionConfig):
        self.vector_db = vector_db
        self.config = config
        logger.info("BatchIndexer initialized")
    
    async def index_documents_batch(
        self, 
        documents: List[Any], 
        collection_name: str,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Index a batch of documents to the vector database."""
        batch_size = batch_size or self.config.batch_size
        
        try:
            logger.info(f"Indexing {len(documents)} documents in batches of {batch_size}")
            
            total_indexed = 0
            start_time = time.time()
            
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(documents) + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                # Index batch
                batch_result = await self._index_single_batch(batch, collection_name)
                total_indexed += batch_result.get('indexed_count', 0)
                
                # Log progress
                if batch_num % 10 == 0 or batch_num == total_batches:
                    elapsed = time.time() - start_time
                    rate = total_indexed / elapsed if elapsed > 0 else 0
                    logger.info(f"Progress: {total_indexed}/{len(documents)} documents indexed "
                              f"({rate:.1f} docs/sec)")
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'total_documents': len(documents),
                'indexed_count': total_indexed,
                'processing_time_seconds': round(processing_time, 2),
                'indexing_rate': round(total_indexed / processing_time, 2) if processing_time > 0 else 0,
                'collection_name': collection_name,
            }
            
            logger.info(f"Batch indexing completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in batch indexing: {str(e)}")
            raise
    
    async def _index_single_batch(self, documents: List[Any], collection_name: str) -> Dict[str, Any]:
        """Index a single batch of documents."""
        try:
            # Index to vector database using the correct method signature
            result = self.vector_db.add_documents(documents)
            
            return {
                'indexed_count': len(documents),
                'success': True,
            }
            
        except Exception as e:
            logger.error(f"Error indexing batch: {str(e)}")
            return {
                'indexed_count': 0,
                'success': False,
                'error': str(e),
            }


class IngestionPipeline:
    """Main ingestion pipeline orchestrator."""
    
    def __init__(
        self, 
        vector_db,
        config: Optional[IngestionConfig] = None,
        doc_config: Optional[DocumentProcessorConfig] = None
    ):
        self.vector_db = vector_db
        self.config = config or IngestionConfig()
        self.doc_config = doc_config or DocumentProcessorConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=10,
            max_chunk_size=2000
        )
        
        self.csv_loader = CSVLoader(self.config)
        self.document_processor = DocumentProcessor(self.doc_config)
        self.batch_indexer = BatchIndexer(vector_db, self.config)
        
        logger.info("IngestionPipeline initialized")
    
    async def ingest_csv_file(
        self, 
        file_path: Union[str, Path],
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ingest Disney reviews from CSV file."""
        collection_name = collection_name or self.config.collection_name
        
        try:
            start_time = time.time()
            logger.info(f"Starting ingestion of {file_path} into collection '{collection_name}'")
            
            # Step 1: Load reviews from CSV
            logger.info("Step 1: Loading reviews from CSV")
            reviews = self.csv_loader.load_reviews(file_path)
            
            if not reviews:
                raise ValueError("No reviews found in CSV file")
            
            # Step 2: Process reviews into documents
            logger.info("Step 2: Processing reviews into documents")
            documents = self.document_processor.process_reviews_batch(reviews)
            
            if not documents:
                raise ValueError("No documents generated from reviews")
            
            # Step 3: Get processing statistics
            stats = self.document_processor.get_processing_stats(documents)
            logger.info(f"Document processing stats: {stats}")
            
            # Step 4: Index documents to vector database
            logger.info("Step 3: Indexing documents to vector database")
            indexing_result = await self.batch_indexer.index_documents_batch(
                documents, collection_name
            )
            
            # Step 5: Validate indexing
            logger.info("Step 4: Validating indexing")
            validation_result = await self._validate_indexing(collection_name, len(documents))
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Compile results
            result = {
                'success': True,
                'file_path': str(file_path),
                'collection_name': collection_name,
                'total_reviews': len(reviews),
                'total_documents': len(documents),
                'indexed_documents': indexing_result.get('indexed_count', 0),
                'processing_stats': stats,
                'indexing_result': indexing_result,
                'validation_result': validation_result,
                'total_processing_time_seconds': round(total_time, 2),
                'processing_rate': round(len(documents) / total_time, 2) if total_time > 0 else 0,
                'timestamp': datetime.now().isoformat(),
            }
            
            logger.info(f"Ingestion completed successfully: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in ingestion pipeline: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'file_path': str(file_path),
                'collection_name': collection_name,
                'timestamp': datetime.now().isoformat(),
            }
    
    async def _validate_indexing(self, collection_name: str, expected_count: int) -> Dict[str, Any]:
        """Validate that indexing was successful."""
        try:
            # Get collection stats
            stats = self.vector_db.get_collection_stats()
            actual_count = stats.get('document_count', 0)
            
            success = actual_count >= expected_count * 0.95  # Allow 5% tolerance
            
            return {
                'success': success,
                'expected_count': expected_count,
                'actual_count': actual_count,
                'validation_passed': success,
            }
            
        except Exception as e:
            logger.error(f"Error validating indexing: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'validation_passed': False,
            }
    
    async def reindex_collection(
        self, 
        file_path: Union[str, Path],
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Reindex a collection (clear and re-ingest)."""
        collection_name = collection_name or self.config.collection_name
        
        try:
            logger.info(f"Reindexing collection '{collection_name}'")
            
            # Clear existing collection
            self.vector_db.delete_collection()
            logger.info(f"Cleared existing collection '{collection_name}'")
            
            # Re-ingest
            result = await self.ingest_csv_file(file_path, collection_name)
            result['reindexed'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error reindexing collection: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'reindexed': False,
            }
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion pipeline statistics."""
        return {
            'config': self.config.dict(),
            'doc_config': self.doc_config.dict(),
            'pipeline_initialized': True,
            'timestamp': datetime.now().isoformat(),
        }


def create_ingestion_pipeline(
    vector_db,
    batch_size: int = 100,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    collection_name: str = "disney_reviews"
) -> IngestionPipeline:
    """Create an ingestion pipeline with custom configuration."""
    config = IngestionConfig(
        batch_size=batch_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        collection_name=collection_name
    )
    
    doc_config = DocumentProcessorConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=10,
        max_chunk_size=2000
    )
    
    return IngestionPipeline(vector_db, config, doc_config)
