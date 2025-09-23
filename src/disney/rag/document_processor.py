"""
Document processing module using LangChain for Disney reviews.

This module provides document preprocessing, cleaning, chunking, and metadata
extraction capabilities using LangChain's document processing pipeline.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import CSVLoader
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentProcessorConfig(BaseModel):
    """Configuration for document processing."""
    
    chunk_size: int = Field(default=1000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size")
    max_chunk_size: int = Field(default=2000, description="Maximum chunk size")
    separators: List[str] = Field(
        default=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        description="Text separators for splitting"
    )
    keep_separator: bool = Field(default=True, description="Keep separators in chunks")
    strip_whitespace: bool = Field(default=True, description="Strip whitespace from chunks")


class MetadataExtractor:
    """Extracts and enriches metadata from Disney reviews."""
    
    def __init__(self):
        self.rating_pattern = re.compile(r'(\d+)\s*stars?', re.IGNORECASE)
        self.date_patterns = [
            re.compile(r'(\d{4})', re.IGNORECASE),  # Year
            re.compile(r'(\d{1,2})/(\d{1,2})/(\d{4})', re.IGNORECASE),  # MM/DD/YYYY
            re.compile(r'(\d{4})-(\d{1,2})-(\d{1,2})', re.IGNORECASE),  # YYYY-MM-DD
        ]
        self.branch_patterns = [
            re.compile(r'disneyland', re.IGNORECASE),
            re.compile(r'disney world', re.IGNORECASE),
            re.compile(r'california', re.IGNORECASE),
            re.compile(r'florida', re.IGNORECASE),
        ]
    
    def extract_rating(self, text: str) -> Optional[int]:
        """Extract rating from text."""
        match = self.rating_pattern.search(text)
        if match:
            rating = int(match.group(1))
            return min(max(rating, 1), 5)  # Clamp between 1-5
        return None
    
    def extract_year(self, text: str) -> Optional[int]:
        """Extract year from text."""
        for pattern in self.date_patterns:
            match = pattern.search(text)
            if match:
                if len(match.groups()) == 1:
                    year = int(match.group(1))
                else:
                    year = int(match.group(-1))  # Last group is usually year
                
                current_year = datetime.now().year
                if 2000 <= year <= current_year:
                    return year
        return None
    
    def extract_branch(self, text: str) -> Optional[str]:
        """Extract Disney branch from text."""
        text_lower = text.lower()
        if 'disneyland' in text_lower or 'california' in text_lower:
            return 'Disneyland'
        elif 'disney world' in text_lower or 'florida' in text_lower:
            return 'Disney World'
        return None
    
    def extract_sentiment_keywords(self, text: str) -> List[str]:
        """Extract sentiment-related keywords."""
        positive_words = [
            'amazing', 'awesome', 'fantastic', 'wonderful', 'excellent', 'great',
            'love', 'loved', 'perfect', 'best', 'incredible', 'magical', 'fun'
        ]
        negative_words = [
            'terrible', 'awful', 'horrible', 'disappointing', 'bad', 'worst',
            'hate', 'hated', 'waste', 'boring', 'overpriced', 'crowded'
        ]
        
        text_lower = text.lower()
        keywords = []
        
        for word in positive_words:
            if word in text_lower:
                keywords.append(f"positive_{word}")
        
        for word in negative_words:
            if word in text_lower:
                keywords.append(f"negative_{word}")
        
        return keywords
    
    def extract_attractions(self, text: str) -> List[str]:
        """Extract attraction mentions from text."""
        attractions = [
            'space mountain', 'pirates of the caribbean', 'haunted mansion',
            'it\'s a small world', 'jungle cruise', 'big thunder mountain',
            'splash mountain', 'matterhorn', 'indiana jones', 'star wars',
            'tower of terror', 'rock \'n\' roller coaster', 'expedition everest',
            'avatar', 'toy story', 'frozen', 'aladdin', 'lion king'
        ]
        
        text_lower = text.lower()
        mentioned_attractions = []
        
        for attraction in attractions:
            if attraction in text_lower:
                mentioned_attractions.append(attraction.title())
        
        return mentioned_attractions
    
    def enrich_metadata(self, text: str, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich metadata with extracted information."""
        enriched = base_metadata.copy()
        
        # Extract rating if not already present
        if 'rating' not in enriched or enriched['rating'] is None:
            rating = self.extract_rating(text)
            if rating:
                enriched['rating'] = rating
        
        # Extract year if not already present
        if 'year' not in enriched or enriched['year'] is None:
            year = self.extract_year(text)
            if year:
                enriched['year'] = year
        
        # Extract branch if not already present
        if 'branch' not in enriched or enriched['branch'] is None:
            branch = self.extract_branch(text)
            if branch:
                enriched['branch'] = branch
        
        # Add sentiment keywords
        sentiment_keywords = self.extract_sentiment_keywords(text)
        if sentiment_keywords:
            enriched['sentiment_keywords'] = sentiment_keywords
        
        # Add attractions
        attractions = self.extract_attractions(text)
        if attractions:
            enriched['attractions'] = attractions
        
        # Add processing timestamp
        enriched['processed_at'] = datetime.now().isoformat()
        
        return enriched


class TextSplitter:
    """LangChain-based text splitting for Disney reviews."""
    
    def __init__(self, config: DocumentProcessorConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=config.separators,
            keep_separator=config.keep_separator,
            is_separator_regex=False,
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        try:
            chunks = self.splitter.split_text(text)
            
            # Filter chunks by size
            filtered_chunks = []
            for chunk in chunks:
                if self.config.min_chunk_size <= len(chunk) <= self.config.max_chunk_size:
                    if self.config.strip_whitespace:
                        chunk = chunk.strip()
                    if chunk:  # Only add non-empty chunks
                        filtered_chunks.append(chunk)
            
            logger.info(f"Split text into {len(filtered_chunks)} chunks")
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            # Fallback: return original text as single chunk
            return [text] if text.strip() else []


class DocumentProcessor:
    """Main document processing class using LangChain."""
    
    def __init__(self, config: Optional[DocumentProcessorConfig] = None):
        self.config = config or DocumentProcessorConfig()
        self.text_splitter = TextSplitter(self.config)
        self.metadata_extractor = MetadataExtractor()
        logger.info("DocumentProcessor initialized")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'"]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def process_review(self, review_data: Dict[str, Any]) -> List[Document]:
        """Process a single Disney review into LangChain Documents."""
        try:
            # Extract basic information
            review_id = review_data.get('id', f"review_{hash(str(review_data))}")
            content = review_data.get('content', '')
            rating = review_data.get('rating')
            branch = review_data.get('branch')
            year = review_data.get('year')
            
            # Clean content
            cleaned_content = self.clean_text(content)
            if not cleaned_content:
                logger.warning(f"Empty content for review {review_id}")
                return []
            
            # Create base metadata
            base_metadata = {
                'id': review_id,
                'rating': rating,
                'branch': branch,
                'year': year,
                'source': 'disney_reviews',
                'content_length': len(cleaned_content),
            }
            
            # Enrich metadata
            enriched_metadata = self.metadata_extractor.enrich_metadata(
                cleaned_content, base_metadata
            )
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(cleaned_content)
            
            # Create LangChain Documents
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = enriched_metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_length': len(chunk),
                })
                
                document = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(document)
            
            logger.info(f"Processed review {review_id} into {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing review {review_data.get('id', 'unknown')}: {str(e)}")
            return []
    
    def process_reviews_batch(self, reviews: List[Dict[str, Any]]) -> List[Document]:
        """Process a batch of Disney reviews."""
        all_documents = []
        
        logger.info(f"Processing batch of {len(reviews)} reviews")
        
        for review in reviews:
            documents = self.process_review(review)
            all_documents.extend(documents)
        
        logger.info(f"Processed {len(reviews)} reviews into {len(all_documents)} total documents")
        return all_documents
    
    def process_csv_file(self, file_path: str, text_column: str = 'Review_Text') -> List[Document]:
        """Process Disney reviews CSV file."""
        try:
            logger.info(f"Loading CSV file: {file_path}")
            
            # Load CSV using pandas for better control
            df = pd.read_csv(file_path)
            
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in CSV file")
            
            # Convert DataFrame to list of dictionaries
            reviews = []
            for idx, row in df.iterrows():
                review_data = {
                    'id': f"review_{idx}",
                    'content': str(row[text_column]) if pd.notna(row[text_column]) else '',
                    'rating': row.get('Rating', None),
                    'branch': row.get('Branch', None),
                    'year': row.get('Year_Month', None),
                }
                reviews.append(review_data)
            
            # Process reviews
            documents = self.process_reviews_batch(reviews)
            
            logger.info(f"Successfully processed CSV file into {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {str(e)}")
            raise
    
    def get_processing_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about processed documents."""
        if not documents:
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'average_chunk_size': 0,
                'rating_distribution': {},
                'branch_distribution': {},
                'year_distribution': {},
            }
        
        total_chunks = len(documents)
        total_content_length = sum(len(doc.page_content) for doc in documents)
        average_chunk_size = total_content_length / total_chunks if total_chunks > 0 else 0
        
        # Rating distribution
        rating_dist = {}
        for doc in documents:
            rating = doc.metadata.get('rating')
            if rating:
                rating_dist[rating] = rating_dist.get(rating, 0) + 1
        
        # Branch distribution
        branch_dist = {}
        for doc in documents:
            branch = doc.metadata.get('branch')
            if branch:
                branch_dist[branch] = branch_dist.get(branch, 0) + 1
        
        # Year distribution
        year_dist = {}
        for doc in documents:
            year = doc.metadata.get('year')
            if year:
                year_dist[year] = year_dist.get(year, 0) + 1
        
        return {
            'total_documents': len(set(doc.metadata.get('id') for doc in documents)),
            'total_chunks': total_chunks,
            'average_chunk_size': round(average_chunk_size, 2),
            'rating_distribution': rating_dist,
            'branch_distribution': branch_dist,
            'year_distribution': year_dist,
        }


def create_document_processor(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_chunk_size: int = 100,
    max_chunk_size: int = 2000
) -> DocumentProcessor:
    """Create a DocumentProcessor with custom configuration."""
    config = DocumentProcessorConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size
    )
    return DocumentProcessor(config)
