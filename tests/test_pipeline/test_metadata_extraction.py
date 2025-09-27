"""
Integration tests for metadata extraction in the pipeline.

This module tests metadata extraction functionality in the context of the
full data ingestion pipeline.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from disney.pipeline.ingest import DataIngester


class TestMetadataExtractionIntegration:
    """Test metadata extraction in pipeline context."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ingester = DataIngester(chroma_host="localhost", chroma_port=8000)
    
    def test_month_extraction_in_pipeline(self):
        """Test that month is properly extracted in pipeline."""
        # Create sample DataFrame with Year_Month data
        sample_data = pd.DataFrame({
            'Review_Text': [
                'Great experience at Disneyland!',
                'Amazing time at Disney World!',
                'Wonderful visit to the park!'
            ],
            'Rating': [5, 4, 3],
            'Year_Month': ['2019-4', '2020-12', '2021-6'],
            'Branch': ['Disneyland', 'Disney World', 'Disneyland'],
            'Review_ID': ['123', '456', '789'],
            'Reviewer_Location': ['USA', 'Canada', 'UK']
        })
        
        # Process the data
        documents = self.ingester.preprocess_reviews(sample_data)
        
        # Verify month extraction
        assert len(documents) == 3
        
        # Check first document
        doc1 = documents[0]
        assert doc1['metadata']['rating'] == 5
        assert doc1['metadata']['year'] == 2019
        assert doc1['metadata']['month'] == 4  # NEW: Month should be extracted
        assert doc1['metadata']['branch'] == 'Disneyland'
        assert doc1['metadata']['reviewer_location'] == 'USA'
        
        # Check second document
        doc2 = documents[1]
        assert doc2['metadata']['rating'] == 4
        assert doc2['metadata']['year'] == 2020
        assert doc2['metadata']['month'] == 12  # NEW: Month should be extracted
        assert doc2['metadata']['branch'] == 'Disney World'
        assert doc2['metadata']['reviewer_location'] == 'Canada'
        
        # Check third document
        doc3 = documents[2]
        assert doc3['metadata']['rating'] == 3
        assert doc3['metadata']['year'] == 2021
        assert doc3['metadata']['month'] == 6  # NEW: Month should be extracted
        assert doc3['metadata']['branch'] == 'Disneyland'
        assert doc3['metadata']['reviewer_location'] == 'UK'
    
    def test_metadata_validation_in_pipeline(self):
        """Test metadata validation in pipeline."""
        # Create sample DataFrame with invalid data
        sample_data = pd.DataFrame({
            'Review_Text': [
                'Great experience!',
                'Amazing time!',
                'Wonderful visit!'
            ],
            'Rating': [6, -1, 'invalid'],  # Invalid ratings
            'Year_Month': ['2030-4', '1999-12', 'invalid'],  # Invalid dates
            'Branch': ['Unknown', 'Disneyland', 'Disney World'],
            'Review_ID': ['123', '456', '789'],
            'Reviewer_Location': ['USA', 'Canada', 'UK']
        })
        
        # Process the data
        documents = self.ingester.preprocess_reviews(sample_data)
        
        # Verify invalid data is handled gracefully - all documents filtered out due to invalid metadata
        assert len(documents) == 0  # All documents filtered out due to invalid metadata
        
        # The preprocessing logic filters out documents with any invalid metadata
        # This is the expected behavior - invalid data is handled by filtering out the entire document
    
    def test_error_handling_in_extraction(self):
        """Test error handling during extraction."""
        # Create sample DataFrame with problematic data
        sample_data = pd.DataFrame({
            'Review_Text': [
                'Great experience!',
                'Amazing time!'
            ],
            'Rating': [None, float('nan')],  # Problematic ratings
            'Year_Month': [None, ''],  # Problematic dates
            'Branch': [None, float('nan')],  # Problematic branches
            'Review_ID': [None, ''],  # Problematic IDs
            'Reviewer_Location': [None, float('nan')]  # Problematic locations
        })
        
        # Process the data - should not raise exceptions
        documents = self.ingester.preprocess_reviews(sample_data)
        
        # Verify data is processed despite errors - all documents filtered out due to invalid metadata
        assert len(documents) == 0  # All documents filtered out due to invalid metadata
        
        # The preprocessing logic filters out documents with any invalid metadata
    
    def test_backward_compatibility(self):
        """Test that existing functionality still works."""
        # Create sample DataFrame with typical data
        sample_data = pd.DataFrame({
            'Review_Text': [
                'Great experience at Disneyland!',
                'Amazing time at Disney World!'
            ],
            'Rating': [5, 4],
            'Year_Month': ['2019-4', '2020-12'],
            'Branch': ['Disneyland', 'Disney World'],
            'Review_ID': ['123', '456'],
            'Reviewer_Location': ['USA', 'Canada']
        })
        
        # Process the data
        documents = self.ingester.preprocess_reviews(sample_data)
        
        # Verify backward compatibility
        assert len(documents) == 2
        
        for doc in documents:
            # Check that all expected fields are present
            assert 'id' in doc
            assert 'content' in doc
            assert 'metadata' in doc
            
            # Check that metadata contains expected fields
            metadata = doc['metadata']
            assert 'rating' in metadata
            assert 'year' in metadata
            assert 'month' in metadata  # NEW: Month should be present
            assert 'branch' in metadata
            assert 'reviewer_location' in metadata  # NEW: Reviewer location should be present
            assert 'original_index' in metadata
    
    def test_metadata_extraction_with_real_csv_format(self):
        """Test metadata extraction with real CSV format data."""
        # Create DataFrame that mimics the real CSV format
        sample_data = pd.DataFrame({
            'Review_ID': ['670772142', '670682799', '670623270'],
            'Rating': [4, 4, 4],
            'Year_Month': ['2019-4', '2019-5', '2019-4'],
            'Reviewer_Location': ['Australia', 'Philippines', 'United Arab Emirates'],
            'Review_Text': [
                'If you\'ve ever been to Disneyland anywhere you\'ll find Disneyland Hong Kong very similar...',
                'Its been a while since d last time we visit HK Disneyland...',
                'Thanks God it wasn\'t too hot or too humid when I was visiting the park...'
            ],
            'Branch': ['Disneyland_HongKong', 'Disneyland_HongKong', 'Disneyland_HongKong']
        })
        
        # Process the data
        documents = self.ingester.preprocess_reviews(sample_data)
        
        # Verify extraction results
        assert len(documents) == 3
        
        # Check first document
        doc1 = documents[0]
        assert doc1['id'] == '670772142'
        assert doc1['metadata']['rating'] == 4
        assert doc1['metadata']['year'] == 2019
        assert doc1['metadata']['month'] == 4
        assert doc1['metadata']['branch'] == 'Disneyland_HongKong'  # No normalization
        assert doc1['metadata']['reviewer_location'] == 'Australia'
        
        # Check second document
        doc2 = documents[1]
        assert doc2['id'] == '670682799'
        assert doc2['metadata']['rating'] == 4
        assert doc2['metadata']['year'] == 2019
        assert doc2['metadata']['month'] == 5
        assert doc2['metadata']['branch'] == 'Disneyland_HongKong'  # No normalization
        assert doc2['metadata']['reviewer_location'] == 'Philippines'
        
        # Check third document
        doc3 = documents[2]
        assert doc3['id'] == '670623270'
        assert doc3['metadata']['rating'] == 4
        assert doc3['metadata']['year'] == 2019
        assert doc3['metadata']['month'] == 4
        assert doc3['metadata']['branch'] == 'Disneyland_HongKong'  # No normalization
        assert doc3['metadata']['reviewer_location'] == 'United Arab Emirates'
    
    def test_metadata_extraction_performance(self):
        """Test metadata extraction performance with larger dataset."""
        # Create a larger dataset
        large_data = []
        for i in range(100):
            large_data.append({
                'Review_ID': f'review_{i}',
                'Rating': (i % 5) + 1,
                'Year_Month': f'2019-{(i % 12) + 1}',
                'Reviewer_Location': 'USA',
                'Review_Text': f'Review number {i} about Disney experience.',
                'Branch': 'Disneyland' if i % 2 == 0 else 'Disney World'
            })
        
        sample_data = pd.DataFrame(large_data)
        
        # Process the data
        documents = self.ingester.preprocess_reviews(sample_data)
        
        # Verify all documents were processed
        assert len(documents) == 100
        
        # Verify metadata extraction worked for all documents
        for i, doc in enumerate(documents):
            assert doc['metadata']['rating'] == (i % 5) + 1
            assert doc['metadata']['year'] == 2019
            assert doc['metadata']['month'] == (i % 12) + 1
            assert doc['metadata']['branch'] in ['Disneyland', 'Disney World']
            assert doc['metadata']['reviewer_location'] == 'USA'
    
    def test_metadata_extraction_with_missing_columns(self):
        """Test metadata extraction when some columns are missing."""
        # Create DataFrame with missing columns
        sample_data = pd.DataFrame({
            'Review_Text': [
                'Great experience!',
                'Amazing time!'
            ],
            'Rating': [5, 4]
            # Missing other columns
        })
        
        # Process the data
        documents = self.ingester.preprocess_reviews(sample_data)
        
        # Verify data is processed despite missing columns - all documents filtered out due to missing required metadata
        assert len(documents) == 0  # All documents filtered out due to missing required metadata
        
        # The preprocessing logic filters out documents with any missing required metadata
