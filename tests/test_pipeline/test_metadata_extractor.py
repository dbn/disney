"""
Tests for the metadata extraction module.

This module tests the MetadataExtractor class and its individual extraction functions.
"""

import pytest
from datetime import datetime
from disney.pipeline.metadata_extractor import MetadataExtractor


class TestMetadataExtractor:
    """Test metadata extraction functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = MetadataExtractor()
    
    def test_extract_rating_valid(self):
        """Test rating extraction with valid values."""
        # Test valid ratings
        assert self.extractor.extract_rating(5) == 5
        assert self.extractor.extract_rating(3) == 3
        assert self.extractor.extract_rating(1) == 1
        assert self.extractor.extract_rating("4") == 4
        assert self.extractor.extract_rating(4.0) == 4
    
    def test_extract_rating_invalid(self):
        """Test rating extraction with invalid values."""
        # Test invalid ratings
        assert self.extractor.extract_rating(0) is None
        assert self.extractor.extract_rating(6) is None
        assert self.extractor.extract_rating(-1) is None
        assert self.extractor.extract_rating("invalid") is None
        assert self.extractor.extract_rating(None) is None
        assert self.extractor.extract_rating("") is None
        assert self.extractor.extract_rating("nan") is None
    
    def test_extract_year_valid(self):
        """Test year extraction with valid Year_Month formats."""
        # Test valid year formats
        assert self.extractor.extract_year("2019-4") == 2019
        assert self.extractor.extract_year("2020-12") == 2020
        assert self.extractor.extract_year("2023-1") == 2023
        assert self.extractor.extract_year("2015") == 2015
    
    def test_extract_year_invalid(self):
        """Test year extraction with invalid formats."""
        # Test invalid year formats
        assert self.extractor.extract_year("") is None
        assert self.extractor.extract_year("nan") is None
        assert self.extractor.extract_year("invalid") is None
        assert self.extractor.extract_year("1999-4") is None  # Too old
        assert self.extractor.extract_year("2030-4") is None  # Too future
        assert self.extractor.extract_year("19-4") is None  # Invalid format
    
    def test_extract_month_valid(self):
        """Test month extraction with valid Year_Month formats."""
        # Test valid month formats
        assert self.extractor.extract_month("2019-4") == 4
        assert self.extractor.extract_month("2020-12") == 12
        assert self.extractor.extract_month("2023-1") == 1
        assert self.extractor.extract_month("2015-6") == 6
    
    def test_extract_month_invalid(self):
        """Test month extraction with invalid formats."""
        # Test invalid month formats
        assert self.extractor.extract_month("") is None
        assert self.extractor.extract_month("nan") is None
        assert self.extractor.extract_month("invalid") is None
        assert self.extractor.extract_month("2019") is None  # No month
        assert self.extractor.extract_month("2019-13") is None  # Invalid month
        assert self.extractor.extract_month("2019-0") is None  # Invalid month
    
    def test_extract_branch_normalization(self):
        """Test branch name extraction (no normalization currently implemented)."""
        # Test branch extraction - currently no normalization is implemented
        assert self.extractor.extract_branch("Disneyland") == "Disneyland"
        assert self.extractor.extract_branch("disneyland") == "disneyland"  # No normalization
        assert self.extractor.extract_branch("DISNEYLAND") == "DISNEYLAND"  # No normalization
        assert self.extractor.extract_branch("Disney World") == "Disney World"
        assert self.extractor.extract_branch("disney world") == "disney world"  # No normalization
        assert self.extractor.extract_branch("Disneyland_HongKong") == "Disneyland_HongKong"  # No normalization
        assert self.extractor.extract_branch("Unknown") == "Unknown"
        assert self.extractor.extract_branch("") == "Unknown"
        assert self.extractor.extract_branch(None) == "Unknown"
        assert self.extractor.extract_branch("nan") == "Unknown"
    
    def test_extract_review_id(self):
        """Test review ID extraction."""
        # Test valid review IDs
        assert self.extractor.extract_review_id("12345") == "12345"
        assert self.extractor.extract_review_id(12345) == "12345"
        assert self.extractor.extract_review_id("review_123") == "review_123"
        
        # Test invalid review IDs (should generate fallback)
        result = self.extractor.extract_review_id(None)
        assert result.startswith("review_")
        result_empty = self.extractor.extract_review_id("")
        assert result_empty.startswith("review_")
        result_nan = self.extractor.extract_review_id("nan")
        assert result_nan.startswith("review_")
    
    def test_extract_reviewer_location(self):
        """Test reviewer location extraction."""
        # Test valid locations
        assert self.extractor.extract_reviewer_location("United States") == "United States"
        assert self.extractor.extract_reviewer_location("Australia") == "Australia"
        assert self.extractor.extract_reviewer_location("United Kingdom") == "United Kingdom"
        
        # Test invalid locations
        assert self.extractor.extract_reviewer_location("") is None
        assert self.extractor.extract_reviewer_location(None) is None
        assert self.extractor.extract_reviewer_location("nan") is None
    
    def test_validation_functions(self):
        """Test all validation functions."""
        # Test rating validation
        assert self.extractor.validate_rating(1) is True
        assert self.extractor.validate_rating(3) is True
        assert self.extractor.validate_rating(5) is True
        assert self.extractor.validate_rating(0) is False
        assert self.extractor.validate_rating(6) is False
        
        # Test year validation
        current_year = datetime.now().year
        assert self.extractor.validate_year(2000) is True
        assert self.extractor.validate_year(current_year) is True
        assert self.extractor.validate_year(1999) is False
        assert self.extractor.validate_year(current_year + 1) is False
        
        # Test month validation
        assert self.extractor.validate_month(1) is True
        assert self.extractor.validate_month(6) is True
        assert self.extractor.validate_month(12) is True
        assert self.extractor.validate_month(0) is False
        assert self.extractor.validate_month(13) is False
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with various data types
        assert self.extractor.extract_rating(True) == 1  # Boolean True converts to 1
        assert self.extractor.extract_rating(False) is None  # Boolean False converts to 0, which is invalid
        assert self.extractor.extract_rating([]) is None  # List
        assert self.extractor.extract_rating({}) is None  # Dict
        
        # Test with whitespace
        assert self.extractor.extract_branch("  Disneyland  ") == "Disneyland"  # Whitespace stripped
        assert self.extractor.extract_reviewer_location("  Australia  ") == "Australia"  # Whitespace stripped
        
        # Test with special characters
        assert self.extractor.extract_branch("Disneyland-HongKong") == "Disneyland-HongKong"  # No normalization
        assert self.extractor.extract_reviewer_location("United States") == "United States"
    
    def test_extract_all_metadata(self):
        """Test extracting all metadata from row data."""
        # Test with complete row data
        row_data = {
            'Rating': 4,
            'Year_Month': '2019-4',
            'Branch': 'Disneyland',
            'Review_ID': '12345',
            'Reviewer_Location': 'United States'
        }
        
        metadata = self.extractor.extract_all_metadata(row_data)
        
        assert metadata['rating'] == 4
        assert metadata['year'] == 2019
        assert metadata['month'] == 4
        assert metadata['branch'] == 'Disneyland'
        assert metadata['review_id'] == '12345'
        assert metadata['reviewer_location'] == 'United States'
    
    def test_extract_all_metadata_partial(self):
        """Test extracting metadata with partial row data."""
        # Test with partial row data
        row_data = {
            'Rating': 3,
            'Year_Month': '2020-12',
            'Branch': 'Disney World'
        }
        
        metadata = self.extractor.extract_all_metadata(row_data)
        
        assert metadata['rating'] == 3
        assert metadata['year'] == 2020
        assert metadata['month'] == 12
        assert metadata['branch'] == 'Disney World'
        assert 'review_id' in metadata
        assert 'reviewer_location' not in metadata
    
    def test_get_extraction_stats(self):
        """Test extraction statistics generation."""
        # Test with empty list
        stats = self.extractor.get_extraction_stats([])
        assert stats == {}
        
        # Test with sample metadata
        metadata_list = [
            {'rating': 5, 'year': 2019, 'month': 4, 'branch': 'Disneyland'},
            {'rating': 3, 'year': 2020, 'month': 6, 'branch': 'Disney World'},
            {'rating': 4, 'year': 2021, 'month': 8, 'branch': 'Disneyland'}
        ]
        
        stats = self.extractor.get_extraction_stats(metadata_list)
        
        assert stats['total_records'] == 3
        assert 'fields_present' in stats
        assert 'field_counts' in stats
        
        # Check field presence
        assert stats['fields_present']['rating']['count'] == 3
        assert stats['fields_present']['rating']['percentage'] == 100.0
        assert stats['fields_present']['year']['count'] == 3
        assert stats['fields_present']['month']['count'] == 3
        
        # Check field counts
        assert stats['field_counts']['rating'][5] == 1
        assert stats['field_counts']['rating'][3] == 1
        assert stats['field_counts']['rating'][4] == 1
        assert stats['field_counts']['branch']['Disneyland'] == 2
        assert stats['field_counts']['branch']['Disney World'] == 1


class TestMetadataExtractorIntegration:
    """Test metadata extractor integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = MetadataExtractor()
    
    def test_real_world_data_scenarios(self):
        """Test with real-world data scenarios."""
        # Test with typical CSV row data
        csv_row = {
            'Review_ID': '670772142',
            'Rating': 4,
            'Year_Month': '2019-4',
            'Reviewer_Location': 'Australia',
            'Review_Text': 'Great experience at Disneyland!',
            'Branch': 'Disneyland_HongKong'
        }
        
        metadata = self.extractor.extract_all_metadata(csv_row)
        
        assert metadata['rating'] == 4
        assert metadata['year'] == 2019
        assert metadata['month'] == 4
        assert metadata['branch'] == 'Disneyland_HongKong'  # No normalization
        assert metadata['review_id'] == '670772142'
        assert metadata['reviewer_location'] == 'Australia'
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data."""
        # Test with malformed data
        malformed_row = {
            'Rating': 'invalid',
            'Year_Month': 'not-a-date',
            'Branch': None,
            'Review_ID': '',
            'Reviewer_Location': 'nan'
        }
        
        metadata = self.extractor.extract_all_metadata(malformed_row)
        
        # Should handle malformed data gracefully
        assert 'rating' not in metadata or metadata['rating'] is None
        assert 'year' not in metadata or metadata['year'] is None
        assert 'month' not in metadata or metadata['month'] is None
        assert metadata['branch'] == 'Unknown'
        assert 'review_id' in metadata  # Should generate fallback
        assert 'reviewer_location' not in metadata or metadata['reviewer_location'] is None
    
    def test_performance_with_large_dataset(self):
        """Test performance with simulated large dataset."""
        # Create a large list of metadata
        large_metadata_list = []
        for i in range(1000):
            metadata = {
                'rating': (i % 5) + 1,
                'year': 2019 + (i % 4),
                'month': (i % 12) + 1,
                'branch': 'Disneyland' if i % 2 == 0 else 'Disney World'
            }
            large_metadata_list.append(metadata)
        
        # Test statistics generation
        stats = self.extractor.get_extraction_stats(large_metadata_list)
        
        assert stats['total_records'] == 1000
        assert stats['fields_present']['rating']['count'] == 1000
        assert stats['fields_present']['year']['count'] == 1000
        assert stats['fields_present']['month']['count'] == 1000
        assert stats['fields_present']['branch']['count'] == 1000
