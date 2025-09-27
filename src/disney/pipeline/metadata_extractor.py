"""
Unified metadata extraction for Disney reviews.

This module provides centralized metadata extraction functionality that can be used
across the Disney reviews pipeline, ensuring consistent extraction and validation
of metadata fields.
"""

import logging
from datetime import datetime
from typing import Any, Optional, Dict, List

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Unified metadata extraction for Disney reviews."""
    
    def __init__(self):
        """Initialize the metadata extractor."""
        self.current_year = datetime.now().year
        logger.debug("MetadataExtractor initialized")
    
    def extract_rating(self, value: Any) -> Optional[int]:
        """Extract and validate rating (1-5).
        
        Args:
            value: Rating value from CSV or other source
            
        Returns:
            Validated rating between 1-5, or None if invalid
        """
        if value is None or value == '' or str(value).lower() == 'nan':
            return None
        
        try:
            rating = int(float(value))
            if self.validate_rating(rating):
                return rating
            else:
                logger.warning(f"Rating {rating} is out of valid range (1-5)")
                return None
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to extract rating from {value}: {e}")
            return None
    
    def extract_year(self, year_month: str) -> Optional[int]:
        """Extract year from Year_Month format (YYYY-M).
        
        Args:
            year_month: Year_Month string in format "YYYY-M" (e.g., "2019-4")
            
        Returns:
            Extracted year, or None if invalid
        """
        if not year_month or year_month == 'nan' or year_month.lower() == 'missing':
            return None
        
        try:
            # Handle formats like "2019-4", "2020-12"
            if '-' in year_month:
                parts = year_month.split('-')
                if len(parts) == 2:
                    year = int(parts[0])
                    if self.validate_year(year):
                        return year
            else:
                # Handle case where only year is provided
                year = int(year_month)
                if self.validate_year(year):
                    return year
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to extract year from {year_month}: {e}")
        
        return None
    
    def extract_month(self, year_month: str) -> Optional[int]:
        """Extract month from Year_Month format (YYYY-M).
        
        Args:
            year_month: Year_Month string in format "YYYY-M" (e.g., "2019-4")
            
        Returns:
            Extracted month (1-12), or None if invalid
        """
        if not year_month or year_month == 'nan' or year_month.lower() == 'missing':
            return None
        
        try:
            # Handle formats like "2019-4", "2020-12"
            if '-' in year_month:
                parts = year_month.split('-')
                if len(parts) == 2:
                    month = int(parts[1])
                    if self.validate_month(month):
                        return month
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to extract month from {year_month}: {e}")
        
        return None
    
    def extract_branch(self, value: Any) -> str:
        """Extract and normalize branch name.
        
        Args:
            value: Branch value from CSV or other source
            
        Returns:
            Normalized branch name
        """
        if value is None or value == '' or str(value).lower() == 'nan':
            return 'Unknown'
        
        branch = str(value).strip()
        return self.normalize_branch(branch)
    
    def extract_review_id(self, value: Any) -> str:
        """Extract and format review ID.
        
        Args:
            value: Review ID from CSV or other source
            
        Returns:
            Formatted review ID
        """
        if value is None or value == '' or str(value).lower() == 'nan':
            return f"review_{hash(str(value))}"
        
        return str(value).strip()
    
    def extract_reviewer_location(self, value: Any) -> Optional[str]:
        """Extract reviewer location.
        
        Args:
            value: Reviewer location from CSV or other source
            
        Returns:
            Reviewer location, or None if invalid
        """
        if value is None or value == '' or str(value).lower() == 'nan':
            return None
        
        location = str(value).strip()
        return location if location else None
    
    def validate_rating(self, rating: int) -> bool:
        """Validate rating is between 1-5.
        
        Args:
            rating: Rating value to validate
            
        Returns:
            True if valid, False otherwise
        """
        return 1 <= rating <= 5
    
    def validate_year(self, year: int) -> bool:
        """Validate year is reasonable (2000-current).
        
        Args:
            year: Year value to validate
            
        Returns:
            True if valid, False otherwise
        """
        return 2000 <= year <= self.current_year
    
    def validate_month(self, month: int) -> bool:
        """Validate month is between 1-12.
        
        Args:
            month: Month value to validate
            
        Returns:
            True if valid, False otherwise
        """
        return 1 <= month <= 12
    
    def normalize_branch(self, branch: str) -> str:
        """Normalize branch names to standard format.
        
        Args:
            branch: Raw branch name
            
        Returns:
            Normalized branch name
        """
        if not branch:
            return 'Unknown'
        
        branch_lower = branch.lower().strip()
        
        # Map common variations to standard names
        branch_mapping = {
            'disneyland': 'Disneyland',
            'disney world': 'Disney World',
            'disneyland_hongkong': 'Disneyland Hong Kong',
            'disneyland hong kong': 'Disneyland Hong Kong',
            'disneyland paris': 'Disneyland Paris',
            'tokyo disneyland': 'Tokyo Disneyland',
            'tokyo disney': 'Tokyo Disneyland',
            'unknown': 'Unknown'
        }
        
        # Check for exact matches first
        if branch_lower in branch_mapping:
            return branch_mapping[branch_lower]
        
        # Check for partial matches
        for key, value in branch_mapping.items():
            if key in branch_lower:
                return value
        
        # If no match found, return the original with proper capitalization
        return branch.title()
    
    def extract_all_metadata(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all metadata from a single row of data.
        
        Args:
            row_data: Dictionary containing row data from CSV
            
        Returns:
            Dictionary containing all extracted metadata
        """
        metadata = {}
        
        # Extract rating
        rating = self.extract_rating(row_data.get('Rating'))
        if rating is not None:
            metadata['rating'] = rating
        
        # Extract year and month from Year_Month
        year_month = str(row_data.get('Year_Month', ''))
        year = self.extract_year(year_month)
        month = self.extract_month(year_month)
        
        if year is not None:
            metadata['year'] = year
        if month is not None:
            metadata['month'] = month
        
        # Extract branch
        branch = self.extract_branch(row_data.get('Branch'))
        metadata['branch'] = branch
        
        # Extract review ID
        review_id = self.extract_review_id(row_data.get('Review_ID'))
        metadata['review_id'] = review_id
        
        # Extract reviewer location
        location = self.extract_reviewer_location(row_data.get('Reviewer_Location'))
        if location is not None:
            metadata['reviewer_location'] = location
        
        return metadata
    
    def get_extraction_stats(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about metadata extraction.
        
        Args:
            metadata_list: List of metadata dictionaries
            
        Returns:
            Dictionary containing extraction statistics
        """
        if not metadata_list:
            return {}
        
        stats = {
            'total_records': len(metadata_list),
            'fields_present': {},
            'field_counts': {}
        }
        
        # Count presence of each field
        for metadata in metadata_list:
            for field, value in metadata.items():
                if field not in stats['fields_present']:
                    stats['fields_present'][field] = 0
                if value is not None and value != '':
                    stats['fields_present'][field] += 1
                
                # Count specific values for categorical fields
                if field in ['rating', 'year', 'month', 'branch']:
                    if field not in stats['field_counts']:
                        stats['field_counts'][field] = {}
                    if value is not None:
                        if value not in stats['field_counts'][field]:
                            stats['field_counts'][field][value] = 0
                        stats['field_counts'][field][value] += 1
        
        # Calculate percentages
        for field in stats['fields_present']:
            count = stats['fields_present'][field]
            stats['fields_present'][field] = {
                'count': count,
                'percentage': round((count / len(metadata_list)) * 100, 2)
            }
        
        return stats
