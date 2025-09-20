#!/usr/bin/env python3
"""Data migration utilities script."""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from disney.shared.logging import setup_logging

logger = setup_logging("data-migration")


def validate_data_file(file_path: str) -> bool:
    """Validate the Disney reviews data file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        logger.info(f"Validating data file: {file_path}")
        
        # Check if file exists
        if not Path(file_path).exists():
            logger.error(f"Data file not found: {file_path}")
            return False
        
        # Load and validate CSV
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_columns = ['Review_Text', 'Rating', 'Year_Month', 'Branch']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check data quality
        total_reviews = len(df)
        empty_reviews = df['Review_Text'].isna().sum()
        valid_reviews = total_reviews - empty_reviews
        
        logger.info(f"Data validation results:")
        logger.info(f"  Total reviews: {total_reviews}")
        logger.info(f"  Valid reviews: {valid_reviews}")
        logger.info(f"  Empty reviews: {empty_reviews}")
        
        if valid_reviews < 100:
            logger.warning("Very few valid reviews found")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating data file: {str(e)}")
        return False


def analyze_data(file_path: str):
    """Analyze the Disney reviews data.
    
    Args:
        file_path: Path to the data file
    """
    try:
        logger.info(f"Analyzing data file: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Basic statistics
        print("\n📊 Data Analysis Report")
        print("=" * 50)
        print(f"Total reviews: {len(df)}")
        print(f"Date range: {df['Year_Month'].min()} to {df['Year_Month'].max()}")
        print(f"Branches: {df['Branch'].unique().tolist()}")
        print(f"Rating distribution:")
        print(df['Rating'].value_counts().sort_index())
        
        # Review length analysis
        df['review_length'] = df['Review_Text'].str.len()
        print(f"\nReview length statistics:")
        print(f"  Mean: {df['review_length'].mean():.1f} characters")
        print(f"  Median: {df['review_length'].median():.1f} characters")
        print(f"  Min: {df['review_length'].min()} characters")
        print(f"  Max: {df['review_length'].max()} characters")
        
        # Sample reviews
        print(f"\n📝 Sample reviews:")
        for i, (_, row) in enumerate(df.head(3).iterrows()):
            print(f"\n{i+1}. Rating: {row['Rating']}, Branch: {row['Branch']}")
            print(f"   {row['Review_Text'][:200]}...")
        
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")


def main():
    """Main migration function."""
    if len(sys.argv) < 2:
        print("Usage: python migrate_data.py <command> [file_path]")
        print("Commands:")
        print("  validate <file_path>  - Validate data file")
        print("  analyze <file_path>   - Analyze data file")
        sys.exit(1)
    
    command = sys.argv[1]
    file_path = sys.argv[2] if len(sys.argv) > 2 else "data/DisneylandReviews.csv"
    
    if command == "validate":
        success = validate_data_file(file_path)
        sys.exit(0 if success else 1)
    elif command == "analyze":
        analyze_data(file_path)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
