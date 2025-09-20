#!/usr/bin/env python3
"""Data pipeline runner script."""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from disney.pipeline.ingest import get_ingester
from disney.shared.logging import setup_logging

logger = setup_logging("data-pipeline-runner")


async def main():
    """Main pipeline execution function."""
    try:
        logger.info("Starting Disney reviews data pipeline")
        
        # Get ingester instance
        ingester = get_ingester()
        
        # Run ingestion
        success = await ingester.run_ingestion()
        
        if success:
            logger.info("Data pipeline completed successfully")
            sys.exit(0)
        else:
            logger.error("Data pipeline failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Data pipeline error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
