"""Logging configuration for Disney AI services."""

import logging
import sys
from typing import Optional

from .config import settings


def setup_logging(service_name: str, log_level: Optional[str] = None) -> logging.Logger:
    """Set up structured logging for a service.
    
    Args:
        service_name: Name of the service for log identification
        log_level: Optional log level override
        
    Returns:
        Configured logger instance
    """
    level = log_level or settings.log_level
    
    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger
