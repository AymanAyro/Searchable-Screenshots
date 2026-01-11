"""Logging configuration and utilities for Searchable Screenshots."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import sys


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    console_output: bool = True,
) -> logging.Logger:
    """Setup application-wide logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file (enables file logging with rotation)
        console_output: Whether to output logs to console
        
    Returns:
        Configured root logger
    """
    # Get root logger
    logger = logging.getLogger("searchable_screenshots")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__ of the module)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"searchable_screenshots.{name}")


class TimingContext:
    """Context manager for timing operations and logging."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        """Initialize timing context.
        
        Args:
            logger: Logger instance
            operation: Description of the operation being timed
        """
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} (took {elapsed:.2f}s)")
        else:
            self.logger.error(f"Failed: {self.operation} (took {elapsed:.2f}s): {exc_val}")
        return False  # Don't suppress exceptions
