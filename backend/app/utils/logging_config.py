"""
Structured logging configuration for the application.

This module provides a centralized logging setup with structured output,
making it easier to monitor and debug the application.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname:8}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_colors: bool = True
):
    """
    Setup structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        enable_colors: Enable colored output for console

    Example:
        setup_logging(log_level="DEBUG", log_file=Path("logs/app.log"))
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler with UTF-8 encoding support for Windows
    # Use a TextIOWrapper to ensure UTF-8 encoding on all platforms
    import io
    if hasattr(sys.stdout, 'buffer'):
        # For Windows, wrap stdout buffer with UTF-8 encoding
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding='utf-8',
            errors='replace',  # Replace unencodable characters instead of failing
            line_buffering=True
        )
    else:
        utf8_stdout = sys.stdout

    console_handler = logging.StreamHandler(utf8_stdout)
    console_handler.setLevel(numeric_level)

    if enable_colors:
        console_format = ColoredFormatter(
            fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)

        file_format = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

    # Reduce noise from external libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    # pdfminer's LZW/CCITT decoders emit a DEBUG line per code — on large PDFs this
    # produces millions of lines and can exhaust memory. Keep these quiet always.
    for _noisy in ("pdfminer", "pdfplumber", "PIL", "fontTools", "fitz"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Name of the module (usually __name__)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Application started")
    """
    return logging.getLogger(name)


# Context manager for logging function execution
class LogExecutionTime:
    """Context manager to log function execution time."""

    def __init__(self, logger: logging.Logger, operation: str):
        """
        Initialize the context manager.

        Args:
            logger: Logger instance
            operation: Description of the operation
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
            self.logger.info(
                f"Completed: {self.operation} (took {elapsed:.3f}s)"
            )
        else:
            self.logger.error(
                f"Failed: {self.operation} (took {elapsed:.3f}s) - "
                f"{exc_type.__name__}: {exc_val}"
            )

        return False  # Don't suppress exceptions


# Convenience function for logging with context
def log_operation(logger: logging.Logger, operation: str):
    """
    Decorator/context manager for logging operations.

    Can be used as a decorator or context manager:

    As context manager:
        with log_operation(logger, "processing document"):
            # do work
            pass

    Args:
        logger: Logger instance
        operation: Description of the operation

    Returns:
        LogExecutionTime context manager
    """
    return LogExecutionTime(logger, operation)
