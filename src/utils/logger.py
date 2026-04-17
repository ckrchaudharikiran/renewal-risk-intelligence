"""Logging helpers for project modules."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Create or return a configured logger instance."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
