"""
Cognitive Computing Package

A comprehensive Python package for cognitive computing including:
- Sparse Distributed Memory (SDM)
- Holographic Reduced Representations (HRR)
- Vector Symbolic Architectures (VSA) [Coming Soon]
- Hyperdimensional Computing (HDC) [Coming Soon]

This package provides implementations of various cognitive computing paradigms
that enable robust, efficient, and brain-inspired computing methods.
"""

from cognitive_computing.version import (
    __version__,
    __version_info__,
    __status__,
    __author__,
    __author_email__,
    __license__,
)

# Import main modules when they're available
try:
    from cognitive_computing import sdm
except ImportError:
    pass

try:
    from cognitive_computing import hrr
except ImportError:
    pass

# Define what's available when using "from cognitive_computing import *"
__all__ = [
    "sdm",  # Sparse Distributed Memory
    "hrr",  # Holographic Reduced Representations
    # Future modules:
    # "vsa",  # Vector Symbolic Architectures
    # "hdc",  # Hyperdimensional Computing
    "__version__",
    "__version_info__",
    "__status__",
    "__author__",
    "__author_email__",
    "__license__",
]

# Package-level configuration
import logging

# Set up package logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def configure_logging(level=logging.INFO, format=None):
    """
    Configure logging for the cognitive computing package.
    
    Parameters
    ----------
    level : int, optional
        Logging level (default: logging.INFO)
    format : str, optional
        Logging format string. If None, uses default format.
    """
    if format is None:
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(level=level, format=format)
    logger.setLevel(level)

# Display package info
def info():
    """Display package information."""
    info_str = f"""
Cognitive Computing Package v{__version__}
Status: {__status__}
Author: {__author__}
License: {__license__}

Available modules:
- sdm: Sparse Distributed Memory
- hrr: Holographic Reduced Representations

For more information, visit:
https://github.com/cognitive-computing/cognitive-computing
    """
    print(info_str)