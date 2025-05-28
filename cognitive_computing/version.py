"""
Version information for cognitive-computing package.

This file is used by setup.py and the package to maintain version consistency.
"""

__version__ = "0.1.0"
__version_info__ = tuple(int(i) for i in __version__.split("."))

# Development status
__status__ = "Alpha"

# Package metadata
__title__ = "cognitive-computing"
__description__ = "A comprehensive Python package for cognitive computing"
__url__ = "https://github.com/cognitive-computing/cognitive-computing"
__author__ = "Cognitive Computing Contributors"
__author_email__ = "contact@cognitive-computing.org"
__license__ = "MIT"
__copyright__ = "Copyright 2024 Cognitive Computing Contributors"

# Supported Python versions
__python_requires__ = ">=3.8"

# Module information
__all__ = [
    "__version__",
    "__version_info__",
    "__status__",
    "__title__",
    "__description__",
    "__url__",
    "__author__",
    "__author_email__",
    "__license__",
    "__copyright__",
    "__python_requires__",
]