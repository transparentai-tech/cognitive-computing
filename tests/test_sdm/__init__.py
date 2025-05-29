"""
Test suite for Sparse Distributed Memory (SDM) module.

This package contains comprehensive tests for all SDM components:
- Core SDM implementation (test_core.py)
- Memory components (test_memory.py)
- Address decoders (test_address_decoder.py)
- Utility functions (test_utils.py)
- Visualization tools (test_visualizations.py)

The tests cover:
- Unit tests for individual components
- Integration tests for component interactions
- Performance benchmarks
- Edge cases and error handling
"""

# Common test fixtures and utilities can be imported here
# and made available to all SDM tests

import pytest
import numpy as np
from cognitive_computing.sdm import SDM, SDMConfig


@pytest.fixture
def basic_sdm_config():
    """Provide a basic SDM configuration for tests."""
    return SDMConfig(
        dimension=256,
        num_hard_locations=100,
        activation_radius=115,
        seed=42
    )


@pytest.fixture
def small_sdm():
    """Provide a small SDM instance for quick tests."""
    config = SDMConfig(
        dimension=128,
        num_hard_locations=50,
        activation_radius=57,
        seed=42
    )
    return SDM(config)


# Export common fixtures
__all__ = [
    'basic_sdm_config',
    'small_sdm'
]