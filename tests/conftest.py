"""
Pytest configuration and shared fixtures for cognitive computing tests.

This file is automatically loaded by pytest and provides:
- Global test configuration
- Shared fixtures available to all tests
- Custom markers for test categorization
- Test environment setup
"""

import pytest
import numpy as np
import matplotlib
import logging
import warnings
from typing import Generator
import tempfile
import os

# Configure matplotlib to use non-interactive backend for tests
matplotlib.use('Agg')


# Configure logging for tests
@pytest.fixture(autouse=True)
def configure_test_logging():
    """Configure logging for test runs."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress some verbose libraries during tests
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


# Custom markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


# Global fixtures
@pytest.fixture
def random_seed():
    """Provide a fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def rng(random_seed):
    """Provide a seeded random number generator."""
    return np.random.RandomState(random_seed)


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def binary_pattern_1d(rng):
    """Generate a random 1D binary pattern."""
    def _generate(size: int) -> np.ndarray:
        return rng.randint(0, 2, size=size, dtype=np.uint8)
    return _generate


@pytest.fixture
def binary_pattern_2d(rng):
    """Generate a random 2D binary pattern."""
    def _generate(shape: tuple) -> np.ndarray:
        return rng.randint(0, 2, size=shape, dtype=np.uint8)
    return _generate


@pytest.fixture
def continuous_pattern(rng):
    """Generate a random continuous pattern."""
    def _generate(size: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
        return rng.uniform(low, high, size=size)
    return _generate


# Test environment setup
@pytest.fixture(scope="session", autouse=True)
def test_environment_setup():
    """Set up test environment once per session."""
    # Disable warnings during tests unless explicitly testing warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    
    # Set numpy error handling
    np.seterr(all='raise', under='ignore')
    
    yield
    
    # Cleanup can go here if needed


# Performance tracking
@pytest.fixture
def benchmark_timer():
    """Simple timer for benchmarking."""
    import time
    
    class Timer:
        def __init__(self):
            self.times = []
        
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.end = time.perf_counter()
            self.times.append(self.end - self.start)
        
        @property
        def last_time(self):
            return self.times[-1] if self.times else 0.0
        
        @property
        def mean_time(self):
            return np.mean(self.times) if self.times else 0.0
    
    return Timer()


# Plotting cleanup
@pytest.fixture(autouse=True)
def cleanup_plots():
    """Ensure all matplotlib plots are closed after each test."""
    yield
    import matplotlib.pyplot as plt
    plt.close('all')


# Skip markers for optional dependencies
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip tests with missing dependencies."""
    
    # Check for optional dependencies
    try:
        import plotly
        has_plotly = True
    except ImportError:
        has_plotly = False
    
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
    
    try:
        import cupy
        has_cupy = True
    except ImportError:
        has_cupy = False
    
    skip_plotly = pytest.mark.skip(reason="plotly not installed")
    skip_torch = pytest.mark.skip(reason="torch not installed")
    skip_cupy = pytest.mark.skip(reason="cupy not installed")
    
    for item in items:
        # Skip plotly tests if not installed
        if "plotly" in item.nodeid and not has_plotly:
            item.add_marker(skip_plotly)
        
        # Skip GPU tests if no GPU libraries
        if item.get_closest_marker("gpu"):
            if not has_torch and not has_cupy:
                item.add_marker(pytest.mark.skip(reason="No GPU libraries installed"))


# Hypothesis strategies (if using property-based testing)
try:
    from hypothesis import strategies as st
    
    @pytest.fixture
    def binary_array_strategy():
        """Hypothesis strategy for binary arrays."""
        return st.lists(
            st.integers(0, 1),
            min_size=10,
            max_size=1000
        ).map(lambda x: np.array(x, dtype=np.uint8))
    
except ImportError:
    # Hypothesis not installed, skip strategy fixtures
    pass