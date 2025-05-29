# Installation Guide

This guide covers the installation of the `cognitive-computing` package and its dependencies.

## Table of Contents
- [Requirements](#requirements)
- [Quick Install](#quick-install)
- [Development Installation](#development-installation)
- [Optional Dependencies](#optional-dependencies)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Verifying Installation](#verifying-installation)
- [Troubleshooting](#troubleshooting)
- [Uninstallation](#uninstallation)

## Requirements

### Python Version
- Python 3.8 or higher is required
- Python 3.9-3.11 recommended for best compatibility

### Core Dependencies
The following packages are automatically installed:
- `numpy >= 1.21.0` - Numerical computing
- `scipy >= 1.7.0` - Scientific computing
- `scikit-learn >= 1.0.0` - Machine learning utilities
- `matplotlib >= 3.4.0` - Plotting and visualization
- `seaborn >= 0.11.0` - Statistical visualization
- `tqdm >= 4.62.0` - Progress bars
- `joblib >= 1.1.0` - Parallel processing

### System Requirements
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large-scale SDM)
- **Storage**: ~100MB for package and dependencies
- **OS**: Windows, macOS, or Linux

## Quick Install

### From PyPI (When Published)
```bash
pip install cognitive-computing
```

### From Source (Current Method)
```bash
# Clone the repository
git clone https://github.com/cognitive-computing/cognitive-computing.git
cd cognitive-computing

# Install in standard mode
pip install .
```

### Editable Installation (Recommended for Development)
```bash
# Install in editable/development mode
pip install -e .
```

## Development Installation

For contributing to the package or running tests:

```bash
# Clone the repository
git clone https://github.com/cognitive-computing/cognitive-computing.git
cd cognitive-computing

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install with all development dependencies
pip install -e ".[dev]"
```

### Development Dependencies
The `[dev]` extra includes:
- `pytest >= 7.0.0` - Testing framework
- `pytest-cov >= 3.0.0` - Test coverage
- `black >= 22.0.0` - Code formatting
- `flake8 >= 4.0.0` - Linting
- `mypy >= 0.990` - Type checking
- `sphinx >= 4.0.0` - Documentation
- `sphinx-rtd-theme >= 1.0.0` - Documentation theme
- `jupyter >= 1.0.0` - Jupyter notebooks
- `ipython >= 8.0.0` - Interactive Python

## Optional Dependencies

### Visualization Enhancement
For additional visualization capabilities:

```bash
pip install -e ".[viz]"
```

Includes:
- `plotly >= 5.0.0` - Interactive 3D visualizations
- `networkx >= 2.6.0` - Network analysis
- `graphviz >= 0.19.0` - Graph visualization

### GPU Acceleration
For GPU-accelerated operations:

```bash
pip install -e ".[gpu]"
```

Includes:
- `cupy >= 10.0.0` - CUDA acceleration
- `torch >= 1.10.0` - PyTorch for neural operations

**Note**: GPU dependencies require appropriate CUDA installation.

### All Optional Dependencies
```bash
pip install -e ".[dev,viz,gpu]"
```

## Platform-Specific Instructions

### Windows

1. **Install Python**: Download from [python.org](https://www.python.org/downloads/)
2. **Install Visual C++ Build Tools** (for some dependencies):
   ```bash
   # May be required for packages like scipy
   # Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   ```
3. **Install package**:
   ```bash
   pip install cognitive-computing
   ```

### macOS

1. **Install Python** (if not using system Python):
   ```bash
   # Using Homebrew
   brew install python@3.9
   ```
2. **Install package**:
   ```bash
   pip install cognitive-computing
   ```

### Linux (Ubuntu/Debian)

1. **Install Python and dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install python3.9 python3.9-dev python3-pip
   ```
2. **Install package**:
   ```bash
   pip install cognitive-computing
   ```

### Conda Environment

If using Anaconda or Miniconda:

```bash
# Create new environment
conda create -n cognitive python=3.9

# Activate environment
conda activate cognitive

# Install package
pip install cognitive-computing

# Or install some dependencies via conda
conda install numpy scipy matplotlib scikit-learn
pip install cognitive-computing
```

## Verifying Installation

### Basic Verification
```python
# Test basic import
import cognitive_computing
print(cognitive_computing.__version__)

# Test SDM import
from cognitive_computing.sdm import create_sdm
sdm = create_sdm(dimension=1000)
print(f"Created SDM with {sdm.config.num_hard_locations} locations")
```

### Run Test Suite
```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_sdm/

# Run with coverage report
pytest --cov=cognitive_computing --cov-report=html
```

### Check Optional Dependencies
```python
# Check visualization support
try:
    import plotly
    print("Plotly installed: Interactive visualizations available")
except ImportError:
    print("Plotly not installed: Install with pip install cognitive-computing[viz]")

# Check GPU support
try:
    import cupy
    print("CuPy installed: GPU acceleration available")
except ImportError:
    print("CuPy not installed: Install with pip install cognitive-computing[gpu]")
```

## Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'cognitive_computing'
**Solution**: Ensure you're in the correct directory and the package is installed:
```bash
pip list | grep cognitive-computing
```

#### 2. NumPy Version Conflicts
**Solution**: Update NumPy to the required version:
```bash
pip install --upgrade numpy>=1.21.0
```

#### 3. Memory Errors with Large SDMs
**Solution**: Reduce the number of hard locations or use a machine with more RAM:
```python
# Instead of
sdm = create_sdm(dimension=10000)

# Use smaller configuration
from cognitive_computing.sdm import SDMConfig, SDM
config = SDMConfig(dimension=5000, num_hard_locations=1000)
sdm = SDM(config)
```

#### 4. Matplotlib Backend Issues
**Solution**: Set the backend explicitly:
```python
import matplotlib
matplotlib.use('Agg')  # For headless environments
# or
matplotlib.use('TkAgg')  # For interactive use
```

#### 5. Permission Errors
**Solution**: Use `--user` flag or virtual environment:
```bash
pip install --user cognitive-computing
```

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/cognitive-computing/cognitive-computing/issues)
2. Search the documentation
3. Create a new issue with:
   - Python version: `python --version`
   - Package version: `pip show cognitive-computing`
   - Error message and traceback
   - Minimal reproducible example

## Uninstallation

### Remove Package
```bash
pip uninstall cognitive-computing
```

### Remove All Dependencies
```bash
# Save current environment
pip freeze > requirements-backup.txt

# Uninstall package and dependencies
pip uninstall cognitive-computing
pip uninstall -r requirements.txt
```

### Clean Cache
```bash
# Clear pip cache
pip cache purge

# Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete
```

## Next Steps

After installation:
- Read the [Quick Start Guide](../README.md#quick-start)
- Explore the [SDM Overview](sdm/overview.md)
- Try the [Examples](sdm/examples.md)
- Check the [API Reference](sdm/api_reference.md)

## Development Setup

For contributors:

1. Fork the repository
2. Clone your fork
3. Install in development mode with all extras
4. Create a feature branch
5. Make changes and add tests
6. Run tests and linting
7. Submit a pull request

See [Contributing Guide](contributing.md) for detailed instructions.