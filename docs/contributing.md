# Contributing to Cognitive Computing

Thank you for your interest in contributing to the Cognitive Computing package! This document provides guidelines and instructions for contributing.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Development Process](#development-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)
- [Adding New Features](#adding-new-features)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow:

- **Be respectful and inclusive**: Welcome newcomers and treat everyone with respect
- **Be patient**: Remember that everyone was new once
- **Be constructive**: Provide helpful feedback and suggestions
- **Be collaborative**: Work together to solve problems
- **Be mindful**: Consider how your contributions affect others

## How Can I Contribute?

### Types of Contributions

#### 1. Report Bugs
Report bugs by [opening a new issue](https://github.com/cognitive-computing/cognitive-computing/issues) with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment details (OS, Python version, package version)
- Code snippets or error messages

#### 2. Suggest Enhancements
Enhancement suggestions are welcome! Open an issue with:
- Use case description
- Proposed solution
- Alternative solutions considered
- Mockups or examples if applicable

#### 3. Write Code
- Fix bugs (look for issues labeled `bug`)
- Implement new features (check issues labeled `enhancement`)
- Improve performance
- Add new cognitive computing paradigms

#### 4. Improve Documentation
- Fix typos or clarify existing docs
- Add examples and tutorials
- Improve API documentation
- Translate documentation

#### 5. Write Tests
- Increase test coverage
- Add edge case tests
- Improve test performance
- Add integration tests

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/cognitive-computing.git
cd cognitive-computing
git remote add upstream https://github.com/cognitive-computing/cognitive-computing.git
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Development Dependencies

```bash
# Install in editable mode with all dependencies
pip install -e ".[dev,viz]"

# Install pre-commit hooks
pre-commit install
```

### 4. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

## Development Process

### 1. Make Changes

- Write clean, documented code
- Follow the style guidelines
- Add tests for new functionality
- Update documentation as needed

### 2. Run Tests Locally

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_sdm/test_core.py

# Run with coverage
pytest --cov=cognitive_computing --cov-report=html

# Run only fast tests
pytest -m "not slow"
```

### 3. Check Code Quality

```bash
# Format code with black
black cognitive_computing tests

# Check linting
flake8 cognitive_computing tests

# Type checking
mypy cognitive_computing

# Run all checks
make check  # If Makefile is available
```

### 4. Update Documentation

```bash
# Build documentation locally
cd docs
make html
# View at docs/_build/html/index.html
```

## Code Style Guidelines

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these tools:
- **Black** for formatting (line length: 88)
- **Flake8** for linting
- **MyPy** for type checking

### Key Conventions

#### 1. Imports
```python
# Standard library
import os
import sys
from typing import List, Optional

# Third-party
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from cognitive_computing.common.base import CognitiveMemory
```

#### 2. Docstrings
Use NumPy-style docstrings:

```python
def compute_similarity(pattern1: np.ndarray, pattern2: np.ndarray) -> float:
    """
    Compute similarity between two binary patterns.
    
    Parameters
    ----------
    pattern1 : np.ndarray
        First binary pattern
    pattern2 : np.ndarray
        Second binary pattern
        
    Returns
    -------
    float
        Similarity score between 0 and 1
        
    Examples
    --------
    >>> p1 = np.array([1, 0, 1, 0])
    >>> p2 = np.array([1, 0, 0, 1])
    >>> compute_similarity(p1, p2)
    0.5
    """
```

#### 3. Type Hints
Always use type hints:

```python
from typing import List, Optional, Tuple, Union

def process_data(
    data: np.ndarray,
    threshold: float = 0.5,
    return_indices: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
    ...
```

#### 4. Class Design
```python
class NewDecoder(AddressDecoder):
    """One-line summary.
    
    Longer description of the decoder and its purpose.
    
    Parameters
    ----------
    config : DecoderConfig
        Configuration parameters
        
    Attributes
    ----------
    some_attribute : type
        Description
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        self._private_attribute = None
        
    def public_method(self) -> None:
        """Public methods have docstrings."""
        pass
        
    def _private_method(self) -> None:
        # Private methods use comments
        pass
```

### Naming Conventions

- **Classes**: `CamelCase`
- **Functions/Methods**: `snake_case`
- **Constants**: `UPPER_CASE`
- **Private**: `_leading_underscore`

## Testing Guidelines

### Test Structure

```python
import pytest
import numpy as np
from cognitive_computing.sdm import SDM

class TestSDMFeature:
    """Group related tests in classes."""
    
    @pytest.fixture
    def sample_sdm(self):
        """Provide test fixture."""
        return SDM(SDMConfig(dimension=256))
    
    def test_basic_functionality(self, sample_sdm):
        """Test names should be descriptive."""
        # Arrange
        pattern = np.random.randint(0, 2, 256)
        
        # Act
        sample_sdm.store(pattern, pattern)
        
        # Assert
        recalled = sample_sdm.recall(pattern)
        assert np.array_equal(recalled, pattern)
    
    def test_edge_case(self):
        """Test edge cases and error conditions."""
        with pytest.raises(ValueError, match="Invalid dimension"):
            SDM(SDMConfig(dimension=0))
```

### Testing Requirements

- All new features must have tests
- Maintain or increase code coverage (aim for >90%)
- Include both unit and integration tests
- Test edge cases and error conditions
- Use meaningful test names

### Performance Tests

```python
@pytest.mark.slow
def test_large_scale_performance():
    """Mark slow tests that can be skipped."""
    # Test with large parameters
    pass

@pytest.mark.benchmark
def test_operation_speed(benchmark):
    """Use pytest-benchmark for performance tests."""
    result = benchmark(function_to_test, arg1, arg2)
```

## Documentation Guidelines

### Code Documentation

1. **All public APIs** must have docstrings
2. **Complex algorithms** should have explanatory comments
3. **Mathematical formulas** should include references

### Documentation Files

When adding new features:

1. Update relevant `.md` files in `docs/`
2. Add examples to example scripts
3. Update the README if needed
4. Add to API reference

### Example Documentation

```python
def new_feature(data: np.ndarray) -> np.ndarray:
    """
    Short description of the feature.
    
    Longer explanation of what the feature does, why it's useful,
    and any important details users should know.
    
    .. math::
        f(x) = \\sum_{i=1}^{n} x_i^2
        
    Parameters
    ----------
    data : np.ndarray
        Description of the parameter
        
    Returns
    -------
    np.ndarray
        Description of the return value
        
    See Also
    --------
    related_function : Brief description
    
    References
    ----------
    .. [1] Author, "Paper Title," Journal, 2023.
    
    Examples
    --------
    >>> data = np.array([1, 2, 3, 4])
    >>> result = new_feature(data)
    >>> print(result)
    [1 4 9 16]
    """
```

## Pull Request Process

### 1. Before Submitting

- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### 2. PR Title and Description

**Title Format**: `[TYPE] Brief description`

Types:
- `[FEAT]` - New feature
- `[FIX]` - Bug fix
- `[DOCS]` - Documentation only
- `[TEST]` - Test only
- `[PERF]` - Performance improvement
- `[REFACTOR]` - Code refactoring

**Description Template**:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] New tests added
- [ ] Coverage maintained/increased

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### 3. Code Review

- Respond to feedback constructively
- Make requested changes
- Re-request review when ready

### 4. After Merge

- Delete your feature branch
- Update your local main branch
- Celebrate! 🎉

## Project Structure

Understanding the project structure helps when contributing:

```
cognitive-computing/
├── cognitive_computing/
│   ├── common/          # Shared base classes
│   ├── sdm/            # Sparse Distributed Memory
│   ├── hrr/            # Holographic Reduced Representations (future)
│   ├── vsa/            # Vector Symbolic Architectures (future)
│   └── hdc/            # Hyperdimensional Computing (future)
├── tests/              # Test suite
├── docs/               # Documentation
├── examples/           # Example scripts
└── benchmarks/         # Performance benchmarks
```

## Adding New Features

### Adding a New Decoder

1. Create file: `cognitive_computing/sdm/decoders/new_decoder.py`
2. Implement the decoder:
```python
from cognitive_computing.sdm.address_decoder import AddressDecoder

class NewDecoder(AddressDecoder):
    def decode(self, address: np.ndarray) -> np.ndarray:
        # Implementation
        pass
```

3. Add tests: `tests/test_sdm/test_new_decoder.py`
4. Update `__init__.py` to export the decoder
5. Add documentation and examples

### Adding a New Cognitive Paradigm

1. Create new module: `cognitive_computing/new_paradigm/`
2. Implement base classes following existing patterns
3. Add comprehensive tests
4. Create documentation
5. Add examples

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist

1. Update version in `cognitive_computing/version.py`
2. Update CHANGELOG.md
3. Run full test suite
4. Build and test documentation
5. Create release PR
6. Tag release after merge
7. Build and upload to PyPI

## Getting Help

- Check existing issues and PRs
- Read the documentation
- Ask questions in issues (label: `question`)
- Join discussions in the community

## Recognition

Contributors are recognized in:
- The AUTHORS file
- Release notes
- Project documentation

Thank you for contributing to Cognitive Computing! Your efforts help advance the field of brain-inspired computing.