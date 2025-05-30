# WSL Setup Instructions for Cognitive Computing Project

## Current Status
- ✅ Python 3.12.3 is installed
- ❌ pip (package manager) is missing
- ❌ python3-venv (virtual environment) is missing
- ❌ Development tools are missing

## Required Setup Steps

### 1. Install Essential Python Tools
Run these commands in your WSL terminal:

```bash
# Update package list and install Python development tools
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev build-essential

# Verify installation
python3 --version
pip3 --version
```

### 2. Create Virtual Environment
```bash
# Navigate to project directory
cd /mnt/c/Users/Ian/Documents/Projects/cognitive-computing

# Create virtual environment test asdf
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (prompt should show (venv))
which python
```

### 3. Install Project Dependencies
```bash
# Install core dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import cognitive_computing; print('✅ Package installed successfully')"
```

### 4. Run Tests to Verify Phase 1
```bash
# Run all SDM core tests
pytest tests/test_sdm/test_core.py -v

# Run with coverage
pytest tests/test_sdm/test_core.py --cov=cognitive_computing.sdm.core

# Run quick tests (skip slow ones)
pytest tests/test_sdm/test_core.py -m "not slow" -v
```

## Alternative: Quick Commands
If you want to run everything at once:

```bash
# Install system packages (requires password)
sudo apt update && sudo apt install -y python3-pip python3-venv python3-dev build-essential

# Set up project environment
cd /mnt/c/Users/Ian/Documents/Projects/cognitive-computing
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/test_sdm/test_core.py -v
```

## Troubleshooting

### If pip install fails:
```bash
# Update pip first
pip install --upgrade pip setuptools wheel
```

### If tests have import errors:
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall in development mode
pip install -e .
```

### To deactivate virtual environment when done:
```bash
deactivate
```