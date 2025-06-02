"""
Setup configuration for the cognitive-computing package.

This file defines the package metadata and dependencies for pip installation.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from version.py
version = {}
with open(os.path.join("cognitive_computing", "version.py")) as fp:
    exec(fp.read(), version)

setup(
    name="cognitive-computing",
    version=version["__version__"],
    author="Cognitive Computing Contributors",
    author_email="contact@cognitive-computing.org",
    description="A comprehensive Python package for cognitive computing including Sparse Distributed Memory, Vector Symbolic Architectures, and Hyperdimensional Computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cognitive-computing/cognitive-computing",
    project_urls={
        "Bug Tracker": "https://github.com/cognitive-computing/cognitive-computing/issues",
        "Documentation": "https://cognitive-computing.readthedocs.io",
        "Source Code": "https://github.com/cognitive-computing/cognitive-computing",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "docs", "docs.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
        "networkx>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.990",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "graphviz>=0.19.0",
        ],
        "gpu": [
            "cupy>=10.0.0",
            "torch>=1.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cognitive-computing=cognitive_computing.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="cognitive computing, sparse distributed memory, vector symbolic architecture, hyperdimensional computing, SDM, VSA, HDC",
)