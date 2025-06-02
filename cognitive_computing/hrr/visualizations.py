"""
Visualization functions for HRR.

This module provides functions for visualizing HRR operations, including
similarity matrices, binding accuracy, and convolution spectra.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

# Optional imports
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .core import HRR
from .operations import CircularConvolution
from .cleanup import CleanupMemory

logger = logging.getLogger(__name__)


def plot_similarity_matrix(vectors: Dict[str, np.ndarray],
                          method: str = "cosine",
                          figsize: Tuple[int, int] = (8, 6),
                          cmap: str = "coolwarm",
                          annot: bool = True) -> Figure:
    """
    Plot similarity matrix between vectors.
    
    Parameters
    ----------
    vectors : Dict[str, np.ndarray]
        Dictionary mapping names to vectors
    method : str
        Similarity method: "cosine" or "dot"
    figsize : Tuple[int, int]
        Figure size
    cmap : str
        Colormap name
    annot : bool
        Whether to annotate cells with values
        
    Returns
    -------
    Figure
        Matplotlib figure
    """
    # Get vector names and create matrix
    names = list(vectors.keys())
    n = len(names)
    similarity_matrix = np.zeros((n, n))
    
    # Compute similarities
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            v1 = vectors[name1]
            v2 = vectors[name2]
            
            if method == "cosine":
                # Cosine similarity
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            else:  # dot
                sim = np.dot(v1, v2)
            
            similarity_matrix[i, j] = sim
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use seaborn heatmap
    sns.heatmap(similarity_matrix, 
                xticklabels=names,
                yticklabels=names,
                cmap=cmap,
                center=0,
                vmin=-1,
                vmax=1,
                annot=annot,
                fmt='.2f',
                ax=ax)
    
    ax.set_title(f"Vector Similarity Matrix ({method})")
    plt.tight_layout()
    
    return fig


def plot_binding_accuracy(hrr: HRR, test_results: Dict[str, Any],
                         figsize: Tuple[int, int] = (10, 6)) -> Figure:
    """
    Plot binding accuracy results.
    
    Parameters
    ----------
    hrr : HRR
        HRR system used for testing
    test_results : Dict[str, Any]
        Results from analyze_binding_capacity
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Extract data
    n_pairs = test_results.get("n_pairs", [])
    accuracies = test_results.get("accuracies", [])
    similarities = test_results.get("mean_similarities", [])
    
    if not isinstance(n_pairs, list):
        # Convert single test to list format
        n_pairs = [n_pairs]
        accuracies = [test_results.get("mean_accuracy", 0)]
        similarities = [test_results.get("mean_similarity", 0)]
    
    # Plot accuracy vs number of pairs
    ax1.plot(n_pairs, accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, 
                label='90% threshold')
    ax1.set_xlabel("Number of Role-Filler Pairs")
    ax1.set_ylabel("Retrieval Accuracy")
    ax1.set_title("Binding Capacity")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    
    # Plot mean similarity vs number of pairs
    ax2.plot(n_pairs, similarities, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Role-Filler Pairs")
    ax2.set_ylabel("Mean Similarity")
    ax2.set_title("Retrieval Quality")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    fig.suptitle(f"HRR Binding Analysis (dimension={hrr.config.dimension})")
    plt.tight_layout()
    
    return fig


def visualize_cleanup_space(cleanup_memory: CleanupMemory,
                          method: str = "tsne",
                          figsize: Tuple[int, int] = (8, 8),
                          interactive: bool = False) -> Union[Figure, Any]:
    """
    Visualize the cleanup memory space in 2D.
    
    Parameters
    ----------
    cleanup_memory : CleanupMemory
        Cleanup memory to visualize
    method : str
        Dimensionality reduction method: "tsne", "pca", "mds"
    figsize : Tuple[int, int]
        Figure size
    interactive : bool
        Whether to create interactive plot (requires plotly)
        
    Returns
    -------
    Figure or plotly figure
        Visualization figure
    """
    if cleanup_memory.size == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No items in cleanup memory", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Cleanup Memory Space")
        return fig
    
    # Get all vectors
    cleanup_memory._rebuild_matrix()
    vectors = cleanup_memory._item_matrix
    names = cleanup_memory._item_names
    
    # Dimensionality reduction
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "mds":
        from sklearn.manifold import MDS
        reducer = MDS(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Reduce dimensions
    coords_2d = reducer.fit_transform(vectors)
    
    if interactive and PLOTLY_AVAILABLE:
        # Create interactive plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode='markers+text',
            marker=dict(size=10, color='blue'),
            text=names,
            textposition="top center"
        ))
        
        fig.update_layout(
            title=f"Cleanup Memory Space ({method.upper()})",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            width=figsize[0] * 100,
            height=figsize[1] * 100
        )
        
        return fig
    else:
        # Create matplotlib plot
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=100, alpha=0.6)
        
        # Add labels
        for i, name in enumerate(names):
            ax.annotate(name, (coords_2d[i, 0], coords_2d[i, 1]),
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(f"Cleanup Memory Space ({method.upper()})")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def plot_convolution_spectrum(a: np.ndarray, b: np.ndarray, 
                            result: Optional[np.ndarray] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> Figure:
    """
    Plot frequency spectrum of convolution operation.
    
    Parameters
    ----------
    a, b : np.ndarray
        Input vectors
    result : np.ndarray, optional
        Convolution result (computed if not provided)
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Figure
        Matplotlib figure
    """
    # Compute convolution if not provided
    if result is None:
        result = CircularConvolution.convolve(a, b)
    
    # Compute FFTs
    fft_a = np.fft.fft(a)
    fft_b = np.fft.fft(b)
    fft_result = np.fft.fft(result)
    
    # Frequency axis
    n = len(a)
    freqs = np.fft.fftfreq(n, 1.0)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot magnitudes
    ax = axes[0, 0]
    ax.plot(freqs[:n//2], np.abs(fft_a[:n//2]), 'b-', label='Vector A')
    ax.plot(freqs[:n//2], np.abs(fft_b[:n//2]), 'r-', label='Vector B')
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    ax.set_title("Input Spectra")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot result spectrum
    ax = axes[0, 1]
    ax.plot(freqs[:n//2], np.abs(fft_result[:n//2]), 'g-', linewidth=2)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    ax.set_title("Convolution Result Spectrum")
    ax.grid(True, alpha=0.3)
    
    # Plot phases
    ax = axes[1, 0]
    ax.plot(freqs[:n//2], np.angle(fft_a[:n//2]), 'b-', label='Vector A')
    ax.plot(freqs[:n//2], np.angle(fft_b[:n//2]), 'r-', label='Vector B')
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Phase (radians)")
    ax.set_title("Input Phases")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot result phase
    ax = axes[1, 1]
    ax.plot(freqs[:n//2], np.angle(fft_result[:n//2]), 'g-', linewidth=2)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Phase (radians)")
    ax.set_title("Convolution Result Phase")
    ax.grid(True, alpha=0.3)
    
    fig.suptitle("Circular Convolution in Frequency Domain")
    plt.tight_layout()
    
    return fig


def animate_unbinding_process(hrr: HRR, composite: np.ndarray,
                            keys: List[np.ndarray],
                            names: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (10, 6)) -> Figure:
    """
    Visualize the unbinding process for multiple keys.
    
    Parameters
    ----------
    hrr : HRR
        HRR system
    composite : np.ndarray
        Composite vector containing multiple bindings
    keys : List[np.ndarray]
        Keys to unbind
    names : List[str], optional
        Names for the keys
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Figure
        Matplotlib figure showing unbinding results
    """
    if names is None:
        names = [f"Key {i+1}" for i in range(len(keys))]
    
    n_keys = len(keys)
    fig, axes = plt.subplots(2, n_keys, figsize=figsize)
    
    if n_keys == 1:
        axes = axes.reshape(2, 1)
    
    # Unbind with each key
    for i, (key, name) in enumerate(zip(keys, names)):
        retrieved = hrr.unbind(composite, key)
        
        # Plot retrieved vector
        ax = axes[0, i]
        ax.plot(retrieved[:100])  # Show first 100 elements
        ax.set_title(f"Unbind with {name}")
        ax.set_xlabel("Element")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        
        # Plot frequency spectrum
        ax = axes[1, i]
        fft = np.fft.fft(retrieved)
        freqs = np.fft.fftfreq(len(retrieved))
        ax.plot(freqs[:len(freqs)//2], np.abs(fft[:len(fft)//2]))
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Magnitude")
        ax.set_title("Spectrum")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Unbinding Process Visualization")
    plt.tight_layout()
    
    return fig


def plot_memory_capacity_curve(dimensions: List[int],
                             capacities: List[float],
                             theoretical: Optional[List[float]] = None,
                             figsize: Tuple[int, int] = (8, 6)) -> Figure:
    """
    Plot memory capacity as a function of dimension.
    
    Parameters
    ----------
    dimensions : List[int]
        Vector dimensions tested
    capacities : List[float]
        Measured capacities
    theoretical : List[float], optional
        Theoretical capacity values
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot measured capacity
    ax.plot(dimensions, capacities, 'bo-', linewidth=2, 
            markersize=8, label='Measured')
    
    # Plot theoretical if provided
    if theoretical is not None:
        ax.plot(dimensions, theoretical, 'r--', linewidth=2, 
                label='Theoretical')
    
    ax.set_xlabel("Vector Dimension")
    ax.set_ylabel("Capacity (# items)")
    ax.set_title("HRR Memory Capacity Scaling")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Log scale might be appropriate
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    return fig


def plot_crosstalk_analysis(n_vectors: List[int],
                          crosstalk_levels: List[float],
                          figsize: Tuple[int, int] = (8, 6)) -> Figure:
    """
    Plot crosstalk levels vs number of bundled vectors.
    
    Parameters
    ----------
    n_vectors : List[int]
        Number of vectors bundled
    crosstalk_levels : List[float]
        Measured crosstalk levels
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(n_vectors, crosstalk_levels, 'go-', linewidth=2, markersize=8)
    
    # Theoretical 1/sqrt(n) line
    theoretical = [1.0 / np.sqrt(n) for n in n_vectors]
    ax.plot(n_vectors, theoretical, 'r--', alpha=0.5, 
            label=r'$1/\sqrt{n}$ (theoretical)')
    
    ax.set_xlabel("Number of Bundled Vectors")
    ax.set_ylabel("Crosstalk Level")
    ax.set_title("Bundling Crosstalk Analysis")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    return fig


def create_performance_dashboard(perf_results: Dict[str, Any],
                               figsize: Tuple[int, int] = (12, 8)) -> Figure:
    """
    Create a dashboard showing HRR performance metrics.
    
    Parameters
    ----------
    perf_results : Dict[str, Any]
        Performance test results
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Figure
        Matplotlib figure with performance dashboard
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Operation times
    ax1 = fig.add_subplot(gs[0, 0])
    operations = ['Bind', 'Unbind', 'Bundle']
    times = [
        perf_results.get('bind_time_mean', 0) * 1000,
        perf_results.get('unbind_time_mean', 0) * 1000,
        perf_results.get('bundle_time_mean', 0) * 1000
    ]
    bars = ax1.bar(operations, times, color=['blue', 'green', 'orange'])
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Average Operation Times")
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}', ha='center', va='bottom')
    
    # Operations per second
    ax2 = fig.add_subplot(gs[0, 1])
    ops_per_sec = perf_results.get('operations_per_second', 0)
    ax2.text(0.5, 0.5, f'{ops_per_sec:.0f}\nops/sec',
            ha='center', va='center', fontsize=24, weight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title("Throughput")
    
    # Memory usage (if available)
    ax3 = fig.add_subplot(gs[0, 2])
    dimension = perf_results.get('dimension', 1024)
    storage_method = perf_results.get('storage_method', 'real')
    memory_per_vector = dimension * (8 if storage_method == 'real' else 16)
    ax3.text(0.5, 0.5, f'{memory_per_vector/1024:.1f} KB\nper vector',
            ha='center', va='center', fontsize=16)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title("Memory Usage")
    
    # Accuracy vs capacity (if available)
    ax4 = fig.add_subplot(gs[1, :2])
    if 'capacity_curve' in perf_results:
        n_items = perf_results['capacity_curve']['n_items']
        accuracies = perf_results['capacity_curve']['accuracies']
        ax4.plot(n_items, accuracies, 'bo-', linewidth=2, markersize=8)
        ax4.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
        ax4.set_xlabel("Number of Stored Items")
        ax4.set_ylabel("Retrieval Accuracy")
        ax4.set_title("Capacity Analysis")
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.05)
    else:
        ax4.text(0.5, 0.5, "No capacity data available",
                ha='center', va='center')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
    
    # System info
    ax5 = fig.add_subplot(gs[1, 2])
    info_text = f"Dimension: {dimension}\n"
    info_text += f"Storage: {storage_method}\n"
    info_text += f"Normalized: {perf_results.get('normalized', True)}\n"
    ax5.text(0.1, 0.5, info_text, ha='left', va='center', fontsize=12)
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    ax5.set_title("Configuration")
    
    fig.suptitle("HRR Performance Dashboard", fontsize=16, weight='bold')
    
    return fig