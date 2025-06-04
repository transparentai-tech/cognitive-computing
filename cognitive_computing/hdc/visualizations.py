"""
Visualization functions for HDC.

This module provides plotting and visualization utilities for
hyperdimensional computing systems.
"""

from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from pathlib import Path
import logging

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from cognitive_computing.hdc.utils import generate_similarity_matrix
from cognitive_computing.hdc.core import HDC
from cognitive_computing.hdc.operations import similarity

logger = logging.getLogger(__name__)


def plot_hypervector(
    hypervector: np.ndarray,
    title: str = "Hypervector Visualization",
    segment_size: int = 100,
    figsize: Tuple[int, int] = (12, 6)
) -> Tuple[Figure, Axes]:
    """
    Plot a hypervector as segments.
    
    Parameters
    ----------
    hypervector : np.ndarray
        Hypervector to visualize
    title : str
        Plot title
    segment_size : int
        Size of segments to display
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[Figure, Axes]
        Matplotlib figure and axes
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # First segment
    segment1 = hypervector[:segment_size]
    axes[0].bar(range(len(segment1)), segment1)
    axes[0].set_title(f"{title} - First {segment_size} dimensions")
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Value")
    
    # Middle segment
    mid_start = len(hypervector) // 2 - segment_size // 2
    segment2 = hypervector[mid_start:mid_start + segment_size]
    axes[1].bar(range(len(segment2)), segment2)
    axes[1].set_title(f"Middle {segment_size} dimensions")
    axes[1].set_xlabel("Dimension")
    axes[1].set_ylabel("Value")
    
    plt.tight_layout()
    return fig, axes


def plot_similarity_matrix(
    vectors: List[np.ndarray],
    labels: Optional[List[str]] = None,
    metric: str = "cosine",
    cmap: str = "coolwarm",
    figsize: Tuple[int, int] = (8, 6),
    annotate: bool = True
) -> Tuple[Figure, Axes]:
    """
    Plot similarity matrix as heatmap.
    
    Parameters
    ----------
    vectors : List[np.ndarray]
        List of hypervectors
    labels : List[str], optional
        Labels for vectors
    metric : str
        Similarity metric
    cmap : str
        Colormap
    figsize : Tuple[int, int]
        Figure size
    annotate : bool
        Whether to annotate cells with values
        
    Returns
    -------
    Tuple[Figure, Axes]
        Matplotlib figure and axes
    """
    # Generate similarity matrix
    sim_matrix, labels = generate_similarity_matrix(vectors, labels, metric)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        sim_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        center=0.0,
        annot=annotate,
        fmt=".2f",
        square=True,
        cbar_kws={"label": f"{metric.capitalize()} Similarity"},
        ax=ax
    )
    
    ax.set_title("Hypervector Similarity Matrix")
    plt.tight_layout()
    
    return fig, ax


def plot_binding_operation(
    a: np.ndarray,
    b: np.ndarray,
    result: np.ndarray,
    operation: str = "bind",
    segment_size: int = 50,
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[Figure, np.ndarray]:
    """
    Visualize binding operation.
    
    Parameters
    ----------
    a : np.ndarray
        First hypervector
    b : np.ndarray
        Second hypervector
    result : np.ndarray
        Result of operation
    operation : str
        Operation name
    segment_size : int
        Segment size to display
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[Figure, np.ndarray]
        Matplotlib figure and axes array
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot segments
    segment = slice(0, segment_size)
    
    axes[0].bar(range(segment_size), a[segment], alpha=0.7)
    axes[0].set_title(f"Vector A (first {segment_size} dimensions)")
    axes[0].set_ylabel("Value")
    
    axes[1].bar(range(segment_size), b[segment], alpha=0.7, color='orange')
    axes[1].set_title(f"Vector B (first {segment_size} dimensions)")
    axes[1].set_ylabel("Value")
    
    axes[2].bar(range(segment_size), result[segment], alpha=0.7, color='green')
    axes[2].set_title(f"Result: A {operation} B")
    axes[2].set_xlabel("Dimension")
    axes[2].set_ylabel("Value")
    
    # Add similarities
    sim_a = similarity(result, a)
    sim_b = similarity(result, b)
    
    fig.suptitle(
        f"{operation.capitalize()} Operation\n"
        f"Similarity(Result, A) = {sim_a:.3f}, "
        f"Similarity(Result, B) = {sim_b:.3f}"
    )
    
    plt.tight_layout()
    return fig, axes


def plot_capacity_analysis(
    metrics: 'HDCPerformanceMetrics',
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[Figure, np.ndarray]:
    """
    Plot capacity analysis results.
    
    Parameters
    ----------
    metrics : HDCPerformanceMetrics
        Performance metrics from capacity analysis
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[Figure, np.ndarray]
        Matplotlib figure and axes array
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Similarity distribution
    if "random_pairs" in metrics.similarity_distribution:
        similarities = metrics.similarity_distribution["random_pairs"]
        axes[0, 0].hist(similarities, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(
            np.mean(similarities),
            color='red',
            linestyle='--',
            label=f'Mean: {np.mean(similarities):.3f}'
        )
        axes[0, 0].set_xlabel("Similarity")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("Pairwise Similarity Distribution")
        axes[0, 0].legend()
    
    # Noise tolerance
    if metrics.noise_tolerance:
        noise_levels = sorted(metrics.noise_tolerance.keys())
        recovery_rates = [metrics.noise_tolerance[n] for n in noise_levels]
        
        axes[0, 1].plot(noise_levels, recovery_rates, marker='o', linewidth=2)
        axes[0, 1].set_xlabel("Noise Level")
        axes[0, 1].set_ylabel("Recovery Rate")
        axes[0, 1].set_title("Noise Tolerance")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1.1)
    
    # Operation times
    if metrics.operation_times:
        operations = list(metrics.operation_times.keys())
        times = list(metrics.operation_times.values())
        
        axes[1, 0].bar(operations, times, alpha=0.7)
        axes[1, 0].set_xlabel("Operation")
        axes[1, 0].set_ylabel("Time (ms)")
        axes[1, 0].set_title("Operation Benchmarks")
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Summary info
    axes[1, 1].text(0.1, 0.8, f"Dimension: {metrics.dimension}", fontsize=12)
    axes[1, 1].text(0.1, 0.7, f"Type: {metrics.hypervector_type}", fontsize=12)
    
    if metrics.capacity_results:
        axes[1, 1].text(
            0.1, 0.6,
            f"Est. Capacity: {metrics.capacity_results.get('estimated_capacity', 'N/A'):,}",
            fontsize=12
        )
        axes[1, 1].text(
            0.1, 0.5,
            f"Mean Similarity: {metrics.capacity_results.get('mean_similarity', 0):.4f}",
            fontsize=12
        )
        axes[1, 1].text(
            0.1, 0.4,
            f"Max Similarity: {metrics.capacity_results.get('max_similarity', 0):.4f}",
            fontsize=12
        )
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title("Summary Statistics")
    
    plt.tight_layout()
    return fig, axes


def plot_classifier_performance(
    performance_metrics: Dict[str, float],
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[Figure, Axes]:
    """
    Plot classifier performance metrics.
    
    Parameters
    ----------
    performance_metrics : Dict[str, float]
        Performance metrics dictionary
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[Figure, Axes]
        Matplotlib figure and axes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy metrics
    accuracy_metrics = {
        k: v for k, v in performance_metrics.items()
        if k.startswith('accuracy_') or k.endswith('_accuracy')
    }
    
    if accuracy_metrics:
        labels = list(accuracy_metrics.keys())
        values = list(accuracy_metrics.values())
        
        ax1.bar(labels, values, alpha=0.7)
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Classification Accuracy")
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, (label, value) in enumerate(zip(labels, values)):
            ax1.text(i, value + 0.01, f'{value:.2f}', ha='center')
    
    # Confusion pairs
    if "confusion_pairs" in performance_metrics:
        confusion = performance_metrics["confusion_pairs"]
        if confusion:
            labels = list(confusion.keys())
            values = list(confusion.values())
            
            ax2.barh(labels, values, alpha=0.7, color='coral')
            ax2.set_xlabel("Count")
            ax2.set_title("Classification Confusions")
        else:
            ax2.text(0.5, 0.5, "No confusions", ha='center', va='center')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
    else:
        ax2.axis('off')
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_hypervector_comparison(
    vectors: Dict[str, np.ndarray],
    segment_size: int = 100,
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[Figure, np.ndarray]:
    """
    Compare multiple hypervectors.
    
    Parameters
    ----------
    vectors : Dict[str, np.ndarray]
        Dictionary of labeled hypervectors
    segment_size : int
        Segment size to display
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[Figure, np.ndarray]
        Matplotlib figure and axes array
    """
    n_vectors = len(vectors)
    fig, axes = plt.subplots(n_vectors, 1, figsize=figsize, sharex=True)
    
    if n_vectors == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_vectors))
    
    for i, (label, vector) in enumerate(vectors.items()):
        segment = vector[:segment_size]
        axes[i].bar(range(len(segment)), segment, alpha=0.7, color=colors[i])
        axes[i].set_ylabel("Value")
        axes[i].set_title(f"{label} (first {segment_size} dimensions)")
    
    axes[-1].set_xlabel("Dimension")
    
    plt.tight_layout()
    return fig, axes


def create_interactive_similarity_plot(
    vectors: List[np.ndarray],
    labels: Optional[List[str]] = None,
    metric: str = "cosine"
) -> Optional['go.Figure']:
    """
    Create interactive similarity matrix plot using Plotly.
    
    Parameters
    ----------
    vectors : List[np.ndarray]
        List of hypervectors
    labels : List[str], optional
        Labels for vectors
    metric : str
        Similarity metric
        
    Returns
    -------
    go.Figure or None
        Plotly figure if available, None otherwise
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available for interactive plots")
        return None
    
    # Generate similarity matrix
    sim_matrix, labels = generate_similarity_matrix(vectors, labels, metric)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0,
        text=np.round(sim_matrix, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title=f"{metric.capitalize()} Similarity")
    ))
    
    fig.update_layout(
        title="Interactive Hypervector Similarity Matrix",
        xaxis_title="Vector",
        yaxis_title="Vector",
        width=800,
        height=700
    )
    
    return fig


def save_plots(
    figures: Union[Figure, List[Figure]],
    base_path: Union[str, Path],
    format: str = "png",
    dpi: int = 300
) -> None:
    """
    Save matplotlib figures to files.
    
    Parameters
    ----------
    figures : Figure or List[Figure]
        Figure(s) to save
    base_path : str or Path
        Base path for saving (without extension)
    format : str
        File format (png, pdf, svg)
    dpi : int
        Resolution for raster formats
    """
    if isinstance(figures, Figure):
        figures = [figures]
    
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    for i, fig in enumerate(figures):
        if len(figures) > 1:
            save_path = f"{base_path}_{i}.{format}"
        else:
            save_path = f"{base_path}.{format}"
        
        fig.savefig(save_path, format=format, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")