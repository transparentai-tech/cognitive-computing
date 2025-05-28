"""
Visualization tools for Sparse Distributed Memory.

This module provides comprehensive visualization functions for analyzing and
understanding SDM behavior, including memory distribution plots, activation
patterns, performance metrics, and interactive visualizations.

Visualization Categories:
- Memory State: Distribution of stored patterns, usage heatmaps
- Activation Patterns: Visualization of which locations activate together
- Performance Metrics: Recall accuracy, noise tolerance curves
- Capacity Analysis: Theoretical vs actual capacity visualization
- Interactive Tools: 3D projections, animation of recall process
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA, TSNE
from sklearn.manifold import MDS
import warnings
import logging
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_memory_distribution(sdm, figsize: Tuple[int, int] = (15, 10),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comprehensive memory distribution analysis.
    
    Creates a multi-panel figure showing various aspects of memory distribution
    including usage patterns, saturation/density, and spatial distribution.
    
    Parameters
    ----------
    sdm : SDM
        SDM instance to visualize
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid of subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Location usage histogram
    ax1 = fig.add_subplot(gs[0, 0])
    usage = sdm.location_usage
    ax1.hist(usage[usage > 0], bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Number of Activations')
    ax1.set_ylabel('Number of Locations')
    ax1.set_title('Location Usage Distribution')
    ax1.axvline(np.mean(usage[usage > 0]), color='red', linestyle='--', 
                label=f'Mean: {np.mean(usage[usage > 0]):.1f}')
    ax1.legend()
    
    # 2. Usage heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    # Reshape usage into 2D for visualization
    grid_size = int(np.sqrt(sdm.config.num_hard_locations))
    if grid_size ** 2 == sdm.config.num_hard_locations:
        usage_grid = usage.reshape(grid_size, grid_size)
    else:
        # Pad to make square
        padded_size = grid_size ** 2
        padded_usage = np.pad(usage, (0, padded_size - len(usage)))
        usage_grid = padded_usage.reshape(grid_size, grid_size)
    
    im = ax2.imshow(usage_grid, cmap='hot', aspect='auto')
    ax2.set_title('Location Usage Heatmap')
    ax2.set_xlabel('Location Index (x)')
    ax2.set_ylabel('Location Index (y)')
    plt.colorbar(im, ax=ax2, label='Activation Count')
    
    # 3. Saturation/Density distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if sdm.config.storage_method == 'counters':
        saturation = np.abs(sdm.counters) / sdm.config.saturation_value
        avg_saturation = np.mean(saturation, axis=1)
        ax3.hist(avg_saturation, bins=30, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Average Saturation Level')
        ax3.set_title('Counter Saturation Distribution')
    else:
        density = np.mean(sdm.binary_storage, axis=1)
        ax3.hist(density, bins=30, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Bit Density')
        ax3.set_title('Binary Density Distribution')
    ax3.set_ylabel('Number of Locations')
    
    # 4. Address space coverage (2D projection)
    ax4 = fig.add_subplot(gs[1, :2])
    # Use PCA to project high-dimensional addresses to 2D
    if sdm.config.dimension > 2:
        pca = PCA(n_components=2, random_state=42)
        locations_2d = pca.fit_transform(sdm.hard_locations)
    else:
        locations_2d = sdm.hard_locations
    
    # Color by usage
    scatter = ax4.scatter(locations_2d[:, 0], locations_2d[:, 1], 
                         c=usage, cmap='viridis', s=50, alpha=0.6)
    ax4.set_title('Hard Locations in Address Space (PCA projection)')
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    plt.colorbar(scatter, ax=ax4, label='Usage Count')
    
    # 5. Memory statistics panel
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    stats = sdm.get_memory_stats()
    
    stats_text = f"""Memory Statistics:
    
    Patterns Stored: {stats['num_patterns_stored']}
    Locations Used: {stats['locations_used']} / {stats['num_hard_locations']}
    Usage Rate: {stats['locations_used'] / stats['num_hard_locations']:.1%}
    
    Avg Usage: {stats['avg_location_usage']:.1f}
    Max Usage: {stats['max_location_usage']}
    Usage Std: {stats['location_usage_std']:.1f}
    
    Dimension: {stats['dimension']}
    Activation Radius: {stats['activation_radius']}"""
    
    if sdm.config.storage_method == 'counters':
        stats_text += f"""
    
    Avg Counter Magnitude: {stats['avg_counter_magnitude']:.2f}
    Max Counter Value: {stats['max_counter_value']}
    Saturation Rate: {stats['counter_saturation_rate']:.1%}"""
    else:
        stats_text += f"""
    
    Avg Bit Density: {stats['avg_bit_density']:.3f}
    Saturated Bits: {stats['fully_saturated_bits']}"""
    
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, 
             verticalalignment='top', fontfamily='monospace')
    ax5.set_title('Summary Statistics')
    
    # 6. Activation overlap matrix
    ax6 = fig.add_subplot(gs[2, :])
    # Sample some stored patterns and compute their activation overlaps
    if len(sdm._stored_addresses) > 0:
        sample_size = min(20, len(sdm._stored_addresses))
        sample_indices = np.random.choice(len(sdm._stored_addresses), 
                                        sample_size, replace=False)
        
        # Compute activation sets
        activation_sets = []
        for idx in sample_indices:
            activated = sdm._get_activated_locations(sdm._stored_addresses[idx])
            activation_sets.append(set(activated))
        
        # Compute overlap matrix
        overlap_matrix = np.zeros((sample_size, sample_size))
        for i in range(sample_size):
            for j in range(sample_size):
                if i != j:
                    overlap = len(activation_sets[i] & activation_sets[j])
                    total = len(activation_sets[i] | activation_sets[j])
                    overlap_matrix[i, j] = overlap / total if total > 0 else 0
                else:
                    overlap_matrix[i, j] = 1.0
        
        im = ax6.imshow(overlap_matrix, cmap='RdBu_r', vmin=0, vmax=1)
        ax6.set_title('Activation Overlap Between Stored Patterns')
        ax6.set_xlabel('Pattern Index')
        ax6.set_ylabel('Pattern Index')
        plt.colorbar(im, ax=ax6, label='Overlap Ratio')
    else:
        ax6.text(0.5, 0.5, 'No patterns stored yet', 
                transform=ax6.transAxes, ha='center', va='center')
        ax6.set_title('Activation Overlap Matrix')
    
    plt.suptitle(f'SDM Memory Distribution Analysis', fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Memory distribution plot saved to {save_path}")
    
    return fig


def plot_activation_pattern(sdm, address: np.ndarray, 
                          comparison_addresses: Optional[List[np.ndarray]] = None,
                          figsize: Tuple[int, int] = (12, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize activation pattern for a given address.
    
    Shows which locations are activated and optionally compares with other addresses.
    
    Parameters
    ----------
    sdm : SDM
        SDM instance
    address : np.ndarray
        Address to visualize activation for
    comparison_addresses : list, optional
        Additional addresses to compare
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Get activated locations
    activated = sdm._get_activated_locations(address)
    activated_set = set(activated)
    
    # 1. Activation pattern bar chart
    ax = axes[0, 0]
    activation_vector = np.zeros(sdm.config.num_hard_locations)
    activation_vector[activated] = 1
    
    # Show a subset for visibility
    subset_size = min(100, sdm.config.num_hard_locations)
    ax.bar(range(subset_size), activation_vector[:subset_size], width=1.0)
    ax.set_xlabel('Location Index')
    ax.set_ylabel('Activated')
    ax.set_title(f'Activation Pattern (first {subset_size} locations)')
    ax.set_ylim(-0.1, 1.1)
    
    # 2. Distance distribution
    ax = axes[0, 1]
    distances = np.sum(sdm.hard_locations != address, axis=1)
    
    ax.hist(distances, bins=50, alpha=0.7, label='All locations')
    ax.hist(distances[activated], bins=20, alpha=0.7, label='Activated')
    ax.axvline(sdm.config.activation_radius, color='red', linestyle='--',
              label=f'Radius: {sdm.config.activation_radius}')
    ax.set_xlabel('Hamming Distance')
    ax.set_ylabel('Count')
    ax.set_title('Distance Distribution')
    ax.legend()
    
    # 3. Spatial visualization of activated locations
    ax = axes[1, 0]
    if sdm.config.dimension > 2:
        # Project to 2D
        pca = PCA(n_components=2, random_state=42)
        locations_2d = pca.fit_transform(sdm.hard_locations)
        addr_2d = pca.transform([address])[0]
    else:
        locations_2d = sdm.hard_locations
        addr_2d = address
    
    # Plot all locations
    ax.scatter(locations_2d[:, 0], locations_2d[:, 1], 
              c='lightgray', s=20, alpha=0.5, label='Inactive')
    
    # Highlight activated locations
    ax.scatter(locations_2d[activated, 0], locations_2d[activated, 1],
              c='red', s=50, alpha=0.8, label='Activated')
    
    # Show query address
    ax.scatter(addr_2d[0], addr_2d[1], c='blue', s=200, 
              marker='*', label='Query Address')
    
    ax.set_title('Activated Locations in Address Space')
    ax.set_xlabel('PC1' if sdm.config.dimension > 2 else 'Dim 1')
    ax.set_ylabel('PC2' if sdm.config.dimension > 2 else 'Dim 2')
    ax.legend()
    
    # 4. Comparison with other addresses (if provided)
    ax = axes[1, 1]
    if comparison_addresses:
        # Compare activation overlaps
        overlaps = []
        labels = ['Query']
        
        for i, comp_addr in enumerate(comparison_addresses):
            comp_activated = set(sdm._get_activated_locations(comp_addr))
            overlap = len(activated_set & comp_activated)
            overlaps.append(overlap)
            labels.append(f'Addr {i+1}')
        
        # Create Venn diagram-like visualization
        y_pos = np.arange(len(overlaps))
        ax.barh(y_pos, overlaps)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels[1:])
        ax.set_xlabel('Overlap with Query')
        ax.set_title('Activation Overlap Comparison')
    else:
        # Show activation statistics
        ax.text(0.1, 0.8, f"Activated Locations: {len(activated)}", 
                transform=ax.transAxes, fontsize=12)
        ax.text(0.1, 0.6, f"Activation Rate: {len(activated)/sdm.config.num_hard_locations:.1%}", 
                transform=ax.transAxes, fontsize=12)
        ax.text(0.1, 0.4, f"Mean Distance: {np.mean(distances):.1f}", 
                transform=ax.transAxes, fontsize=12)
        ax.text(0.1, 0.2, f"Min Distance: {np.min(distances)}", 
                transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        ax.set_title('Activation Statistics')
    
    plt.tight_layout()
    plt.suptitle('SDM Activation Pattern Analysis', fontsize=14, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Activation pattern plot saved to {save_path}")
    
    return fig


def plot_recall_accuracy(test_results: Union[Dict, List[Dict]], 
                        figsize: Tuple[int, int] = (12, 8),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot recall accuracy under various conditions.
    
    Can visualize noise tolerance, capacity curves, or comparative results.
    
    Parameters
    ----------
    test_results : dict or list
        Test results from performance testing
        If dict: should contain 'noise_levels' and 'recall_accuracies'
        If list: multiple test results to compare
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Handle single or multiple test results
    if isinstance(test_results, dict):
        test_results = [test_results]
    
    # 1. Noise tolerance curves
    ax = axes[0, 0]
    for i, results in enumerate(test_results):
        if 'noise_tolerance' in results:
            noise_levels = sorted(results['noise_tolerance'].keys())
            accuracies = [results['noise_tolerance'][n] for n in noise_levels]
            label = results.get('label', f'Test {i+1}')
            ax.plot(noise_levels, accuracies, 'o-', label=label, linewidth=2)
    
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Recall Accuracy')
    ax.set_title('Noise Tolerance')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    if len(test_results) > 1:
        ax.legend()
    
    # 2. Capacity curve (if available)
    ax = axes[0, 1]
    if any('capacity_curve' in r for r in test_results):
        for i, results in enumerate(test_results):
            if 'capacity_curve' in results:
                patterns = results['capacity_curve']['patterns']
                accuracy = results['capacity_curve']['accuracy']
                label = results.get('label', f'Test {i+1}')
                ax.plot(patterns, accuracy, 'o-', label=label, linewidth=2)
        
        ax.set_xlabel('Number of Stored Patterns')
        ax.set_ylabel('Recall Accuracy')
        ax.set_title('Capacity vs Accuracy')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    else:
        # Show bit error distribution if available
        if 'bit_errors' in test_results[0]:
            bit_errors = test_results[0]['bit_errors']
            ax.hist(bit_errors, bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Bit Error Rate')
            ax.set_ylabel('Frequency')
            ax.set_title('Bit Error Distribution')
        else:
            ax.text(0.5, 0.5, 'No capacity data available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Capacity Analysis')
    
    # 3. Performance metrics comparison
    ax = axes[1, 0]
    if len(test_results) > 1:
        # Compare key metrics
        metrics = ['recall_accuracy_mean', 'write_time_mean', 'read_time_mean']
        metric_names = ['Recall Accuracy', 'Write Time (ms)', 'Read Time (ms)']
        
        x = np.arange(len(test_results))
        width = 0.25
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = []
            for result in test_results:
                if hasattr(result, metric):
                    val = getattr(result, metric)
                    if 'time' in metric:
                        val *= 1000  # Convert to ms
                    values.append(val)
                else:
                    values.append(0)
            
            ax.bar(x + i*width, values, width, label=name)
        
        ax.set_xlabel('Test Configuration')
        ax.set_ylabel('Value')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels([r.get('label', f'Test {i+1}') for i, r in enumerate(test_results)])
        ax.legend()
    else:
        # Show single test summary
        result = test_results[0]
        summary_text = "Test Summary:\n\n"
        
        if hasattr(result, 'pattern_count'):
            summary_text += f"Patterns Tested: {result.pattern_count}\n"
            summary_text += f"Dimension: {result.dimension}\n\n"
            summary_text += f"Recall Accuracy: {result.recall_accuracy_mean:.3f} ± {result.recall_accuracy_std:.3f}\n"
            summary_text += f"Write Time: {result.write_time_mean*1000:.2f} ± {result.write_time_std*1000:.2f} ms\n"
            summary_text += f"Read Time: {result.read_time_mean*1000:.2f} ± {result.read_time_std*1000:.2f} ms\n"
            summary_text += f"Capacity Utilization: {result.capacity_utilization:.1%}"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
        ax.set_title('Performance Summary')
    
    # 4. Error analysis
    ax = axes[1, 1]
    if 'error_analysis' in test_results[0]:
        error_data = test_results[0]['error_analysis']
        
        # Plot error vs pattern characteristics
        if 'sparsity_errors' in error_data:
            sparsity = error_data['sparsity_levels']
            errors = error_data['sparsity_errors']
            ax.plot(sparsity, errors, 'o-', label='vs Sparsity')
        
        if 'similarity_errors' in error_data:
            similarity = error_data['similarity_levels']
            errors = error_data['similarity_errors']
            ax.plot(similarity, errors, 's-', label='vs Similarity')
        
        ax.set_xlabel('Pattern Characteristic')
        ax.set_ylabel('Error Rate')
        ax.set_title('Error Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # Show noise level vs error rate relationship
        if test_results[0].get('noise_tolerance'):
            noise_data = test_results[0]['noise_tolerance']
            noise_levels = sorted(noise_data.keys())
            error_rates = [1 - noise_data[n] for n in noise_levels]
            
            ax.plot(noise_levels, error_rates, 'ro-', linewidth=2)
            ax.set_xlabel('Noise Level')
            ax.set_ylabel('Error Rate')
            ax.set_title('Noise vs Error Rate')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('SDM Recall Accuracy Analysis', fontsize=14, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Recall accuracy plot saved to {save_path}")
    
    return fig


def visualize_memory_contents(sdm, num_samples: int = 100,
                            method: str = 'tsne',
                            color_by: str = 'usage',
                            interactive: bool = False,
                            figsize: Tuple[int, int] = (10, 8),
                            save_path: Optional[str] = None) -> Union[plt.Figure, go.Figure]:
    """
    Visualize memory contents using dimensionality reduction.
    
    Creates an interactive or static visualization of memory contents projected
    to 2D or 3D space.
    
    Parameters
    ----------
    sdm : SDM
        SDM instance to visualize
    num_samples : int, optional
        Number of locations to sample for visualization
    method : str, optional
        Dimensionality reduction method: 'pca', 'tsne', 'mds'
    color_by : str, optional
        How to color points: 'usage', 'saturation', 'cluster'
    interactive : bool, optional
        Create interactive plotly visualization
    figsize : tuple, optional
        Figure size for static plot
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure or go.Figure
        Created figure (matplotlib or plotly)
    """
    # Sample locations if needed
    if sdm.config.num_hard_locations > num_samples:
        sample_indices = np.random.choice(sdm.config.num_hard_locations, 
                                        num_samples, replace=False)
    else:
        sample_indices = np.arange(sdm.config.num_hard_locations)
    
    # Get data to visualize
    if sdm.config.storage_method == 'counters':
        data = sdm.counters[sample_indices]
    else:
        data = sdm.binary_storage[sample_indices].astype(float)
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=3 if interactive else 2, random_state=42)
        coords = reducer.fit_transform(data)
        variance_explained = reducer.explained_variance_ratio_
    elif method == 'tsne':
        n_components = 3 if interactive else 2
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, num_samples-1))
        coords = reducer.fit_transform(data)
        variance_explained = None
    elif method == 'mds':
        n_components = 3 if interactive else 2
        mds = MDS(n_components=n_components, random_state=42)
        coords = mds.fit_transform(data)
        variance_explained = None
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Determine colors
    usage = sdm.location_usage[sample_indices]
    
    if color_by == 'usage':
        colors = usage
        color_label = 'Usage Count'
    elif color_by == 'saturation':
        if sdm.config.storage_method == 'counters':
            colors = np.mean(np.abs(data) / sdm.config.saturation_value, axis=1)
            color_label = 'Saturation Level'
        else:
            colors = np.mean(data, axis=1)
            color_label = 'Bit Density'
    elif color_by == 'cluster':
        # Simple k-means clustering
        from sklearn.cluster import KMeans
        n_clusters = min(5, num_samples // 10)
        clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(data)
        colors = clusters
        color_label = 'Cluster'
    else:
        colors = usage
        color_label = 'Usage Count'
    
    if interactive:
        # Create interactive 3D plot with plotly
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title=color_label)
            ),
            text=[f"Location {idx}<br>Usage: {usage[i]}" 
                  for i, idx in enumerate(sample_indices)],
            hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}'
        ))
        
        # Add stored pattern locations if available
        if len(sdm._stored_addresses) > 0:
            # Project stored addresses
            stored_sample = min(50, len(sdm._stored_addresses))
            stored_indices = np.random.choice(len(sdm._stored_addresses), 
                                            stored_sample, replace=False)
            stored_data = np.array([sdm._stored_addresses[i] for i in stored_indices])
            
            if method == 'pca':
                stored_coords = reducer.transform(stored_data)
            else:
                # For t-SNE/MDS, we need to refit with combined data
                combined_data = np.vstack([data, stored_data])
                if method == 'tsne':
                    all_coords = TSNE(n_components=3, random_state=42).fit_transform(combined_data)
                else:
                    all_coords = MDS(n_components=3, random_state=42).fit_transform(combined_data)
                stored_coords = all_coords[num_samples:]
            
            fig.add_trace(go.Scatter3d(
                x=stored_coords[:, 0],
                y=stored_coords[:, 1],
                z=stored_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='diamond'
                ),
                name='Stored Patterns',
                text=[f"Pattern {i}" for i in stored_indices],
                hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}'
            ))
        
        # Update layout
        title = f'SDM Memory Contents Visualization ({method.upper()})'
        if variance_explained is not None:
            title += f'<br>Variance Explained: {sum(variance_explained[:3]):.1%}'
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3'
            ),
            width=800,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive visualization saved to {save_path}")
        
        return fig
    
    else:
        # Create static 2D plot
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Main scatter plot
        ax = axes[0, 0]
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, 
                           cmap='viridis', s=50, alpha=0.7)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        title = f'{method.upper()} Projection'
        if variance_explained is not None:
            title += f' ({sum(variance_explained[:2]):.1%} var)'
        ax.set_title(title)
        plt.colorbar(scatter, ax=ax, label=color_label)
        
        # 2. Density plot
        ax = axes[0, 1]
        from scipy.stats import gaussian_kde
        xy = coords[:, :2].T
        z = gaussian_kde(xy)(xy)
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=z, cmap='hot', s=50)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title('Density Plot')
        plt.colorbar(scatter, ax=ax, label='Density')
        
        # 3. Hierarchical clustering dendrogram
        ax = axes[1, 0]
        if num_samples <= 50:
            # Only show dendrogram for small samples
            linkage_matrix = linkage(coords[:, :2], method='ward')
            dendrogram(linkage_matrix, ax=ax, no_labels=True)
            ax.set_title('Hierarchical Clustering')
            ax.set_xlabel('Location')
            ax.set_ylabel('Distance')
        else:
            # Show distance matrix heatmap for subset
            subset = min(20, num_samples)
            dist_matrix = squareform(pdist(coords[:subset, :2]))
            im = ax.imshow(dist_matrix, cmap='viridis')
            ax.set_title('Distance Matrix (subset)')
            ax.set_xlabel('Location')
            ax.set_ylabel('Location')
            plt.colorbar(im, ax=ax)
        
        # 4. Component analysis
        ax = axes[1, 1]
        if variance_explained is not None:
            # Show explained variance
            components = range(1, len(variance_explained) + 1)
            ax.bar(components[:10], variance_explained[:10])
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Variance Explained')
            ax.set_title('PCA Variance Explained')
        else:
            # Show location statistics
            ax.hist(usage, bins=30, alpha=0.7, label='Usage')
            if sdm.config.storage_method == 'counters':
                saturation = np.mean(np.abs(data) / sdm.config.saturation_value, axis=1)
                ax.hist(saturation * np.max(usage), bins=30, alpha=0.7, label='Saturation (scaled)')
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            ax.set_title('Location Statistics')
            ax.legend()
        
        plt.tight_layout()
        plt.suptitle('SDM Memory Contents Visualization', fontsize=14, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Memory contents visualization saved to {save_path}")
        
        return fig


def plot_decoder_comparison(sdm_instances: Dict[str, Any],
                          test_size: int = 100,
                          figsize: Tuple[int, int] = (15, 10),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare different decoder strategies.
    
    Parameters
    ----------
    sdm_instances : dict
        Dictionary mapping decoder names to SDM instances
    test_size : int, optional
        Number of test patterns
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Created figure
    """
    from cognitive_computing.sdm.utils import generate_random_patterns, add_noise
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Generate test data
    addresses, data = generate_random_patterns(test_size, 
                                             list(sdm_instances.values())[0].config.dimension)
    
    results = {}
    
    # Test each decoder
    for name, sdm in sdm_instances.items():
        # Clear memory
        sdm.clear()
        
        # Store patterns
        for addr, dat in zip(addresses, data):
            sdm.store(addr, dat)
        
        # Test recall with various noise levels
        noise_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        accuracies = []
        activation_counts = []
        
        for noise in noise_levels:
            noise_accuracies = []
            noise_activations = []
            
            for i in range(min(20, test_size)):
                noisy_addr = add_noise(addresses[i], noise)
                activated = sdm._get_activated_locations(noisy_addr)
                noise_activations.append(len(activated))
                
                recalled = sdm.recall(noisy_addr)
                if recalled is not None:
                    accuracy = np.mean(recalled == data[i])
                    noise_accuracies.append(accuracy)
            
            accuracies.append(np.mean(noise_accuracies) if noise_accuracies else 0)
            activation_counts.append(np.mean(noise_activations))
        
        results[name] = {
            'noise_levels': noise_levels,
            'accuracies': accuracies,
            'activation_counts': activation_counts,
            'stats': sdm.get_memory_stats()
        }
    
    # Plot results
    # 1. Noise tolerance comparison
    ax = axes[0, 0]
    for name, result in results.items():
        ax.plot(result['noise_levels'], result['accuracies'], 'o-', label=name, linewidth=2)
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Recall Accuracy')
    ax.set_title('Noise Tolerance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Activation count comparison
    ax = axes[0, 1]
    for name, result in results.items():
        ax.plot(result['noise_levels'], result['activation_counts'], 'o-', label=name, linewidth=2)
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Average Activations')
    ax.set_title('Activation Count vs Noise')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Memory utilization
    ax = axes[0, 2]
    names = list(results.keys())
    utilizations = [r['stats']['locations_used'] / r['stats']['num_hard_locations'] 
                   for r in results.values()]
    ax.bar(names, utilizations)
    ax.set_ylabel('Location Utilization')
    ax.set_title('Memory Utilization')
    ax.set_ylim(0, 1)
    
    # 4. Performance at different noise levels
    noise_idx = 3  # 0.15 noise
    ax = axes[1, 0]
    accuracies_at_noise = [r['accuracies'][noise_idx] for r in results.values()]
    ax.bar(names, accuracies_at_noise)
    ax.set_ylabel('Recall Accuracy')
    ax.set_title(f'Accuracy at {results[names[0]]["noise_levels"][noise_idx]} Noise')
    ax.set_ylim(0, 1)
    
    # 5. Activation uniformity
    ax = axes[1, 1]
    uniformities = []
    for name, sdm in sdm_instances.items():
        usage = sdm.location_usage[sdm.location_usage > 0]
        if len(usage) > 0:
            uniformity = 1 - (np.std(usage) / np.mean(usage))
        else:
            uniformity = 0
        uniformities.append(uniformity)
    
    ax.bar(names, uniformities)
    ax.set_ylabel('Uniformity Score')
    ax.set_title('Location Usage Uniformity')
    ax.set_ylim(0, 1)
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "Decoder Comparison Summary:\n\n"
    for name, result in results.items():
        stats = result['stats']
        summary_text += f"{name}:\n"
        summary_text += f"  Avg Accuracy: {np.mean(result['accuracies']):.3f}\n"
        summary_text += f"  Locations Used: {stats['locations_used']}\n"
        summary_text += f"  Avg Activations: {np.mean(result['activation_counts']):.1f}\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    plt.tight_layout()
    plt.suptitle('SDM Decoder Strategy Comparison', fontsize=14, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Decoder comparison plot saved to {save_path}")
    
    return fig


def create_recall_animation(sdm, address: np.ndarray, 
                          noise_levels: List[float] = None,
                          interval: int = 500,
                          save_path: Optional[str] = None) -> FuncAnimation:
    """
    Create animation showing recall process with increasing noise.
    
    Parameters
    ----------
    sdm : SDM
        SDM instance with stored patterns
    address : np.ndarray
        Original address to recall
    noise_levels : list, optional
        Noise levels to animate through
    interval : int, optional
        Milliseconds between frames
    save_path : str, optional
        Path to save animation (as .gif or .mp4)
        
    Returns
    -------
    FuncAnimation
        Animation object
    """
    from cognitive_computing.sdm.utils import add_noise
    
    if noise_levels is None:
        noise_levels = np.linspace(0, 0.5, 20)
    
    # Store a pattern if none stored
    if len(sdm._stored_addresses) == 0:
        data = np.random.randint(0, 2, sdm.config.dimension)
        sdm.store(address, data)
        original_data = data
    else:
        # Find the stored data for this address
        idx = 0
        for i, addr in enumerate(sdm._stored_addresses):
            if np.array_equal(addr, address):
                idx = i
                break
        original_data = sdm._stored_data[idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Initialize plots
    # 1. Address with noise
    ax1 = axes[0, 0]
    addr_plot = ax1.imshow(address.reshape(1, -1), cmap='binary', aspect='auto')
    ax1.set_title('Input Address')
    ax1.set_xlabel('Bit Position')
    ax1.set_yticks([])
    
    # 2. Activated locations
    ax2 = axes[0, 1]
    activation_bar = None
    
    # 3. Recalled pattern
    ax3 = axes[1, 0]
    recall_plot = ax3.imshow(np.zeros((1, sdm.config.dimension)), 
                           cmap='binary', aspect='auto')
    ax3.set_title('Recalled Pattern')
    ax3.set_xlabel('Bit Position')
    ax3.set_yticks([])
    
    # 4. Accuracy over noise
    ax4 = axes[1, 1]
    accuracy_line, = ax4.plot([], [], 'o-', linewidth=2)
    ax4.set_xlim(0, max(noise_levels))
    ax4.set_ylim(-0.05, 1.05)
    ax4.set_xlabel('Noise Level')
    ax4.set_ylabel('Recall Accuracy')
    ax4.set_title('Recall Accuracy vs Noise')
    ax4.grid(True, alpha=0.3)
    
    noise_history = []
    accuracy_history = []
    
    def animate(frame):
        noise_level = noise_levels[frame]
        
        # Add noise to address
        noisy_addr = add_noise(address, noise_level)
        addr_plot.set_data(noisy_addr.reshape(1, -1))
        ax1.set_title(f'Input Address (Noise: {noise_level:.2f})')
        
        # Get activated locations
        activated = sdm._get_activated_locations(noisy_addr)
        
        # Update activation plot
        nonlocal activation_bar
        if activation_bar is not None:
            activation_bar.remove()
        
        activation_pattern = np.zeros(min(100, sdm.config.num_hard_locations))
        for idx in activated:
            if idx < len(activation_pattern):
                activation_pattern[idx] = 1
        
        activation_bar = ax2.bar(range(len(activation_pattern)), 
                               activation_pattern, width=1.0)
        ax2.set_title(f'Activated Locations ({len(activated)} total)')
        ax2.set_ylim(-0.1, 1.1)
        
        # Recall pattern
        recalled = sdm.recall(noisy_addr)
        
        if recalled is not None:
            recall_plot.set_data(recalled.reshape(1, -1))
            accuracy = np.mean(recalled == original_data)
        else:
            recall_plot.set_data(np.zeros((1, sdm.config.dimension)).reshape(1, -1))
            accuracy = 0.0
        
        ax3.set_title(f'Recalled Pattern (Accuracy: {accuracy:.2%})')
        
        # Update accuracy plot
        noise_history.append(noise_level)
        accuracy_history.append(accuracy)
        accuracy_line.set_data(noise_history, accuracy_history)
        
        return [addr_plot, activation_bar, recall_plot, accuracy_line]
    
    anim = FuncAnimation(fig, animate, frames=len(noise_levels),
                        interval=interval, blit=False)
    
    plt.tight_layout()
    plt.suptitle('SDM Recall Process Animation', fontsize=14, y=1.02)
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow')
        else:
            anim.save(save_path, writer='ffmpeg')
        logger.info(f"Animation saved to {save_path}")
    
    return anim


def plot_theoretical_analysis(dimension_range: Tuple[int, int] = (100, 2000),
                            num_points: int = 20,
                            figsize: Tuple[int, int] = (15, 10),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot theoretical SDM properties across dimensions.
    
    Parameters
    ----------
    dimension_range : tuple, optional
        Range of dimensions to analyze
    num_points : int, optional
        Number of points to plot
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Created figure
    """
    from cognitive_computing.sdm.utils import compute_memory_capacity
    
    dimensions = np.linspace(dimension_range[0], dimension_range[1], num_points, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Analyze properties for each dimension
    critical_distances = []
    capacities_1k = []
    capacities_10k = []
    optimal_radii = []
    activation_probs = []
    
    for dim in dimensions:
        # Critical distance
        critical_dist = 0.451 * dim
        critical_distances.append(critical_dist)
        
        # Capacity with different number of locations
        cap_1k = compute_memory_capacity(dim, 1000, int(critical_dist))
        cap_10k = compute_memory_capacity(dim, 10000, int(critical_dist))
        
        capacities_1k.append(cap_1k['kanerva_estimate'])
        capacities_10k.append(cap_10k['kanerva_estimate'])
        
        # Optimal radius for 1000 locations
        from cognitive_computing.sdm.memory import MemoryOptimizer
        optimal_r = MemoryOptimizer.find_optimal_radius(dim, 1000)
        optimal_radii.append(optimal_r)
        
        # Activation probability
        activation_probs.append(cap_1k['activation_probability'])
    
    # 1. Critical distance scaling
    ax = axes[0, 0]
    ax.plot(dimensions, critical_distances, 'o-', linewidth=2)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Critical Distance')
    ax.set_title('Critical Distance vs Dimension')
    ax.grid(True, alpha=0.3)
    
    # 2. Capacity scaling
    ax = axes[0, 1]
    ax.plot(dimensions, capacities_1k, 'o-', label='1K locations', linewidth=2)
    ax.plot(dimensions, capacities_10k, 's-', label='10K locations', linewidth=2)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Capacity (patterns)')
    ax.set_title('Capacity Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Optimal radius
    ax = axes[0, 2]
    ax.plot(dimensions, optimal_radii, 'o-', label='Optimal', linewidth=2)
    ax.plot(dimensions, critical_distances, 's-', label='Critical', linewidth=2)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Radius')
    ax.set_title('Optimal vs Critical Radius')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Activation probability
    ax = axes[1, 0]
    ax.semilogy(dimensions, activation_probs, 'o-', linewidth=2)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Activation Probability (log)')
    ax.set_title('Activation Probability Scaling')
    ax.grid(True, alpha=0.3)
    
    # 5. Capacity per location
    ax = axes[1, 1]
    efficiency_1k = np.array(capacities_1k) / 1000
    efficiency_10k = np.array(capacities_10k) / 10000
    ax.plot(dimensions, efficiency_1k, 'o-', label='1K locations', linewidth=2)
    ax.plot(dimensions, efficiency_10k, 's-', label='10K locations', linewidth=2)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Patterns per Location')
    ax.set_title('Storage Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Information density
    ax = axes[1, 2]
    # Bits per pattern stored
    bits_per_pattern_1k = dimensions * 1000 / np.array(capacities_1k)
    bits_per_pattern_10k = dimensions * 10000 / np.array(capacities_10k)
    
    ax.plot(dimensions, bits_per_pattern_1k, 'o-', label='1K locations', linewidth=2)
    ax.plot(dimensions, bits_per_pattern_10k, 's-', label='10K locations', linewidth=2)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Bits per Pattern')
    ax.set_title('Storage Cost')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Theoretical SDM Analysis', fontsize=14, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Theoretical analysis plot saved to {save_path}")
    
    return fig