"""
Visualization functions for Vector Symbolic Architectures.

This module provides plotting functions for analyzing VSA operations,
vector distributions, binding comparisons, and performance metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

# Import optional dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .vectors import VSAVector, BinaryVector, BipolarVector, TernaryVector, ComplexVector
from .core import VSA
from .utils import (
    VSACapacityMetrics, VSAPerformanceMetrics,
    analyze_vector_distribution, compare_binding_methods
)


def plot_vector_comparison(
    vectors: List[VSAVector],
    labels: Optional[List[str]] = None,
    method: str = "heatmap",
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Visualize and compare multiple VSA vectors.
    
    Parameters
    ----------
    vectors : List[VSAVector]
        Vectors to compare
    labels : List[str], optional
        Labels for each vector
    method : str
        Visualization method: "heatmap", "line", "radar"
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    Figure or None
        Plotly figure if available
    """
    if not HAS_PLOTLY:
        warnings.warn("Plotly not available for visualization")
        return None
        
    if not vectors:
        raise ValueError("No vectors provided")
        
    if labels is None:
        labels = [f"Vector {i}" for i in range(len(vectors))]
        
    # Ensure same vector type
    vector_type = type(vectors[0])
    if not all(isinstance(v, vector_type) for v in vectors):
        raise ValueError("All vectors must be of the same type")
        
    if method == "heatmap":
        # Create heatmap of vector values
        data = np.stack([v.data for v in vectors])
        
        # Handle complex vectors
        if isinstance(vectors[0], ComplexVector):
            # Show magnitude and phase separately
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Magnitude", "Phase"]
            )
            
            fig.add_trace(
                go.Heatmap(z=np.abs(data), y=labels, colorscale="Viridis"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Heatmap(z=np.angle(data), y=labels, colorscale="HSV"),
                row=1, col=2
            )
            
            fig.update_layout(
                title=title or "Complex Vector Comparison",
                height=400
            )
        else:
            # Regular heatmap
            fig = go.Figure(data=go.Heatmap(
                z=data.real if np.iscomplexobj(data) else data,
                y=labels,
                colorscale="RdBu_r" if isinstance(vectors[0], BipolarVector) else "Viridis"
            ))
            
            fig.update_layout(
                title=title or "Vector Comparison Heatmap",
                xaxis_title="Dimension",
                yaxis_title="Vector",
                height=max(300, 50 * len(vectors))
            )
            
    elif method == "line":
        # Line plot for each vector
        fig = go.Figure()
        
        for i, (vec, label) in enumerate(zip(vectors, labels)):
            data = vec.data
            if isinstance(vec, ComplexVector):
                data = np.abs(data)  # Show magnitude
                
            fig.add_trace(go.Scatter(
                y=data,
                mode='lines',
                name=label,
                line=dict(width=2)
            ))
            
        fig.update_layout(
            title=title or "Vector Component Values",
            xaxis_title="Dimension",
            yaxis_title="Value",
            height=500
        )
        
    elif method == "radar":
        # Radar chart for first few dimensions
        max_dims = min(20, len(vectors[0].data))
        
        fig = go.Figure()
        
        for vec, label in zip(vectors, labels):
            data = vec.data[:max_dims]
            if isinstance(vec, ComplexVector):
                data = np.abs(data)
                
            fig.add_trace(go.Scatterpolar(
                r=data,
                theta=list(range(max_dims)),
                fill='toself',
                name=label
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[np.min([v.data[:max_dims] for v in vectors]),
                           np.max([v.data[:max_dims] for v in vectors])]
                )
            ),
            showlegend=True,
            title=title or f"Vector Comparison (First {max_dims} Dimensions)"
        )
        
    else:
        raise ValueError(f"Unknown method: {method}")
        
    if save_path:
        fig.write_html(save_path)
        
    return fig


def plot_similarity_matrix(
    vectors: List[VSAVector],
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Plot similarity matrix between vectors.
    
    Parameters
    ----------
    vectors : List[VSAVector]
        Vectors to compare
    labels : List[str], optional
        Labels for vectors
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    Figure or None
        Plotly figure if available
    """
    if not HAS_PLOTLY:
        warnings.warn("Plotly not available for visualization")
        return None
        
    n = len(vectors)
    if labels is None:
        labels = [f"V{i}" for i in range(n)]
        
    # Compute similarity matrix
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = vectors[i].similarity(vectors[j])
            
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(similarity_matrix, 3),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=title or "Vector Similarity Matrix",
        xaxis_title="Vector",
        yaxis_title="Vector",
        width=max(600, 50 * n),
        height=max(600, 50 * n)
    )
    
    if save_path:
        fig.write_html(save_path)
        
    return fig


def plot_binding_operation(
    vsa: VSA,
    vec1: VSAVector,
    vec2: VSAVector,
    labels: Optional[Tuple[str, str]] = None,
    show_unbind: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Visualize binding operation between two vectors.
    
    Parameters
    ----------
    vsa : VSA
        VSA instance for binding
    vec1, vec2 : VSAVector
        Vectors to bind
    labels : Tuple[str, str], optional
        Labels for vectors
    show_unbind : bool
        Show unbinding result
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    Figure or None
        Plotly figure if available
    """
    if not HAS_PLOTLY:
        warnings.warn("Plotly not available for visualization")
        return None
        
    if labels is None:
        labels = ("A", "B")
        
    # Perform binding
    bound = vsa.bind(vec1.data, vec2.data)
    bound_vec = type(vec1)(bound)
    
    # Create subplots
    if show_unbind:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"{labels[0]}", f"{labels[1]}",
                f"{labels[0]} ⊗ {labels[1]}", "Unbind Result"
            ],
            vertical_spacing=0.15
        )
        
        # Unbind
        unbound = vsa.unbind(bound, vec1.data)
        unbound_vec = type(vec1)(unbound)
        
        vectors = [vec1, vec2, bound_vec, unbound_vec]
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    else:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"{labels[0]}", f"{labels[1]}",
                f"{labels[0]} ⊗ {labels[1]}", "Similarity"
            ],
            vertical_spacing=0.15
        )
        
        vectors = [vec1, vec2, bound_vec]
        positions = [(1, 1), (1, 2), (2, 1)]
        
    # Plot vectors
    for vec, (row, col) in zip(vectors, positions):
        data = vec.data
        if isinstance(vec, ComplexVector):
            data = np.abs(data)  # Show magnitude
            
        # Limit to first 100 dimensions for clarity
        data = data[:100] if len(data) > 100 else data
        
        fig.add_trace(
            go.Scatter(y=data, mode='lines', showlegend=False),
            row=row, col=col
        )
        
    # Add similarity plot
    if not show_unbind:
        # Compute similarities
        sims = {
            f"{labels[0]} vs Bound": vec1.similarity(bound_vec),
            f"{labels[1]} vs Bound": vec2.similarity(bound_vec),
            f"{labels[0]} vs {labels[1]}": vec1.similarity(vec2)
        }
        
        fig.add_trace(
            go.Bar(
                x=list(sims.keys()),
                y=list(sims.values()),
                showlegend=False
            ),
            row=2, col=2
        )
        
    fig.update_layout(
        title=title or f"Binding Operation ({vsa.config.binding_method})",
        height=800,
        showlegend=False
    )
    
    if save_path:
        fig.write_html(save_path)
        
    return fig


def plot_capacity_analysis(
    capacity_metrics: Union[VSACapacityMetrics, List[VSACapacityMetrics]],
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Plot VSA capacity analysis results.
    
    Parameters
    ----------
    capacity_metrics : VSACapacityMetrics or List
        Capacity analysis results
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    Figure or None
        Plotly figure if available
    """
    if not HAS_PLOTLY:
        warnings.warn("Plotly not available for visualization")
        return None
        
    # Convert single metric to list
    if not isinstance(capacity_metrics, list):
        capacity_metrics = [capacity_metrics]
        
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Empirical vs Theoretical Capacity",
            "Noise Tolerance",
            "Capacity by Method",
            "Efficiency Ratio"
        ]
    )
    
    # Group by dimension
    by_dimension = {}
    for metric in capacity_metrics:
        key = (metric.dimension, metric.vector_type, metric.binding_method)
        if key not in by_dimension:
            by_dimension[key] = []
        by_dimension[key].append(metric)
        
    # Plot 1: Empirical vs Theoretical
    dimensions = []
    empirical = []
    theoretical = []
    labels = []
    
    for (dim, vtype, method), metrics in by_dimension.items():
        dimensions.append(dim)
        empirical.append(np.mean([m.empirical_capacity for m in metrics]))
        theoretical.append(metrics[0].theoretical_capacity)
        labels.append(f"{vtype}-{method}")
        
    fig.add_trace(
        go.Scatter(
            x=dimensions, y=empirical,
            mode='markers+lines',
            name='Empirical',
            marker=dict(size=10)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dimensions, y=theoretical,
            mode='markers+lines',
            name='Theoretical',
            marker=dict(size=10),
            line=dict(dash='dash')
        ),
        row=1, col=1
    )
    
    # Plot 2: Noise Tolerance
    noise_data = []
    for (dim, vtype, method), metrics in by_dimension.items():
        noise_tol = np.mean([m.noise_tolerance for m in metrics])
        noise_data.append({
            'dimension': dim,
            'noise_tolerance': noise_tol,
            'label': f"{vtype}-{method}"
        })
        
    if noise_data:
        fig.add_trace(
            go.Bar(
                x=[d['label'] for d in noise_data],
                y=[d['noise_tolerance'] for d in noise_data],
                showlegend=False
            ),
            row=1, col=2
        )
        
    # Plot 3: Capacity by Method
    method_capacity = {}
    for metric in capacity_metrics:
        method = f"{metric.vector_type}-{metric.binding_method}"
        if method not in method_capacity:
            method_capacity[method] = []
        method_capacity[method].append(metric.max_reliable_bindings)
        
    for method, capacities in method_capacity.items():
        fig.add_trace(
            go.Box(
                y=capacities,
                name=method,
                showlegend=False
            ),
            row=2, col=1
        )
        
    # Plot 4: Efficiency Ratio
    efficiency_data = []
    for metric in capacity_metrics:
        if metric.theoretical_capacity > 0:
            efficiency = metric.empirical_capacity / metric.theoretical_capacity
            efficiency_data.append({
                'method': f"{metric.vector_type}-{metric.binding_method}",
                'efficiency': efficiency,
                'dimension': metric.dimension
            })
            
    if efficiency_data:
        # Group by method
        by_method = {}
        for d in efficiency_data:
            if d['method'] not in by_method:
                by_method[d['method']] = []
            by_method[d['method']].append(d['efficiency'])
            
        fig.add_trace(
            go.Bar(
                x=list(by_method.keys()),
                y=[np.mean(effs) for effs in by_method.values()],
                error_y=dict(
                    type='data',
                    array=[np.std(effs) for effs in by_method.values()]
                ),
                showlegend=False
            ),
            row=2, col=2
        )
        
    # Update layout
    fig.update_xaxes(title_text="Dimension", row=1, col=1)
    fig.update_yaxes(title_text="Capacity", row=1, col=1)
    fig.update_xaxes(title_text="Configuration", row=1, col=2)
    fig.update_yaxes(title_text="Noise Tolerance", row=1, col=2)
    fig.update_xaxes(title_text="Method", row=2, col=1)
    fig.update_yaxes(title_text="Max Bindings", row=2, col=1)
    fig.update_xaxes(title_text="Method", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency Ratio", row=2, col=2)
    
    fig.update_layout(
        title=title or "VSA Capacity Analysis",
        height=800,
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
        
    return fig


def plot_performance_comparison(
    performance_metrics: Dict[str, VSAPerformanceMetrics],
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Plot performance comparison across operations.
    
    Parameters
    ----------
    performance_metrics : Dict[str, VSAPerformanceMetrics]
        Performance metrics by operation
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    Figure or None
        Plotly figure if available
    """
    if not HAS_PLOTLY:
        warnings.warn("Plotly not available for visualization")
        return None
        
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Operations per Second",
            "Mean Operation Time",
            "Time Distribution",
            "Performance by Dimension"
        ]
    )
    
    # Plot 1: Operations per second
    ops = list(performance_metrics.keys())
    ops_per_sec = [m.operations_per_second for m in performance_metrics.values()]
    
    fig.add_trace(
        go.Bar(x=ops, y=ops_per_sec, showlegend=False),
        row=1, col=1
    )
    
    # Plot 2: Mean operation time
    mean_times = [m.mean_time * 1000 for m in performance_metrics.values()]  # Convert to ms
    std_times = [m.std_time * 1000 for m in performance_metrics.values()]
    
    fig.add_trace(
        go.Bar(
            x=ops, y=mean_times,
            error_y=dict(type='data', array=std_times),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Plot 3: Time distribution (box plot)
    for op, metric in performance_metrics.items():
        # Generate sample times based on mean and std
        sample_times = np.random.normal(
            metric.mean_time * 1000,
            metric.std_time * 1000,
            100
        )
        fig.add_trace(
            go.Box(y=sample_times, name=op, showlegend=False),
            row=2, col=1
        )
        
    # Plot 4: Performance by dimension
    dimensions = sorted(set(m.dimension for m in performance_metrics.values()))
    if len(dimensions) > 1:
        for op, metric in performance_metrics.items():
            fig.add_trace(
                go.Scatter(
                    x=[metric.dimension],
                    y=[metric.operations_per_second],
                    mode='markers',
                    name=op,
                    marker=dict(size=15)
                ),
                row=2, col=2
            )
    else:
        # If single dimension, show vector type comparison
        vtypes = set(m.vector_type for m in performance_metrics.values())
        data = []
        for vtype in vtypes:
            metrics = [m for m in performance_metrics.values() if m.vector_type == vtype]
            if metrics:
                data.append({
                    'vector_type': vtype,
                    'avg_ops_per_sec': np.mean([m.operations_per_second for m in metrics])
                })
                
        if data:
            fig.add_trace(
                go.Bar(
                    x=[d['vector_type'] for d in data],
                    y=[d['avg_ops_per_sec'] for d in data],
                    showlegend=False
                ),
                row=2, col=2
            )
            
    # Update layout
    fig.update_xaxes(title_text="Operation", row=1, col=1)
    fig.update_yaxes(title_text="Ops/Second", row=1, col=1)
    fig.update_xaxes(title_text="Operation", row=1, col=2)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=2)
    fig.update_xaxes(title_text="Operation", row=2, col=1)
    fig.update_yaxes(title_text="Time (ms)", row=2, col=1)
    
    if len(dimensions) > 1:
        fig.update_xaxes(title_text="Dimension", row=2, col=2)
        fig.update_yaxes(title_text="Ops/Second", row=2, col=2)
    else:
        fig.update_xaxes(title_text="Vector Type", row=2, col=2)
        fig.update_yaxes(title_text="Avg Ops/Second", row=2, col=2)
        
    fig.update_layout(
        title=title or "VSA Performance Comparison",
        height=800
    )
    
    if save_path:
        fig.write_html(save_path)
        
    return fig


def plot_vector_distribution(
    vectors: List[VSAVector],
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Plot distribution of vector components.
    
    Parameters
    ----------
    vectors : List[VSAVector]
        Vectors to analyze
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    Figure or None
        Plotly figure if available
    """
    if not HAS_PLOTLY:
        warnings.warn("Plotly not available for visualization")
        return None
        
    if not vectors:
        raise ValueError("No vectors provided")
        
    # Analyze distribution
    stats = analyze_vector_distribution(vectors)
    
    # Determine vector type
    vector_type = type(vectors[0])
    
    if isinstance(vectors[0], ComplexVector):
        # For complex vectors, show magnitude and phase distributions
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Magnitude Distribution", "Phase Distribution",
                "Real vs Imaginary", "Phase Scatter"
            ]
        )
        
        # Magnitude histogram
        all_mags = np.abs(np.concatenate([v.data for v in vectors]))
        fig.add_trace(
            go.Histogram(x=all_mags, nbinsx=50, showlegend=False),
            row=1, col=1
        )
        
        # Phase histogram
        all_phases = np.angle(np.concatenate([v.data for v in vectors]))
        fig.add_trace(
            go.Histogram(x=all_phases, nbinsx=50, showlegend=False),
            row=1, col=2
        )
        
        # Real vs Imaginary scatter
        sample_size = min(1000, len(vectors[0].data))
        sample_data = vectors[0].data[:sample_size]
        fig.add_trace(
            go.Scatter(
                x=np.real(sample_data),
                y=np.imag(sample_data),
                mode='markers',
                marker=dict(size=3),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Phase scatter (polar)
        fig.add_trace(
            go.Scatterpolar(
                r=np.abs(sample_data),
                theta=np.angle(sample_data) * 180 / np.pi,
                mode='markers',
                marker=dict(size=3),
                showlegend=False
            ),
            row=2, col=2
        )
        
    elif isinstance(vectors[0], TernaryVector):
        # For ternary vectors, show sparsity patterns
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Value Distribution", "Sparsity Pattern",
                "Non-zero Positions", "Value Counts"
            ]
        )
        
        # Value histogram
        all_values = np.concatenate([v.data for v in vectors])
        fig.add_trace(
            go.Histogram(x=all_values, nbinsx=3, showlegend=False),
            row=1, col=1
        )
        
        # Sparsity heatmap
        sparsity_data = np.stack([v.data != 0 for v in vectors[:10]])  # First 10 vectors
        fig.add_trace(
            go.Heatmap(z=sparsity_data.astype(int), showscale=False),
            row=1, col=2
        )
        
        # Non-zero positions
        nonzero_counts = np.sum(np.stack([v.data != 0 for v in vectors]), axis=0)
        fig.add_trace(
            go.Scatter(y=nonzero_counts, mode='lines', showlegend=False),
            row=2, col=1
        )
        
        # Value counts
        value_counts = {
            "Negative": stats["negative_fraction"],
            "Zero": stats["sparsity"],
            "Positive": stats["positive_fraction"]
        }
        fig.add_trace(
            go.Bar(
                x=list(value_counts.keys()),
                y=list(value_counts.values()),
                showlegend=False
            ),
            row=2, col=2
        )
        
    else:
        # For binary/bipolar vectors, show standard distributions
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Value Distribution", "Component Mean",
                "Component Variance", "Correlation Matrix"
            ]
        )
        
        # Value histogram
        fig.add_trace(
            go.Histogram(
                x=stats["histogram"],
                y=stats["bin_edges"][:-1],
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Component mean
        fig.add_trace(
            go.Scatter(y=stats["mean"], mode='lines', showlegend=False),
            row=1, col=2
        )
        
        # Component variance
        fig.add_trace(
            go.Scatter(y=stats["std"]**2, mode='lines', showlegend=False),
            row=2, col=1
        )
        
        # Correlation matrix (sample)
        if len(vectors) > 1:
            sample_vecs = vectors[:min(10, len(vectors))]
            corr_matrix = np.corrcoef(np.stack([v.data for v in sample_vecs]))
            fig.add_trace(
                go.Heatmap(z=corr_matrix, showscale=True),
                row=2, col=2
            )
            
    # Update layout
    fig.update_layout(
        title=title or f"{vector_type.__name__} Distribution Analysis",
        height=800
    )
    
    if save_path:
        fig.write_html(save_path)
        
    return fig


def plot_binding_comparison(
    dimension: int = 1000,
    vector_type: str = "bipolar",
    num_items: int = 10,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Compare different binding methods.
    
    Parameters
    ----------
    dimension : int
        Vector dimension
    vector_type : str
        Type of vectors
    num_items : int
        Number of items to test
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    Figure or None
        Plotly figure if available
    """
    if not HAS_PLOTLY:
        warnings.warn("Plotly not available for visualization")
        return None
        
    # Compare binding methods
    comparison = compare_binding_methods(dimension, vector_type, num_items)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Retrieval Accuracy", "Algebraic Properties",
            "Performance (Bind)", "Performance (Unbind)"
        ]
    )
    
    methods = list(comparison.keys())
    
    # Plot 1: Retrieval accuracy
    mean_sims = [comparison[m]["mean_similarity"] for m in methods]
    std_sims = [comparison[m]["std_similarity"] for m in methods]
    
    fig.add_trace(
        go.Bar(
            x=methods, y=mean_sims,
            error_y=dict(type='data', array=std_sims),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Plot 2: Algebraic properties
    properties = ["associativity", "commutativity"]
    
    for prop in properties:
        values = [comparison[m][prop] for m in methods]
        fig.add_trace(
            go.Bar(x=methods, y=values, name=prop.capitalize()),
            row=1, col=2
        )
        
    # Plot 3: Bind performance
    bind_perf = [comparison[m]["bind_ops_per_second"] for m in methods]
    fig.add_trace(
        go.Bar(x=methods, y=bind_perf, showlegend=False),
        row=2, col=1
    )
    
    # Plot 4: Unbind performance
    unbind_perf = [comparison[m]["unbind_ops_per_second"] for m in methods]
    fig.add_trace(
        go.Bar(x=methods, y=unbind_perf, showlegend=False),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Method", row=1, col=1)
    fig.update_yaxes(title_text="Similarity", row=1, col=1)
    fig.update_xaxes(title_text="Method", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=2)
    fig.update_xaxes(title_text="Method", row=2, col=1)
    fig.update_yaxes(title_text="Ops/Second", row=2, col=1)
    fig.update_xaxes(title_text="Method", row=2, col=2)
    fig.update_yaxes(title_text="Ops/Second", row=2, col=2)
    
    fig.update_layout(
        title=title or f"Binding Method Comparison ({vector_type}, D={dimension})",
        height=800
    )
    
    if save_path:
        fig.write_html(save_path)
        
    return fig


def create_vsa_dashboard(
    vsa: VSA,
    test_vectors: Optional[List[VSAVector]] = None,
    save_path: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Create comprehensive VSA analysis dashboard.
    
    Parameters
    ----------
    vsa : VSA
        VSA instance to analyze
    test_vectors : List[VSAVector], optional
        Test vectors for analysis
    save_path : str, optional
        Path to save dashboard
        
    Returns
    -------
    Figure or None
        Plotly figure if available
    """
    if not HAS_PLOTLY:
        warnings.warn("Plotly not available for visualization")
        return None
        
    # Generate test vectors if not provided
    if test_vectors is None:
        from .utils import generate_random_vector
        test_vectors = [
            generate_random_vector(vsa.config.dimension, vsa._vector_class)
            for _ in range(10)
        ]
        
    # Create subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            "Configuration", "Vector Distribution", "Similarity Matrix",
            "Binding Example", "Performance", "Capacity Analysis",
            "Memory Usage", "Properties", "Summary"
        ],
        specs=[
            [{"type": "table"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}, {"type": "table"}]
        ]
    )
    
    # 1. Configuration table
    config_data = [
        ["Parameter", "Value"],
        ["Vector Type", vsa.config.vector_type],
        ["Dimension", str(vsa.config.dimension)],
        ["Binding Method", vsa.config.binding_method],
        ["Normalization", str(vsa.config.normalize_result)]
    ]
    
    fig.add_trace(
        go.Table(
            cells=dict(
                values=list(zip(*config_data)),
                align='left'
            )
        ),
        row=1, col=1
    )
    
    # 2. Vector distribution
    all_values = np.concatenate([v.data.flatten() for v in test_vectors])
    fig.add_trace(
        go.Histogram(x=all_values, nbinsx=30, showlegend=False),
        row=1, col=2
    )
    
    # 3. Similarity matrix
    n = min(5, len(test_vectors))
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = test_vectors[i].similarity(test_vectors[j])
            
    fig.add_trace(
        go.Heatmap(z=sim_matrix, colorscale="RdBu_r", showscale=False),
        row=1, col=3
    )
    
    # 4. Binding example
    if len(test_vectors) >= 2:
        bound = vsa.bind(test_vectors[0].data, test_vectors[1].data)
        fig.add_trace(
            go.Scatter(y=bound[:50].flatten(), mode='lines', showlegend=False),
            row=2, col=1
        )
        
    # 5. Performance
    from .utils import benchmark_vsa_operations
    perf = benchmark_vsa_operations(vsa, num_operations=100)
    
    ops = list(perf.keys())
    ops_per_sec = [perf[op].operations_per_second for op in ops]
    
    fig.add_trace(
        go.Bar(x=ops, y=ops_per_sec, showlegend=False),
        row=2, col=2
    )
    
    # 6. Capacity analysis
    from .utils import analyze_binding_capacity
    capacity = analyze_binding_capacity(vsa, num_items=20, num_trials=5)
    
    capacity_data = [
        ["Empirical", f"{capacity.empirical_capacity:.1f}"],
        ["Theoretical", f"{capacity.theoretical_capacity:.1f}"],
        ["Max Bindings", str(capacity.max_reliable_bindings)],
        ["Noise Tolerance", f"{capacity.noise_tolerance:.3f}"]
    ]
    
    fig.add_trace(
        go.Bar(
            x=[d[0] for d in capacity_data],
            y=[float(d[1]) if d[0] != "Max Bindings" else int(d[1]) for d in capacity_data],
            showlegend=False
        ),
        row=2, col=3
    )
    
    # 7. Memory usage
    from .utils import estimate_memory_requirements
    memory = estimate_memory_requirements(
        len(test_vectors),
        vsa.config.dimension,
        vsa.config.vector_type
    )
    
    fig.add_trace(
        go.Bar(
            x=["Basic", "Operations", "Total"],
            y=[memory["basic_storage_mb"], memory["operation_overhead_mb"], memory["total_mb"]],
            showlegend=False
        ),
        row=3, col=1
    )
    
    # 8. Properties table
    props_data = [
        ["Property", "Value"],
        ["Self-inverse", "Yes" if vsa.config.binding_method in ["xor", "permutation"] else "No"],
        ["Commutative", "Yes" if vsa.config.binding_method != "permutation" else "No"],
        ["Supports Bundling", "Yes"],
        ["Preserves Similarity", "Partially"]
    ]
    
    fig.add_trace(
        go.Table(
            cells=dict(
                values=list(zip(*props_data)),
                align='left'
            )
        ),
        row=3, col=2
    )
    
    # 9. Summary table
    summary_data = [
        ["Metric", "Value"],
        ["Total Vectors", str(len(test_vectors))],
        ["Avg Ops/Sec", f"{np.mean(ops_per_sec):.0f}"],
        ["Memory (MB)", f"{memory['total_mb']:.2f}"],
        ["Efficiency", f"{capacity.empirical_capacity/capacity.theoretical_capacity:.2%}"]
    ]
    
    fig.add_trace(
        go.Table(
            cells=dict(
                values=list(zip(*summary_data)),
                align='left'
            )
        ),
        row=3, col=3
    )
    
    # Update layout
    fig.update_layout(
        title="VSA Analysis Dashboard",
        height=1200,
        showlegend=False
    )
    
    if save_path:
        fig.write_html(save_path)
        
    return fig