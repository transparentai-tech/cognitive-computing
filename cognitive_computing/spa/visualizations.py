"""
Visualization functions for Semantic Pointer Architecture.

This module provides plotting and visualization utilities for SPA models,
including similarity matrices, action selection dynamics, network graphs,
production flow, and state evolution animations.
"""

from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from pathlib import Path
import logging
import warnings

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

from .core import Vocabulary, SemanticPointer, SPA
from .actions import BasalGanglia, Thalamus
from .networks import Network, Connection
from .production import ProductionSystem, Production
from .modules import Module

logger = logging.getLogger(__name__)


def plot_similarity_matrix(
    vocab: Vocabulary,
    subset: Optional[List[str]] = None,
    threshold: float = 0.0,
    cmap: str = "coolwarm",
    figsize: Tuple[int, int] = (10, 8),
    annotate: bool = True,
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot similarity matrix for vocabulary.
    
    Parameters
    ----------
    vocab : Vocabulary
        Vocabulary containing semantic pointers
    subset : List[str], optional
        Subset of keys to include (default: all)
    threshold : float
        Minimum similarity to display
    cmap : str
        Colormap name
    figsize : Tuple[int, int]
        Figure size
    annotate : bool
        Whether to annotate cells with values
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    Tuple[Figure, Axes]
        Matplotlib figure and axes
    """
    # Get pointer names
    if subset is None:
        keys = list(vocab.pointers.keys())
    else:
        keys = [k for k in subset if k in vocab]
        
    if not keys:
        raise ValueError("No valid keys found")
        
    # Compute similarity matrix
    n = len(keys)
    sim_matrix = np.zeros((n, n))
    
    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            sim = vocab[key1].dot(vocab[key2])
            if abs(sim) >= threshold:
                sim_matrix[i, j] = sim
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        sim_matrix,
        xticklabels=keys,
        yticklabels=keys,
        cmap=cmap,
        center=0.0,
        annot=annotate,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Similarity"},
        ax=ax,
        vmin=-1,
        vmax=1
    )
    
    ax.set_title(f"Semantic Pointer Similarity Matrix")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_action_selection(
    bg: BasalGanglia,
    history: np.ndarray,
    action_labels: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot action selection dynamics over time.
    
    Parameters
    ----------
    bg : BasalGanglia
        Basal ganglia module
    history : np.ndarray
        Action values over time (time x actions)
    action_labels : List[str], optional
        Labels for actions
    threshold : float, optional
        Selection threshold to display
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    Tuple[Figure, Axes]
        Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get dimensions
    n_steps, n_actions = history.shape
    
    # Create labels if not provided
    if action_labels is None:
        action_labels = [f"Action {i}" for i in range(n_actions)]
    
    # Plot each action's value over time
    time_steps = np.arange(n_steps)
    
    for i, label in enumerate(action_labels):
        ax.plot(time_steps, history[:, i], label=label, linewidth=2)
    
    # Add threshold line if provided
    if threshold is not None:
        ax.axhline(y=threshold, color='k', linestyle='--', 
                   alpha=0.5, label='Threshold')
    
    # Find and highlight selected actions
    selected = history.max(axis=1) > (threshold or 0.0)
    selected_indices = np.where(selected)[0]
    
    if len(selected_indices) > 0:
        selected_actions = history.argmax(axis=1)[selected_indices]
        ax.scatter(selected_indices, history[selected_indices, selected_actions],
                   color='red', s=100, marker='o', zorder=5,
                   label='Selected', edgecolors='black', linewidth=2)
    
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Action Value")
    ax.set_title("Action Selection Dynamics")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_network_graph(
    network: Network,
    layout: str = "spring",
    node_size: int = 300,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Visualize network connectivity graph.
    
    Parameters
    ----------
    network : Network
        SPA network to visualize
    layout : str
        Graph layout algorithm ('spring', 'circular', 'hierarchical')
    node_size : int
        Size of nodes
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    Tuple[Figure, Axes]
        Matplotlib figure and axes
    """
    if not NETWORKX_AVAILABLE:
        warnings.warn("NetworkX not available for network visualization")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "NetworkX required for network visualization",
                ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes for modules
    module_names = list(network.modules.keys())
    G.add_nodes_from(module_names)
    
    # Add edges for connections
    for conn in network.connections:
        G.add_edge(conn.source.name, conn.target.name,
                   weight=np.mean(np.abs(conn.transform)))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "hierarchical":
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    else:
        pos = nx.random_layout(G)
    
    # Draw nodes with different colors for different module types
    node_colors = []
    for name in module_names:
        module = network.modules[name]
        if hasattr(module, 'module_type'):
            if module.module_type == 'state':
                node_colors.append('lightblue')
            elif module.module_type == 'memory':
                node_colors.append('lightgreen')
            elif module.module_type == 'gate':
                node_colors.append('orange')
            else:
                node_colors.append('lightgray')
        else:
            node_colors.append('lightgray')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                          node_size=node_size, ax=ax)
    
    # Draw edges with width based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    if weights:
        # Normalize weights for edge width
        max_weight = max(weights) if weights else 1.0
        edge_widths = [3 * w / max_weight for w in weights]
        
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                              width=edge_widths, arrows=True,
                              arrowsize=20, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    
    ax.set_title("SPA Network Graph")
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Circle((0, 0), 1, facecolor='lightblue', label='State'),
        mpatches.Circle((0, 0), 1, facecolor='lightgreen', label='Memory'),
        mpatches.Circle((0, 0), 1, facecolor='orange', label='Gate'),
        mpatches.Circle((0, 0), 1, facecolor='lightgray', label='Other')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def visualize_production_flow(
    production_system: ProductionSystem,
    executed_productions: Optional[List[Production]] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Visualize production system flow and execution.
    
    Parameters
    ----------
    production_system : ProductionSystem
        Production system to visualize
    executed_productions : List[Production], optional
        List of executed productions to highlight
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    Tuple[Figure, Axes]
        Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get all productions
    productions = production_system.productions
    n_productions = len(productions)
    
    if n_productions == 0:
        ax.text(0.5, 0.5, "No productions in system",
                ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # Layout productions in a grid
    cols = int(np.ceil(np.sqrt(n_productions)))
    rows = int(np.ceil(n_productions / cols))
    
    # Track execution order
    executed_set = set(executed_productions) if executed_productions else set()
    
    # Draw each production
    for i, prod in enumerate(productions):
        row = i // cols
        col = i % cols
        
        # Position
        x = col * 3 + 1
        y = rows - row - 1
        
        # Determine color based on execution
        if prod in executed_set:
            color = 'lightgreen'
            edge_color = 'darkgreen'
            edge_width = 3
        else:
            color = 'lightgray'
            edge_color = 'black'
            edge_width = 1
        
        # Draw production box
        box = FancyBboxPatch(
            (x - 0.8, y - 0.3), 1.6, 0.6,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=edge_width
        )
        ax.add_patch(box)
        
        # Add production name
        ax.text(x, y, prod.name, ha='center', va='center',
                fontsize=10, weight='bold')
        
        # Add condition summary below
        condition_str = f"IF: {prod.condition.description[:30]}..."
        ax.text(x, y - 0.5, condition_str, ha='center', va='center',
                fontsize=8, style='italic')
        
        # Add effect summary above
        effect_str = f"THEN: {prod.effect.description[:30]}..."
        ax.text(x, y + 0.5, effect_str, ha='center', va='center',
                fontsize=8)
    
    # Draw execution order arrows if provided
    if executed_productions and len(executed_productions) > 1:
        for i in range(len(executed_productions) - 1):
            prod1 = executed_productions[i]
            prod2 = executed_productions[i + 1]
            
            # Find positions
            idx1 = productions.index(prod1)
            idx2 = productions.index(prod2)
            
            x1 = (idx1 % cols) * 3 + 1
            y1 = rows - (idx1 // cols) - 1
            x2 = (idx2 % cols) * 3 + 1
            y2 = rows - (idx2 // cols) - 1
            
            # Draw arrow
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='red',
                                     lw=2, alpha=0.7))
    
    # Set axis limits and remove ticks
    ax.set_xlim(-0.5, cols * 3 - 0.5)
    ax.set_ylim(-1, rows + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title("Production System Flow", fontsize=14, weight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor='lightgreen',
                          edgecolor='darkgreen', linewidth=3,
                          label='Executed'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='lightgray',
                          edgecolor='black', linewidth=1,
                          label='Not Executed')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def animate_state_evolution(
    states: List[np.ndarray],
    vocab: Optional[Vocabulary] = None,
    top_k: int = 5,
    interval: int = 100,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> FuncAnimation:
    """
    Animate semantic pointer evolution over time.
    
    Parameters
    ----------
    states : List[np.ndarray]
        List of state vectors over time
    vocab : Vocabulary, optional
        Vocabulary for interpreting states
    top_k : int
        Number of top matches to show
    interval : int
        Animation interval in milliseconds
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save animation
        
    Returns
    -------
    FuncAnimation
        Animation object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    n_steps = len(states)
    n_dims = len(states[0])
    
    # Initialize plots
    if vocab is not None:
        # Bar plot for vocabulary matches
        keys = list(vocab.pointers.keys())[:top_k]
        bars = ax1.bar(range(top_k), [0] * top_k)
        ax1.set_ylim(0, 1.1)
        ax1.set_xlabel("Semantic Pointer")
        ax1.set_ylabel("Similarity")
        ax1.set_title("Top Vocabulary Matches")
        ax1.set_xticks(range(top_k))
        ax1.set_xticklabels(keys, rotation=45, ha='right')
    else:
        # Line plot for raw vector
        line, = ax1.plot([], [], 'b-')
        ax1.set_xlim(0, min(100, n_dims))
        ax1.set_ylim(-2, 2)
        ax1.set_xlabel("Dimension")
        ax1.set_ylabel("Value")
        ax1.set_title("State Vector (First 100 Dimensions)")
    
    # Progress bar
    progress_bar = ax2.barh([0], [0], height=0.5)
    ax2.set_xlim(0, n_steps)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel("Time Step")
    ax2.set_yticks([])
    ax2.set_title("Progress")
    
    # Animation update function
    def update(frame):
        state = states[frame]
        
        if vocab is not None:
            # Update vocabulary matches
            similarities = []
            for key in vocab.pointers.keys():
                # Create a semantic pointer from the state vector
                state_pointer = SemanticPointer(state, vocabulary=vocab)
                sim = state_pointer.dot(vocab[key])
                similarities.append((key, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Update bars
            for i, bar in enumerate(bars):
                if i < len(similarities):
                    bar.set_height(similarities[i][1])
                    bar.set_label(similarities[i][0])
                else:
                    bar.set_height(0)
            
            # Update labels
            labels = [sim[0] for sim in similarities[:top_k]]
            ax1.set_xticklabels(labels, rotation=45, ha='right')
        else:
            # Update line plot
            dims_to_show = min(100, n_dims)
            line.set_data(range(dims_to_show), state[:dims_to_show])
        
        # Update progress bar
        progress_bar[0].set_width(frame + 1)
        
        # Update title with time step
        fig.suptitle(f"State Evolution - Step {frame + 1}/{n_steps}",
                     fontsize=14, weight='bold')
        
        return bars if vocab else [line], progress_bar
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_steps,
                        interval=interval, blit=False)
    
    plt.tight_layout()
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000/interval)
        else:
            anim.save(save_path, writer='ffmpeg', fps=1000/interval)
    
    return anim


def plot_module_activity(
    module: Module,
    activity_history: np.ndarray,
    time_window: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot module activity over time.
    
    Parameters
    ----------
    module : Module
        SPA module
    activity_history : np.ndarray
        Activity values over time
    time_window : Tuple[int, int], optional
        Time window to display
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    Tuple[Figure, Axes]
        Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle time window
    if time_window is not None:
        start, end = time_window
        activity = activity_history[start:end]
        time_steps = np.arange(start, end)
    else:
        activity = activity_history
        time_steps = np.arange(len(activity))
    
    # Plot activity
    if activity.ndim == 1:
        ax.plot(time_steps, activity, 'b-', linewidth=2)
        ax.fill_between(time_steps, 0, activity, alpha=0.3)
    else:
        # Multiple channels
        for i in range(activity.shape[1]):
            ax.plot(time_steps, activity[:, i],
                   label=f"Channel {i}", linewidth=2)
    
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Activity Level")
    ax.set_title(f"{module.name} Activity Over Time")
    ax.grid(True, alpha=0.3)
    
    if activity.ndim > 1:
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_vocabulary_structure(
    vocab: Vocabulary,
    method: str = "pca",
    n_components: int = 2,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Visualize vocabulary structure using dimensionality reduction.
    
    Parameters
    ----------
    vocab : Vocabulary
        Vocabulary to visualize
    method : str
        Reduction method ('pca', 'tsne', 'mds')
    n_components : int
        Number of components (2 or 3)
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    Tuple[Figure, Axes]
        Matplotlib figure and axes
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE, MDS
    
    # Get all vectors
    keys = list(vocab.pointers.keys())
    vectors = np.array([vocab[key].vector for key in keys])
    
    # Apply dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(vectors)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(vectors)
    elif method == "mds":
        reducer = MDS(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(vectors)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create plot
    if n_components == 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(reduced[:, 0], reduced[:, 1], s=100, alpha=0.6)
        
        # Add labels
        for i, key in enumerate(keys):
            ax.annotate(key, (reduced[i, 0], reduced[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8)
        
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
    else:
        # 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                  s=100, alpha=0.6)
        
        # Add labels
        for i, key in enumerate(keys):
            ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2],
                   key, fontsize=9, alpha=0.8)
        
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_zlabel(f"{method.upper()} Component 3")
    
    ax.set_title(f"Vocabulary Structure ({method.upper()})")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_interactive_network(
    network: Network,
    port: int = 8050
) -> None:
    """
    Create interactive network visualization using Plotly/Dash.
    
    Parameters
    ----------
    network : Network
        SPA network to visualize
    port : int
        Port for Dash server
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("Plotly not available for interactive visualization")
        return
    
    try:
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output
    except ImportError:
        warnings.warn("Dash not available for interactive visualization")
        return
    
    # Create network graph data
    if not NETWORKX_AVAILABLE:
        warnings.warn("NetworkX required for network graph creation")
        return
    
    # Build graph
    G = nx.DiGraph()
    module_names = list(network.modules.keys())
    G.add_nodes_from(module_names)
    
    for conn in network.connections:
        G.add_edge(conn.source.name, conn.target.name,
                   weight=np.mean(np.abs(conn.transform)))
    
    # Create Plotly figure
    pos = nx.spring_layout(G)
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=edge[2]['weight'] * 2, color='gray'),
            hoverinfo='none'
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition='bottom center',
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        hoverinfo='text'
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Interactive SPA Network",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    # Create Dash app
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("SPA Network Visualization"),
        dcc.Graph(id='network-graph', figure=fig),
        html.Div(id='node-info')
    ])
    
    @app.callback(
        Output('node-info', 'children'),
        Input('network-graph', 'clickData')
    )
    def display_node_info(clickData):
        if clickData is None:
            return "Click on a node to see details"
        
        point = clickData['points'][0]
        if 'text' in point:
            node_name = point['text']
            module = network.modules.get(node_name)
            if module:
                info = [
                    html.H3(f"Module: {node_name}"),
                    html.P(f"Type: {module.__class__.__name__}"),
                    html.P(f"Dimensions: {module.dimensions}")
                ]
                return info
        
        return "No information available"
    
    print(f"Starting interactive visualization at http://localhost:{port}")
    app.run_server(debug=False, port=port)