#!/usr/bin/env python3
"""
Graph Encoding Demo: Representing Graph Structures with VSA

This script demonstrates how to encode and manipulate graph structures
using Vector Symbolic Architectures, including:
- Node and edge representation
- Graph traversal and search
- Subgraph matching
- Graph transformations
- Knowledge graphs
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from cognitive_computing.vsa import (
    create_vsa, VSA, GraphEncoder,
    BinaryVector, BipolarVector
)


class GraphVSA:
    """Enhanced graph operations using VSA."""
    
    def __init__(self, dimension: int = 10000):
        """Initialize graph VSA system."""
        self.vsa = create_vsa(
            dimension=dimension,
            vector_type='bipolar',
            vsa_type='custom',
            binding_method='multiplication'
        )
        self.encoder = GraphEncoder(self.vsa)
        
        # Role vectors for graph structures
        self.roles = {
            'FROM': self.vsa.generate_vector(),
            'TO': self.vsa.generate_vector(),
            'TYPE': self.vsa.generate_vector(),
            'WEIGHT': self.vsa.generate_vector(),
            'LABEL': self.vsa.generate_vector(),
            'NEIGHBORS': self.vsa.generate_vector(),
            'PATH': self.vsa.generate_vector()
        }
        
        # Store role vectors in VSA memory
        for role_name, role_vec in self.roles.items():
            self.vsa.store(role_name, role_vec)
    
    def encode_directed_edge(self, source: str, target: str, 
                           edge_type: Optional[str] = None,
                           weight: Optional[float] = None) -> np.ndarray:
        """
        Encode a directed edge with optional type and weight.
        
        Parameters
        ----------
        source : str
            Source node
        target : str
            Target node
        edge_type : str, optional
            Type of edge (e.g., 'follows', 'likes')
        weight : float, optional
            Edge weight
            
        Returns
        -------
        np.ndarray
            Encoded edge vector
        """
        # Get or create node vectors
        source_vec = self.encoder._get_node_vector(source)
        target_vec = self.encoder._get_node_vector(target)
        
        # Basic edge: FROM + TO binding
        edge_components = [
            self.vsa.bind(self.roles['FROM'], source_vec),
            self.vsa.bind(self.roles['TO'], target_vec)
        ]
        
        # Add edge type if specified
        if edge_type:
            type_vec = self.vsa.generate_vector()
            self.vsa.store(f"edge_type_{edge_type}", type_vec)
            edge_components.append(
                self.vsa.bind(self.roles['TYPE'], type_vec)
            )
        
        # Add weight if specified
        if weight is not None:
            # Discretize weight to levels
            weight_level = int(weight * 10)  # 0-10 scale
            weight_vec = self.vsa.generate_vector()
            self.vsa.store(f"weight_{weight_level}", weight_vec)
            edge_components.append(
                self.vsa.bind(self.roles['WEIGHT'], weight_vec)
            )
        
        return self.vsa.bundle(edge_components)
    
    def encode_path(self, path: List[str]) -> np.ndarray:
        """
        Encode a path through the graph.
        
        Parameters
        ----------
        path : List[str]
            Sequence of nodes in the path
            
        Returns
        -------
        np.ndarray
            Encoded path vector
        """
        if len(path) < 2:
            return self.encoder._get_node_vector(path[0]) if path else np.zeros(self.vsa.config.dimension)
        
        # Encode path as sequence of transitions
        path_components = []
        
        # Add ordered nodes with position encoding
        for i, node in enumerate(path):
            node_vec = self.encoder._get_node_vector(node)
            # Use permutation for position encoding
            positioned_node = self.vsa.permute(node_vec, shift=i)
            path_components.append(positioned_node)
        
        # Add edge transitions
        for i in range(len(path) - 1):
            edge = self.encode_directed_edge(path[i], path[i+1])
            # Position encode the edge
            positioned_edge = self.vsa.permute(edge, shift=i*10)  # Different shift scale
            path_components.append(positioned_edge)
        
        return self.vsa.bundle(path_components)
    
    def find_neighbors(self, node: str, graph_vec: np.ndarray) -> List[Tuple[str, float]]:
        """
        Find neighbors of a node in the encoded graph.
        
        Parameters
        ----------
        node : str
            Node to find neighbors for
        graph_vec : np.ndarray
            Encoded graph
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (neighbor, similarity) pairs
        """
        node_vec = self.encoder._get_node_vector(node)
        
        # For each potential neighbor, check if edge exists
        neighbors = []
        
        for other_node, other_vec in self.encoder.node_vectors.items():
            if other_node != node:
                # Create the edge vector for this potential connection
                edge_query = self.vsa.bundle([
                    self.vsa.bind(self.roles['FROM'], node_vec),
                    self.vsa.bind(self.roles['TO'], other_vec)
                ])
                
                # Check similarity with graph
                edge_similarity = self.vsa.similarity(graph_vec, edge_query)
                
                # If similarity is high enough, edge exists
                if edge_similarity > 0.15:  # Threshold for edge presence
                    neighbors.append((other_node, edge_similarity))
        
        return sorted(neighbors, key=lambda x: x[1], reverse=True)
    
    def find_shortest_path(self, graph_vec: np.ndarray, start: str, end: str,
                          max_length: int = 5) -> Optional[List[str]]:
        """
        Find shortest path between nodes using VSA operations.
        
        This is a demonstration of graph search using VSA - for production
        use, traditional algorithms would be more efficient.
        
        Parameters
        ----------
        graph_vec : np.ndarray
            Encoded graph
        start : str
            Start node
        end : str
            End node
        max_length : int
            Maximum path length to search
            
        Returns
        -------
        List[str] or None
            Shortest path if found
        """
        if start == end:
            return [start]
        
        # BFS-like search using VSA
        visited = {start}
        paths = [[start]]
        
        for length in range(1, max_length + 1):
            new_paths = []
            
            for path in paths:
                current = path[-1]
                neighbors = self.find_neighbors(current, graph_vec)
                
                for neighbor, _ in neighbors:
                    if neighbor == end:
                        return path + [neighbor]
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_paths.append(path + [neighbor])
            
            paths = new_paths
            if not paths:
                break
        
        return None


def demonstrate_basic_graph_encoding():
    """Demonstrate basic graph encoding operations."""
    print("=== Basic Graph Encoding ===\n")
    
    # Create graph VSA system
    graph_vsa = GraphVSA(dimension=10000)
    
    # Example 1: Simple directed graph
    print("1. Encoding a Simple Social Network:")
    
    # Define edges (who follows whom)
    edges = [
        ("Alice", "Bob"),
        ("Bob", "Charlie"),
        ("Charlie", "Alice"),
        ("Bob", "David"),
        ("David", "Eve"),
        ("Eve", "Bob")
    ]
    
    # Encode edges
    edge_vectors = []
    for source, target in edges:
        edge_vec = graph_vsa.encode_directed_edge(source, target, edge_type="follows")
        edge_vectors.append(edge_vec)
        print(f"  {source} -> {target}")
    
    # Bundle into graph
    graph_vec = graph_vsa.vsa.bundle(edge_vectors)
    print(f"\nEncoded graph with {len(edges)} edges")
    
    # Find neighbors
    print("\n2. Finding Neighbors:")
    for person in ["Alice", "Bob", "Charlie"]:
        neighbors = graph_vsa.find_neighbors(person, graph_vec)
        print(f"  {person} follows: {[n[0] for n in neighbors]}")
    
    # Example 2: Weighted graph
    print("\n3. Encoding a Weighted Graph:")
    
    weighted_edges = [
        ("A", "B", 0.8),
        ("B", "C", 0.5),
        ("C", "D", 0.9),
        ("A", "D", 0.3),
        ("B", "D", 0.6)
    ]
    
    weighted_edge_vectors = []
    for source, target, weight in weighted_edges:
        edge_vec = graph_vsa.encode_directed_edge(
            source, target, 
            edge_type="connected",
            weight=weight
        )
        weighted_edge_vectors.append(edge_vec)
        print(f"  {source} -> {target} (weight: {weight})")
    
    weighted_graph = graph_vsa.vsa.bundle(weighted_edge_vectors)
    
    # Example 3: Path encoding
    print("\n4. Encoding Paths:")
    
    paths = [
        ["Alice", "Bob", "Charlie"],
        ["Alice", "Bob", "David", "Eve"],
        ["Eve", "Bob", "Charlie", "Alice"]
    ]
    
    path_vectors = []
    for path in paths:
        path_vec = graph_vsa.encode_path(path)
        path_vectors.append(path_vec)
        print(f"  Path: {' -> '.join(path)}")
    
    # Check path similarity
    print("\n5. Path Similarity:")
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            sim = graph_vsa.vsa.similarity(path_vectors[i], path_vectors[j])
            print(f"  Path {i+1} vs Path {j+1}: {sim:.3f}")


def demonstrate_knowledge_graph():
    """Demonstrate knowledge graph operations."""
    print("\n=== Knowledge Graph Demo ===\n")
    
    graph_vsa = GraphVSA(dimension=10000)
    
    # Create a simple knowledge graph
    print("1. Building a Knowledge Graph:")
    
    # Facts as (subject, relation, object)
    facts = [
        ("Python", "is_a", "Language"),
        ("Java", "is_a", "Language"),
        ("NumPy", "is_a", "Library"),
        ("Pandas", "is_a", "Library"),
        ("NumPy", "used_for", "NumericalComputing"),
        ("Pandas", "used_for", "DataAnalysis"),
        ("Python", "has_library", "NumPy"),
        ("Python", "has_library", "Pandas"),
        ("DataScience", "uses", "Python"),
        ("DataScience", "uses", "Pandas"),
        ("MachineLearning", "requires", "NumPy"),
        ("MachineLearning", "is_part_of", "DataScience")
    ]
    
    # Encode facts
    fact_vectors = []
    for subject, relation, obj in facts:
        # Encode as directed edge with relation type
        fact_vec = graph_vsa.encode_directed_edge(
            subject, obj, edge_type=relation
        )
        fact_vectors.append(fact_vec)
        print(f"  {subject} --[{relation}]--> {obj}")
    
    # Create knowledge graph
    knowledge_graph = graph_vsa.vsa.bundle(fact_vectors)
    
    # Query the knowledge graph
    print("\n2. Querying the Knowledge Graph:")
    
    # What libraries does Python have?
    python_vec = graph_vsa.encoder._get_node_vector("Python")
    has_library_vec = graph_vsa.vsa.memory.get("edge_type_has_library")
    
    if has_library_vec is not None:
        # Create query
        query = graph_vsa.vsa.bundle([
            graph_vsa.vsa.bind(graph_vsa.roles['FROM'], python_vec),
            graph_vsa.vsa.bind(graph_vsa.roles['TYPE'], has_library_vec)
        ])
        
        # Check similarity
        sim = graph_vsa.vsa.similarity(knowledge_graph, query)
        print(f"  Query 'Python has_library ?': similarity = {sim:.3f}")
    
    # Find related concepts
    print("\n3. Finding Related Concepts:")
    
    concepts = ["Python", "DataScience", "NumPy"]
    for concept in concepts:
        neighbors = graph_vsa.find_neighbors(concept, knowledge_graph)
        if neighbors:
            print(f"  {concept} is related to: {[n[0] for n in neighbors[:3]]}")


def demonstrate_graph_matching():
    """Demonstrate subgraph pattern matching."""
    print("\n=== Graph Pattern Matching ===\n")
    
    graph_vsa = GraphVSA(dimension=10000)
    
    # Create a larger graph
    print("1. Creating a Graph with Patterns:")
    
    # Define a graph with triangular patterns
    edges = [
        # Triangle 1: A-B-C
        ("A", "B"), ("B", "C"), ("C", "A"),
        # Triangle 2: D-E-F
        ("D", "E"), ("E", "F"), ("F", "D"),
        # Connection between triangles
        ("C", "D"),
        # Additional edges
        ("B", "G"), ("G", "H"), ("H", "I")
    ]
    
    edge_vectors = []
    for source, target in edges:
        edge_vec = graph_vsa.encode_directed_edge(source, target)
        edge_vectors.append(edge_vec)
    
    main_graph = graph_vsa.vsa.bundle(edge_vectors)
    print(f"  Created graph with {len(edges)} edges")
    
    # Create pattern to match (triangle)
    print("\n2. Creating Triangle Pattern:")
    triangle_pattern = graph_vsa.vsa.bundle([
        graph_vsa.encode_directed_edge("X", "Y"),
        graph_vsa.encode_directed_edge("Y", "Z"),
        graph_vsa.encode_directed_edge("Z", "X")
    ])
    
    # Create non-triangle pattern
    print("\n3. Creating Linear Pattern:")
    linear_pattern = graph_vsa.vsa.bundle([
        graph_vsa.encode_directed_edge("X", "Y"),
        graph_vsa.encode_directed_edge("Y", "Z")
    ])
    
    # Compare patterns
    print("\n4. Pattern Similarity with Main Graph:")
    triangle_sim = graph_vsa.vsa.similarity(main_graph, triangle_pattern)
    linear_sim = graph_vsa.vsa.similarity(main_graph, linear_pattern)
    
    print(f"  Triangle pattern similarity: {triangle_sim:.3f}")
    print(f"  Linear pattern similarity: {linear_sim:.3f}")
    print(f"  Triangle pattern is {'more' if triangle_sim > linear_sim else 'less'} prevalent")


def demonstrate_graph_transformations():
    """Demonstrate graph transformations using VSA."""
    print("\n=== Graph Transformations ===\n")
    
    graph_vsa = GraphVSA(dimension=10000)
    
    # Create initial graph
    print("1. Original Graph:")
    original_edges = [
        ("A", "B", "friend"),
        ("B", "C", "friend"),
        ("C", "D", "colleague"),
        ("D", "A", "colleague")
    ]
    
    original_vectors = []
    for source, target, rel_type in original_edges:
        edge_vec = graph_vsa.encode_directed_edge(source, target, edge_type=rel_type)
        original_vectors.append(edge_vec)
        print(f"  {source} --[{rel_type}]--> {target}")
    
    original_graph = graph_vsa.vsa.bundle(original_vectors)
    
    # Transform: Make graph bidirectional
    print("\n2. Bidirectional Transformation:")
    bidirectional_vectors = original_vectors.copy()
    
    for source, target, rel_type in original_edges:
        # Add reverse edge
        reverse_edge = graph_vsa.encode_directed_edge(target, source, edge_type=rel_type)
        bidirectional_vectors.append(reverse_edge)
        print(f"  Added: {target} --[{rel_type}]--> {source}")
    
    bidirectional_graph = graph_vsa.vsa.bundle(bidirectional_vectors)
    
    # Transform: Change relationship types
    print("\n3. Relationship Type Transformation:")
    
    # Create transformation: friend -> close_friend, colleague -> acquaintance
    transformed_vectors = []
    for source, target, rel_type in original_edges:
        new_type = "close_friend" if rel_type == "friend" else "acquaintance"
        edge_vec = graph_vsa.encode_directed_edge(source, target, edge_type=new_type)
        transformed_vectors.append(edge_vec)
        print(f"  {source} --[{rel_type} → {new_type}]--> {target}")
    
    transformed_graph = graph_vsa.vsa.bundle(transformed_vectors)
    
    # Compare graphs
    print("\n4. Graph Similarities:")
    print(f"  Original vs Bidirectional: {graph_vsa.vsa.similarity(original_graph, bidirectional_graph):.3f}")
    print(f"  Original vs Transformed: {graph_vsa.vsa.similarity(original_graph, transformed_graph):.3f}")
    print(f"  Bidirectional vs Transformed: {graph_vsa.vsa.similarity(bidirectional_graph, transformed_graph):.3f}")


def demonstrate_graph_composition():
    """Demonstrate composing multiple graphs."""
    print("\n=== Graph Composition ===\n")
    
    graph_vsa = GraphVSA(dimension=10000)
    
    # Create subgraphs
    print("1. Creating Component Graphs:")
    
    # Social network subgraph
    social_edges = [
        ("Alice", "Bob", "friend"),
        ("Bob", "Charlie", "friend"),
        ("Alice", "Charlie", "colleague")
    ]
    
    social_vectors = []
    print("\n  Social Network:")
    for s, t, r in social_edges:
        vec = graph_vsa.encode_directed_edge(s, t, edge_type=r)
        social_vectors.append(vec)
        print(f"    {s} --[{r}]--> {t}")
    
    social_graph = graph_vsa.vsa.bundle(social_vectors)
    
    # Work hierarchy subgraph
    work_edges = [
        ("Alice", "Project1", "works_on"),
        ("Bob", "Project1", "works_on"),
        ("Charlie", "Project2", "works_on"),
        ("Project1", "DepartmentA", "belongs_to"),
        ("Project2", "DepartmentA", "belongs_to")
    ]
    
    work_vectors = []
    print("\n  Work Hierarchy:")
    for s, t, r in work_edges:
        vec = graph_vsa.encode_directed_edge(s, t, edge_type=r)
        work_vectors.append(vec)
        print(f"    {s} --[{r}]--> {t}")
    
    work_graph = graph_vsa.vsa.bundle(work_vectors)
    
    # Compose graphs
    print("\n2. Composing Graphs:")
    
    # Union (include both)
    union_graph = graph_vsa.vsa.bundle([social_graph, work_graph])
    print("  Created union of social and work graphs")
    
    # Weighted composition
    weighted_graph = graph_vsa.vsa.bundle(
        [social_graph, work_graph],
        weights=[0.7, 0.3]  # Emphasize social connections
    )
    print("  Created weighted composition (70% social, 30% work)")
    
    # Query composed graph
    print("\n3. Querying Composed Graph:")
    
    # Find Alice's connections in the composed graph
    alice_neighbors = graph_vsa.find_neighbors("Alice", union_graph)
    print(f"  Alice's connections: {[n[0] for n in alice_neighbors]}")
    
    # Check if work relationships are preserved
    project1_vec = graph_vsa.encoder._get_node_vector("Project1")
    works_on_vec = graph_vsa.vsa.memory.get("edge_type_works_on")
    
    if works_on_vec is not None:
        work_query = graph_vsa.vsa.bind(graph_vsa.roles['TYPE'], works_on_vec)
        work_sim = graph_vsa.vsa.similarity(union_graph, work_query)
        print(f"  'works_on' relationships preserved: {work_sim:.3f}")


def visualize_graph_example():
    """Create a visual example of graph encoding."""
    print("\n=== Visualizing Graph Encoding ===\n")
    
    # Create a simple graph for visualization
    G = nx.DiGraph()
    edges = [
        ("A", "B"), ("B", "C"), ("C", "D"),
        ("D", "A"), ("B", "D"), ("A", "C")
    ]
    G.add_edges_from(edges)
    
    # Create layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original graph
    ax1.set_title("Original Graph Structure")
    nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue',
            node_size=1000, font_size=16, arrows=True,
            edge_color='gray', arrowsize=20)
    
    # Create VSA encoding visualization
    ax2.set_title("VSA Encoding Process")
    ax2.text(0.5, 0.9, "Graph → VSA Vectors", ha='center', fontsize=14, 
             transform=ax2.transAxes, weight='bold')
    
    # Show encoding steps
    steps = [
        "1. Each node → Random vector",
        "2. Each edge → Bind(FROM, source) ⊕ Bind(TO, target)",
        "3. Graph → Bundle all edge vectors",
        "4. Properties preserved:",
        "   • Node connectivity",
        "   • Path information",
        "   • Subgraph patterns"
    ]
    
    for i, step in enumerate(steps):
        ax2.text(0.1, 0.7 - i*0.1, step, transform=ax2.transAxes, fontsize=10)
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('graph_encoding_visualization.png', dpi=150, bbox_inches='tight')
    print("  Saved visualization to 'graph_encoding_visualization.png'")
    
    # Show statistics
    graph_vsa = GraphVSA(dimension=10000)
    edge_vectors = []
    for source, target in edges:
        edge_vec = graph_vsa.encode_directed_edge(source, target)
        edge_vectors.append(edge_vec)
    
    graph_vec = graph_vsa.vsa.bundle(edge_vectors)
    
    print(f"\n  Graph Statistics:")
    print(f"    Nodes: {G.number_of_nodes()}")
    print(f"    Edges: {G.number_of_edges()}")
    print(f"    Vector dimension: {graph_vsa.vsa.config.dimension}")
    print(f"    Memory per edge: ~{graph_vsa.vsa.config.dimension * 8 / 1024:.1f} KB")
    print(f"    Total graph encoding: ~{graph_vsa.vsa.config.dimension * 8 / 1024:.1f} KB")


def main():
    """Run all graph encoding demonstrations."""
    print("\n" + "="*60)
    print("VSA GRAPH ENCODING DEMONSTRATION")
    print("="*60 + "\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstrations
    demonstrate_basic_graph_encoding()
    print("\n" + "-"*60 + "\n")
    
    demonstrate_knowledge_graph()
    print("\n" + "-"*60 + "\n")
    
    demonstrate_graph_matching()
    print("\n" + "-"*60 + "\n")
    
    demonstrate_graph_transformations()
    print("\n" + "-"*60 + "\n")
    
    demonstrate_graph_composition()
    print("\n" + "-"*60 + "\n")
    
    visualize_graph_example()
    
    # Summary
    print("\n=== Summary ===\n")
    print("VSA enables powerful graph operations:")
    print("• Encode nodes, edges, and paths as vectors")
    print("• Preserve graph structure in distributed representation")
    print("• Query graphs using vector operations")
    print("• Compose and transform graphs algebraically")
    print("• Match patterns and subgraphs")
    print("• Build and query knowledge graphs")
    
    print("\nKey advantages:")
    print("• Fixed-size representation regardless of graph size")
    print("• Supports approximate matching and similarity")
    print("• Compositional: can merge and transform graphs")
    print("• Fault-tolerant: robust to noise and errors")
    
    print("\n" + "="*60)
    print("Graph Encoding Demo Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()