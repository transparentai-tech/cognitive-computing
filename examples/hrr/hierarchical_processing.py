#!/usr/bin/env python3
"""
Hierarchical processing demonstration using HRR.

This example demonstrates:
1. Tree structure encoding
2. Nested data representation
3. Recursive retrieval
4. Part-whole relationships
5. Organizational hierarchies
6. File system representations
7. Concept taxonomies
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle

from cognitive_computing.hrr import (
    create_hrr,
    HRR,
    HierarchicalEncoder,
    CleanupMemory,
    CleanupMemoryConfig,
    generate_random_vector,
    generate_unitary_vector,
    plot_similarity_matrix,
)


class HierarchicalProcessor:
    """A hierarchical data processing system using HRR."""
    
    def __init__(self, dimension: int = 1024):
        """Initialize the hierarchical processor."""
        self.dimension = dimension
        self.hrr = create_hrr(dimension=dimension)
        self.encoder = HierarchicalEncoder(self.hrr)
        
        # Cleanup memory for nodes
        self.node_memory = CleanupMemory(
            CleanupMemoryConfig(threshold=0.3),
            dimension
        )
        
        # Role vectors for relationships
        self.roles = {
            "parent": generate_unitary_vector(dimension),
            "child": generate_unitary_vector(dimension),
            "sibling": generate_unitary_vector(dimension),
            "contains": generate_unitary_vector(dimension),
            "type": generate_unitary_vector(dimension),
            "value": generate_unitary_vector(dimension),
        }
        
    def add_node(self, name: str, vector: Optional[np.ndarray] = None) -> np.ndarray:
        """Add a node to the vocabulary."""
        if vector is None:
            vector = generate_random_vector(self.dimension)
        self.node_memory.add_item(name, vector)
        return vector
    
    def _register_tree_items(self, tree: Dict[str, Any]):
        """Recursively register all items in a tree."""
        for key, value in tree.items():
            # Register the key
            if isinstance(key, str) and key not in self.hrr.memory:
                vector = self.add_node(key)
                self.hrr.add_item(key, vector)
            
            # Handle the value
            if isinstance(value, dict):
                # Recursively process nested dictionary
                self._register_tree_items(value)
            elif isinstance(value, str):
                # Register string value
                if value not in self.hrr.memory:
                    vector = self.add_node(value)
                    self.hrr.add_item(value, vector)
    
    def encode_tree(self, tree: Dict[str, Any]) -> np.ndarray:
        """Encode a tree structure."""
        # Auto-register all items in the tree
        self._register_tree_items(tree)
        return self.encoder.encode_tree(tree)
    
    def query_path(self, encoding: np.ndarray, path: List[str]) -> Tuple[str, float]:
        """Query a specific path in the hierarchy."""
        result = self.encoder.decode_path(encoding, path)
        
        # If result is a leaf node, clean it up
        try:
            name, _, confidence = self.node_memory.cleanup(result)
            return name, confidence
        except ValueError:
            return "Unknown", 0.0
    
    def visualize_tree(self, tree: Dict[str, Any], title: str = "Tree Structure"):
        """Visualize a tree structure."""
        G = nx.DiGraph()
        
        def add_nodes(node: Dict[str, Any], parent: Optional[str] = None, prefix: str = ""):
            """Recursively add nodes to graph."""
            node_id = prefix + str(id(node))
            
            if isinstance(node, dict):
                label = node.get('name', 'node')
                G.add_node(node_id, label=label)
                
                if parent:
                    G.add_edge(parent, node_id)
                
                # Add children
                for key, child in node.items():
                    if key not in ['name', 'type', 'value']:
                        add_nodes(child, node_id, prefix + key + "_")
            else:
                # Leaf node
                G.add_node(node_id, label=str(node))
                if parent:
                    G.add_edge(parent, node_id)
        
        add_nodes(tree)
        
        # Draw the tree
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, labels=labels, with_labels=True, 
                node_color='lightblue', node_size=2000,
                font_size=10, arrows=True, arrowsize=20)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def demonstrate_basic_hierarchy():
    """Demonstrate basic hierarchical encoding."""
    print("\n" + "="*60)
    print("1. BASIC HIERARCHICAL ENCODING")
    print("="*60)
    
    processor = HierarchicalProcessor(dimension=1024)
    
    # Create a simple organizational hierarchy
    org_chart = {
        "name": "Company",
        "CEO": {
            "name": "Alice",
            "CTO": {
                "name": "Bob",
                "Engineer1": {"name": "Charlie"},
                "Engineer2": {"name": "David"}
            },
            "CFO": {
                "name": "Eve",
                "Accountant": {"name": "Frank"}
            }
        }
    }
    
    print("Organizational hierarchy:")
    print("  Company")
    print("    └── CEO: Alice")
    print("        ├── CTO: Bob")
    print("        │   ├── Engineer1: Charlie")
    print("        │   └── Engineer2: David")
    print("        └── CFO: Eve")
    print("            └── Accountant: Frank")
    
    # Encode the hierarchy (items will be auto-registered)
    encoded = processor.encode_tree(org_chart)
    
    # Query different paths
    queries = [
        (["CEO", "name"], "CEO's name"),
        (["CEO", "CTO", "name"], "CTO's name"),
        (["CEO", "CTO", "Engineer1", "name"], "Engineer1's name"),
        (["CEO", "CFO", "Accountant", "name"], "Accountant's name"),
    ]
    
    print("\nQuerying the hierarchy:")
    for path, description in queries:
        result, confidence = processor.query_path(encoded, path)
        print(f"  {description} ({' → '.join(path)}): {result} (conf: {confidence:.3f})")
    
    # Visualize the tree
    processor.visualize_tree(org_chart, "Organizational Chart")


def demonstrate_file_system():
    """Demonstrate file system hierarchy encoding."""
    print("\n" + "="*60)
    print("2. FILE SYSTEM HIERARCHY")
    print("="*60)
    
    processor = HierarchicalProcessor(dimension=1024)
    
    # Create a file system structure
    file_system = {
        "name": "root",
        "type": "directory",
        "home": {
            "name": "home",
            "type": "directory",
            "user": {
                "name": "user",
                "type": "directory",
                "documents": {
                    "name": "documents",
                    "type": "directory",
                    "report.txt": {
                        "name": "report.txt",
                        "type": "file",
                        "size": "10KB"
                    },
                    "data.csv": {
                        "name": "data.csv",
                        "type": "file",
                        "size": "25KB"
                    }
                },
                "pictures": {
                    "name": "pictures",
                    "type": "directory",
                    "vacation.jpg": {
                        "name": "vacation.jpg",
                        "type": "file",
                        "size": "2MB"
                    }
                }
            }
        },
        "etc": {
            "name": "etc",
            "type": "directory",
            "config": {
                "name": "config",
                "type": "file",
                "size": "1KB"
            }
        }
    }
    
    # Encode the file system
    encoded = processor.encode_tree(file_system)
    
    print("File system structure encoded")
    
    # Query file information
    queries = [
        (["home", "user", "documents", "report.txt", "type"], "report.txt type"),
        (["home", "user", "documents", "data.csv", "size"], "data.csv size"),
        (["home", "user", "pictures", "vacation.jpg", "name"], "vacation.jpg name"),
        (["etc", "config", "type"], "config type"),
    ]
    
    print("\nQuerying file system:")
    for path, description in queries:
        try:
            result, confidence = processor.query_path(encoded, path)
            print(f"  {description}: {result} (conf: {confidence:.3f})")
        except:
            print(f"  {description}: Not found")
    
    # Visualize the file system
    processor.visualize_tree(file_system, "File System Structure")


def demonstrate_concept_taxonomy():
    """Demonstrate concept taxonomy encoding."""
    print("\n" + "="*60)
    print("3. CONCEPT TAXONOMY")
    print("="*60)
    
    processor = HierarchicalProcessor(dimension=1024)
    
    # Create an animal taxonomy
    taxonomy = {
        "name": "Animal",
        "Mammal": {
            "name": "Mammal",
            "Primate": {
                "name": "Primate",
                "Human": {"name": "Human", "legs": "2"},
                "Chimpanzee": {"name": "Chimpanzee", "legs": "2"}
            },
            "Carnivore": {
                "name": "Carnivore",
                "Cat": {"name": "Cat", "legs": "4"},
                "Dog": {"name": "Dog", "legs": "4"}
            }
        },
        "Bird": {
            "name": "Bird",
            "Raptor": {
                "name": "Raptor",
                "Eagle": {"name": "Eagle", "wingspan": "large"},
                "Hawk": {"name": "Hawk", "wingspan": "medium"}
            },
            "Songbird": {
                "name": "Songbird",
                "Robin": {"name": "Robin", "wingspan": "small"},
                "Sparrow": {"name": "Sparrow", "wingspan": "small"}
            }
        }
    }
    
    # Encode the taxonomy
    encoded = processor.encode_tree(taxonomy)
    
    print("Animal taxonomy encoded")
    
    # Test inheritance-like queries
    print("\nQuerying taxonomy:")
    
    # Direct queries
    queries = [
        (["Mammal", "Primate", "Human", "legs"], "Human legs"),
        (["Bird", "Raptor", "Eagle", "wingspan"], "Eagle wingspan"),
        (["Mammal", "Carnivore", "Cat", "name"], "Cat name"),
    ]
    
    for path, description in queries:
        result, confidence = processor.query_path(encoded, path)
        print(f"  {description}: {result} (conf: {confidence:.3f})")
    
    # Visualize the taxonomy
    processor.visualize_tree(taxonomy, "Animal Taxonomy")


def demonstrate_nested_structures():
    """Demonstrate deeply nested structures."""
    print("\n" + "="*60)
    print("4. NESTED DATA STRUCTURES")
    print("="*60)
    
    processor = HierarchicalProcessor(dimension=1024)
    
    # Create a nested data structure (like JSON)
    data = {
        "name": "config",
        "database": {
            "name": "database",
            "connection": {
                "name": "connection",
                "host": {"value": "localhost"},
                "port": {"value": "5432"},
                "credentials": {
                    "name": "credentials",
                    "username": {"value": "admin"},
                    "password": {"value": "secret"}
                }
            },
            "settings": {
                "name": "settings",
                "pool_size": {"value": "10"},
                "timeout": {"value": "30"}
            }
        },
        "application": {
            "name": "application",
            "server": {
                "name": "server",
                "host": {"value": "0.0.0.0"},
                "port": {"value": "8080"}
            },
            "features": {
                "name": "features",
                "auth": {"value": "enabled"},
                "cache": {"value": "redis"}
            }
        }
    }
    
    # Encode the structure
    encoded = processor.encode_tree(data)
    
    print("Configuration structure encoded")
    
    # Query nested values
    queries = [
        (["database", "connection", "host", "value"], "DB host"),
        (["database", "connection", "credentials", "username", "value"], "DB username"),
        (["application", "server", "port", "value"], "Server port"),
        (["application", "features", "cache", "value"], "Cache type"),
    ]
    
    print("\nQuerying nested configuration:")
    for path, description in queries:
        result, confidence = processor.query_path(encoded, path)
        print(f"  {description}: {result} (conf: {confidence:.3f})")
        print(f"    Path: {' → '.join(path)}")


def demonstrate_part_whole_relationships():
    """Demonstrate part-whole relationships."""
    print("\n" + "="*60)
    print("5. PART-WHOLE RELATIONSHIPS")
    print("="*60)
    
    processor = HierarchicalProcessor(dimension=1024)
    
    # Create a part-whole hierarchy (e.g., a car)
    car = {
        "name": "Car",
        "Engine": {
            "name": "Engine",
            "type": "V6",
            "Cylinders": {
                "name": "Cylinders",
                "count": "6",
                "Piston": {"name": "Piston", "material": "aluminum"},
                "Valve": {"name": "Valve", "type": "overhead"}
            },
            "Cooling": {
                "name": "Cooling",
                "Radiator": {"name": "Radiator", "capacity": "2L"},
                "Fan": {"name": "Fan", "type": "electric"}
            }
        },
        "Transmission": {
            "name": "Transmission",
            "type": "automatic",
            "Gears": {"name": "Gears", "count": "6"},
            "Clutch": {"name": "Clutch", "type": "hydraulic"}
        },
        "Wheels": {
            "name": "Wheels",
            "count": "4",
            "Tire": {"name": "Tire", "type": "radial"},
            "Rim": {"name": "Rim", "material": "alloy"}
        }
    }
    
    # Encode the structure
    encoded = processor.encode_tree(car)
    
    print("Car part-whole structure encoded")
    
    # Query parts
    queries = [
        (["Engine", "type"], "Engine type"),
        (["Engine", "Cylinders", "count"], "Number of cylinders"),
        (["Engine", "Cylinders", "Piston", "material"], "Piston material"),
        (["Transmission", "type"], "Transmission type"),
        (["Wheels", "Tire", "type"], "Tire type"),
    ]
    
    print("\nQuerying car parts:")
    for path, description in queries:
        result, confidence = processor.query_path(encoded, path)
        print(f"  {description}: {result} (conf: {confidence:.3f})")
    
    # Visualize the part-whole structure
    processor.visualize_tree(car, "Car Part-Whole Structure")


def demonstrate_recursive_structures():
    """Demonstrate recursive structure handling."""
    print("\n" + "="*60)
    print("6. RECURSIVE STRUCTURES")
    print("="*60)
    
    processor = HierarchicalProcessor(dimension=1024)
    
    # Create a recursive structure (simplified for HRR)
    # Note: True recursion is challenging in HRR, so we simulate with depth limit
    linked_list = {
        "name": "Node1",
        "value": "A",
        "next": {
            "name": "Node2",
            "value": "B",
            "next": {
                "name": "Node3",
                "value": "C",
                "next": {
                    "name": "Node4",
                    "value": "D",
                    "next": {"name": "null"}
                }
            }
        }
    }
    
    # Encode the structure
    encoded = processor.encode_tree(linked_list)
    
    print("Linked list structure encoded")
    
    # Traverse the list
    print("\nTraversing linked list:")
    
    current_path = []
    for i in range(5):
        if i == 0:
            value_path = ["value"]
        else:
            value_path = ["next"] * i + ["value"]
        
        try:
            result, confidence = processor.query_path(encoded, value_path)
            if confidence > 0.2:
                print(f"  Node {i+1}: {result} (conf: {confidence:.3f})")
            else:
                break
        except:
            break


def demonstrate_hierarchy_comparison():
    """Compare different hierarchical structures."""
    print("\n" + "="*60)
    print("7. HIERARCHY COMPARISON")
    print("="*60)
    
    processor = HierarchicalProcessor(dimension=1024)
    
    # Create two similar organizational structures
    org1 = {
        "name": "Company1",
        "CEO": {
            "name": "Alice",
            "CTO": {"name": "Bob"},
            "CFO": {"name": "Charlie"}
        }
    }
    
    org2 = {
        "name": "Company2",
        "CEO": {
            "name": "David",
            "CTO": {"name": "Eve"},
            "CFO": {"name": "Frank"}
        }
    }
    
    org3 = {
        "name": "Company3",
        "President": {
            "name": "George",
            "VP_Tech": {"name": "Helen"},
            "VP_Finance": {"name": "Ivan"}
        }
    }
    
    # Encode all structures
    encoded1 = processor.encode_tree(org1)
    encoded2 = processor.encode_tree(org2)
    encoded3 = processor.encode_tree(org3)
    
    # Compare structures
    structures = {
        "Company1": encoded1,
        "Company2": encoded2,
        "Company3": encoded3
    }
    
    print("Comparing organizational structures:")
    print("  Company1 and Company2: Same structure, different people")
    print("  Company3: Different structure")
    
    # Create similarity matrix
    fig = plot_similarity_matrix(structures)
    plt.title("Organizational Structure Similarity")
    plt.tight_layout()
    plt.show()


def demonstrate_dynamic_hierarchy():
    """Demonstrate building and modifying hierarchies dynamically."""
    print("\n" + "="*60)
    print("8. DYNAMIC HIERARCHY BUILDING")
    print("="*60)
    
    processor = HierarchicalProcessor(dimension=1024)
    
    # Start with a simple hierarchy
    hierarchy = {
        "name": "Root",
        "A": {"name": "A", "value": "1"}
    }
    
    print("Initial hierarchy: Root → A")
    
    # Encode initial state
    encoded1 = processor.encode_tree(hierarchy)
    
    # Add more nodes
    hierarchy["B"] = {"name": "B", "value": "2"}
    hierarchy["A"]["A1"] = {"name": "A1", "value": "1.1"}
    
    print("After adding nodes: Root → A → A1")
    print("                         → B")
    
    # Encode modified state
    encoded2 = processor.encode_tree(hierarchy)
    
    # Compare before and after
    similarity = processor.hrr.similarity(encoded1, encoded2)
    print(f"\nSimilarity between versions: {similarity:.3f}")
    
    # Query new structure
    queries = [
        (["A", "value"], "A's value"),
        (["B", "value"], "B's value"),
        (["A", "A1", "value"], "A1's value"),
    ]
    
    print("\nQuerying modified hierarchy:")
    for path, description in queries:
        try:
            result, confidence = processor.query_path(encoded2, path)
            print(f"  {description}: {result} (conf: {confidence:.3f})")
        except:
            print(f"  {description}: Not found")


def main():
    """Run all hierarchical processing demonstrations."""
    print("="*60)
    print("HIERARCHICAL PROCESSING WITH HRR DEMONSTRATION")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstrations
    demonstrate_basic_hierarchy()
    demonstrate_file_system()
    demonstrate_concept_taxonomy()
    demonstrate_nested_structures()
    demonstrate_part_whole_relationships()
    demonstrate_recursive_structures()
    demonstrate_hierarchy_comparison()
    demonstrate_dynamic_hierarchy()
    
    print("\n" + "="*60)
    print("All demonstrations complete!")
    print("="*60)


if __name__ == "__main__":
    main()