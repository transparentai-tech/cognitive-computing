#!/usr/bin/env python3
"""
Data Encoding Demo: Encoding Various Data Types with VSA

This script demonstrates how to encode different types of data into VSA vectors:
- Text and sequences
- Numerical values (discrete and continuous)
- Spatial data (2D/3D coordinates)
- Temporal data (time series)
- Graph structures
- Images (simplified representation)
- Structured records
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from cognitive_computing.vsa import (
    create_vsa, VSAConfig,
    RandomIndexingEncoder, SpatialEncoder,
    TemporalEncoder, LevelEncoder, GraphEncoder
)


def demonstrate_text_encoding():
    """Demonstrate text and sequence encoding."""
    print("=== Text and Sequence Encoding ===\n")
    
    # Create VSA instance
    vsa = create_vsa(
        dimension=10000,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )
    
    # Example 1: Character-level encoding
    print("1. Character-Level Encoding:")
    
    # Create character vectors
    chars = {}
    for c in 'abcdefghijklmnopqrstuvwxyz ':
        chars[c] = vsa.generate_vector()
    
    # Encode word "hello"
    word = "hello"
    char_vectors = []
    for i, c in enumerate(word):
        # Bind character with position
        pos_vec = vsa.generate_vector()
        char_vec = chars[c]
        char_vectors.append(vsa.bind(pos_vec, char_vec))
    
    word_vec = vsa.bundle(char_vectors)
    
    # Decode characters
    print(f"   Encoded word: '{word}'")
    print("   Decoded characters:")
    # Store position vectors for decoding
    pos_vecs = [vsa.generate_vector() for _ in range(len(word))]
    char_vectors = []
    for i, c in enumerate(word):
        char_vectors.append(vsa.bind(pos_vecs[i], chars[c]))
    
    word_vec = vsa.bundle(char_vectors)
    
    # Decode characters
    print(f"   Encoded word: '{word}'")
    print("   Decoded characters:")
    for i in range(len(word)):
        pos_vec = pos_vecs[i]
        decoded = vsa.unbind(word_vec, pos_vec)
        
        # Find best matching character
        best_char = None
        best_sim = -1
        for c, vec in chars.items():
            sim = vsa.similarity(decoded, vec)
            if sim > best_sim:
                best_sim = sim
                best_char = c
        print(f"   Position {i}: '{best_char}' (sim: {best_sim:.3f})")
    
    # Example 2: Word-level encoding with n-grams
    print("\n2. Word-Level Encoding with N-grams:")
    
    # Random indexing encoder
    ri_encoder = RandomIndexingEncoder(vsa, num_indices=10, window_size=3)
    
    # Encode sentence
    sentence = "the quick brown fox jumps"
    words = sentence.split()
    
    # Create word vectors
    word_vecs = {}
    for word in set(words):
        word_vecs[word] = ri_encoder.encode(word)
    
    # Create sentence representation
    sentence_vecs = []
    for i, word in enumerate(words):
        # Position encoding using permutation
        word_positioned = vsa.permute(word_vecs[word], shift=i)
        sentence_vecs.append(word_positioned)
    sentence_vec = vsa.bundle(sentence_vecs)
    
    print(f"   Encoded sentence: '{sentence}'")
    print(f"   Sentence vector norm: {np.linalg.norm(sentence_vec):.3f}")
    
    # Example 3: Sequence encoding with different methods
    print("\n3. Sequence Encoding Methods:")
    
    sequence = ['first', 'second', 'third', 'fourth']
    
    # Create vectors for sequence items
    seq_vecs = {item: vsa.generate_vector() for item in sequence}
    
    # Positional encoding using permutation
    pos_encoded_vecs = []
    for i, item in enumerate(sequence):
        # Use permutation for position encoding
        pos_vec = vsa.permute(seq_vecs[item], shift=i)
        pos_encoded_vecs.append(pos_vec)
    pos_encoded = vsa.bundle(pos_encoded_vecs)
    
    # Chaining encoding (bind adjacent items)
    chain_encoded_vecs = []
    for i in range(len(sequence) - 1):
        pair = vsa.bind(seq_vecs[sequence[i]], seq_vecs[sequence[i+1]])
        chain_encoded_vecs.append(pair)
    chain_encoded = vsa.bundle(chain_encoded_vecs) if chain_encoded_vecs else seq_vecs[sequence[0]]
    
    print("   Positional encoding: preserves absolute positions")
    print("   Chaining encoding: preserves sequential relationships\n")


def demonstrate_numerical_encoding():
    """Demonstrate encoding of numerical values."""
    print("=== Numerical Value Encoding ===\n")
    
    vsa = create_vsa(
        dimension=10000,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )
    
    # Example 1: Discrete value encoding
    print("1. Discrete Value Encoding:")
    
    # Encode integers 0-9
    int_vectors = {}
    for i in range(10):
        int_vectors[i] = vsa.generate_vector()
        vsa.store(f'int_{i}', int_vectors[i])
    
    # Create role vectors
    tens_role = vsa.generate_vector()
    ones_role = vsa.generate_vector()
    vsa.store('tens', tens_role)
    vsa.store('ones', ones_role)
    
    # Create composite number (e.g., 42)
    tens = vsa.bind(tens_role, int_vectors[4])
    ones = vsa.bind(ones_role, int_vectors[2])
    number_42 = vsa.bundle([tens, ones])
    
    # Decode
    decoded_tens = vsa.unbind(number_42, tens_role)
    decoded_ones = vsa.unbind(number_42, ones_role)
    
    print("   Encoded: 42")
    for digit, vec in int_vectors.items():
        tens_sim = vsa.similarity(decoded_tens, vec)
        ones_sim = vsa.similarity(decoded_ones, vec)
        if tens_sim > 0.7:
            print(f"   Tens place: {digit} (sim: {tens_sim:.3f})")
        if ones_sim > 0.7:
            print(f"   Ones place: {digit} (sim: {ones_sim:.3f})")
    
    # Example 2: Continuous value encoding with levels
    print("\n2. Continuous Value Encoding (Level-based):")
    
    level_encoder = LevelEncoder(vsa, num_levels=10, value_range=(0.0, 100.0))
    
    # Encode different temperatures
    temperatures = [15.5, 25.0, 35.8, 72.3]
    
    print("   Temperature encoding:")
    for temp in temperatures:
        temp_vec = level_encoder.encode(temp)
        # Decode by checking similarity to level vectors
        decoded_value = level_encoder.decode(temp_vec)
        print(f"   {temp}°C → decoded as ~{decoded_value:.1f}°C")
    
    # Example 3: Magnitude and phase encoding
    print("\n3. Magnitude-Phase Encoding:")
    
    # For complex numbers or 2D vectors
    magnitude = 5.0
    angle = np.pi / 4  # 45 degrees
    
    # Encode magnitude (discretized)
    mag_levels = np.linspace(0, 10, 20)
    mag_idx = np.argmin(np.abs(mag_levels - magnitude))
    mag_vec = vsa.generate_vector()
    vsa.store(f'mag_{mag_idx}', mag_vec)
    
    # Encode angle (discretized)
    angle_levels = np.linspace(0, 2*np.pi, 36)  # 10-degree steps
    angle_idx = np.argmin(np.abs(angle_levels - angle))
    angle_vec = vsa.generate_vector()
    vsa.store(f'angle_{angle_idx}', angle_vec)
    
    # Create role vectors for magnitude and phase
    mag_role = vsa.generate_vector()
    phase_role = vsa.generate_vector()
    vsa.store('MAG', mag_role)
    vsa.store('PHASE', phase_role)
    
    # Combine
    complex_vec = vsa.bundle([
        vsa.bind(mag_role, mag_vec),
        vsa.bind(phase_role, angle_vec)
    ])
    
    print(f"   Encoded complex number: {magnitude} * e^(i*{angle:.2f})")
    print(f"   Magnitude level: {mag_idx}, Angle level: {angle_idx}\n")


def demonstrate_spatial_encoding():
    """Demonstrate spatial data encoding."""
    print("=== Spatial Data Encoding ===\n")
    
    vsa = create_vsa(
        dimension=10000,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )
    
    # Example 1: 2D coordinate encoding
    print("1. 2D Coordinate Encoding:")
    
    # Use use_fourier=False to ensure vectors match VSA dimension
    spatial_encoder = SpatialEncoder(vsa, grid_size=(10, 10), use_fourier=False)
    
    # Encode points
    points = [(2, 3), (5, 7), (8, 1)]
    point_vectors = []
    
    for x, y in points:
        point_vec = spatial_encoder.encode([x, y])
        point_vectors.append(point_vec)
        print(f"   Encoded point ({x}, {y})")
    
    # Test spatial relationships
    print("\n   Spatial similarities:")
    for i, (p1, v1) in enumerate(zip(points, point_vectors)):
        for j, (p2, v2) in enumerate(zip(points, point_vectors)):
            if i < j:
                sim = vsa.similarity(v1, v2)
                dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                print(f"   Points {p1} and {p2}: similarity={sim:.3f}, distance={dist:.2f}")
    
    # Example 2: Grid-based scene encoding
    print("\n2. Grid-Based Scene Encoding:")
    
    # Create a simple scene
    scene_grid = np.zeros((10, 10))
    objects = {
        'car': (2, 3),
        'tree': (7, 8),
        'house': (5, 5)
    }
    
    scene_vecs = []
    for obj_name, (x, y) in objects.items():
        obj_vec = vsa.generate_vector()
        vsa.store(obj_name, obj_vec)
        pos_vec = spatial_encoder.encode([x, y])
        obj_at_pos = vsa.bind(obj_vec, pos_vec)
        scene_vecs.append(obj_at_pos)
    scene_vec = vsa.bundle(scene_vecs)
    
    print("   Scene contains:")
    for obj_name, pos in objects.items():
        print(f"   - {obj_name} at position {pos}")
    
    # Example 3: 3D spatial encoding
    print("\n3. 3D Spatial Encoding:")
    
    # Create 3D encoder
    spatial_3d = SpatialEncoder(vsa, grid_size=(8, 8, 8))
    
    # Encode 3D points
    points_3d = [(1, 2, 3), (4, 5, 6), (7, 1, 2)]
    
    for x, y, z in points_3d:
        point_3d_vec = spatial_3d.encode_3d(x, y, z)
        print(f"   Encoded 3D point ({x}, {y}, {z})")
    
    print()


def demonstrate_temporal_encoding():
    """Demonstrate temporal data encoding."""
    print("=== Temporal Data Encoding ===\n")
    
    vsa = create_vsa(
        dimension=10000,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )
    
    # Example 1: Time series encoding
    print("1. Time Series Encoding:")
    
    temporal_encoder = TemporalEncoder(vsa, max_lag=5)
    
    # Create synthetic time series
    time_steps = 20
    signal = np.sin(np.linspace(0, 4*np.pi, time_steps)) + 0.5 * np.random.randn(time_steps)
    
    # Encode time series
    series_vectors = []
    for t in range(time_steps):
        # Discretize value
        value_level = int((signal[t] + 2) * 5)  # Map to 0-20
        value_vec = vsa.generate_vector()
        vsa.store(f'val_{value_level}', value_vec)
        time_vec = temporal_encoder.encode_time_point(t)
        
        # Bind value with time
        point_vec = vsa.bind(value_vec, time_vec)
        series_vectors.append(point_vec)
    
    # Create full time series representation
    timeseries_vec = vsa.bundle(series_vectors)
    
    print(f"   Encoded time series with {time_steps} points")
    print(f"   Signal range: [{signal.min():.2f}, {signal.max():.2f}]")
    
    # Example 2: Event sequence encoding
    print("\n2. Event Sequence Encoding:")
    
    # Define events with timestamps
    events = [
        (0.0, 'login'),
        (0.5, 'browse'),
        (2.3, 'add_to_cart'),
        (3.1, 'checkout'),
        (3.5, 'payment')
    ]
    
    event_vecs = []
    for timestamp, event in events:
        # Encode event
        event_vec = vsa.generate_vector()
        vsa.store(event, event_vec)
        
        # Encode timestamp (discretized)
        time_slot = int(timestamp * 10)  # 0.1 second resolution
        time_vec = temporal_encoder.encode_time_point(time_slot)
        
        # Bind and add to sequence
        event_at_time = vsa.bind(event_vec, time_vec)
        event_vecs.append(event_at_time)
        
        print(f"   Event '{event}' at time {timestamp}s")
    
    # Bundle all events
    event_sequence = vsa.bundle(event_vecs)
    
    # Example 3: Periodic patterns
    print("\n3. Periodic Pattern Encoding:")
    
    # Encode day of week pattern
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_vectors = {}
    
    # Create cyclic encoding
    for i, day in enumerate(days):
        phase = 2 * np.pi * i / len(days)
        # Use phase to create smooth transitions
        day_vec = vsa.generate_vector()
        vsa.store(f'day_{day}', day_vec)
        # Add cyclic component using permutation with shift
        cyclic_vec = vsa.permute(day_vec, shift=int(phase * 100))
        day_vectors[day] = vsa.bundle([day_vec, cyclic_vec], weights=[0.7, 0.3])
    
    # Test cyclicity
    print("   Day similarities (showing weekly cycle):")
    mon = day_vectors['Mon']
    for day, vec in day_vectors.items():
        sim = vsa.similarity(mon, vec)
        print(f"   Mon vs {day}: {sim:.3f}")
    print()


def demonstrate_graph_encoding():
    """Demonstrate graph structure encoding."""
    print("=== Graph Structure Encoding ===\n")
    
    vsa = create_vsa(
        dimension=10000,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )
    
    # Example 1: Simple graph encoding
    print("1. Simple Graph Encoding:")
    
    graph_encoder = GraphEncoder(vsa)
    
    # Define a small graph
    nodes = ['A', 'B', 'C', 'D']
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'C')]
    
    # Encode nodes
    node_vectors = {}
    for node in nodes:
        node_vec = vsa.generate_vector()
        vsa.store(f'node_{node}', node_vec)
        node_vectors[node] = node_vec
    
    # Encode edges
    edge_vectors = []
    for source, target in edges:
        edge_vec = graph_encoder.encode_edge(source, target)
        edge_vectors.append(edge_vec)
        print(f"   Encoded edge: {source} → {target}")
    
    # Create graph representation
    graph_vec = vsa.bundle(edge_vectors)
    
    # Example 2: Attributed graph
    print("\n2. Attributed Graph Encoding:")
    
    # Add node attributes
    node_attributes = {
        'A': {'type': 'start', 'value': 10},
        'B': {'type': 'process', 'value': 20},
        'C': {'type': 'process', 'value': 30},
        'D': {'type': 'end', 'value': 40}
    }
    
    # Encode nodes with attributes
    attributed_nodes = {}
    for node, attrs in node_attributes.items():
        node_vec = node_vectors[node]
        
        # Add attributes
        type_vec = vsa.generate_vector()
        vsa.store(f"type_{attrs['type']}", type_vec)
        value_vec = vsa.generate_vector()
        vsa.store(f"value_{attrs['value']}", value_vec)
        
        # Create role vectors for attributes
        type_role = vsa.generate_vector()
        value_role = vsa.generate_vector()
        vsa.store('TYPE', type_role)
        vsa.store('VALUE', value_role)
        
        attributed_node = vsa.bundle([
            node_vec,
            vsa.bind(type_role, type_vec),
            vsa.bind(value_role, value_vec)
        ])
        
        attributed_nodes[node] = attributed_node
        print(f"   Node {node}: type={attrs['type']}, value={attrs['value']}")
    
    # Example 3: Path encoding
    print("\n3. Path Encoding in Graphs:")
    
    # Encode a path through the graph
    path = ['A', 'B', 'C', 'D']
    
    # Encode path as sequence of transitions
    path_vecs = []
    for i in range(len(path) - 1):
        # Bind current node with next node
        curr_vec = node_vectors[path[i]]
        next_vec = node_vectors[path[i+1]]
        # Use permutation to encode position
        transition = vsa.bind(curr_vec, next_vec)
        transition = vsa.permute(transition, shift=i)
        path_vecs.append(transition)
    
    # Also include individual nodes
    for i, node in enumerate(path):
        node_vec = vsa.permute(node_vectors[node], shift=i*10)  # Different shift to distinguish from transitions
        path_vecs.append(node_vec)
    
    path_vec = vsa.bundle(path_vecs)
    
    print(f"   Encoded path: {' → '.join(path)}")
    
    # Check if nodes are in path
    print("   Path membership check:")
    for node in ['A', 'C', 'E']:
        if node in node_vectors:
            sim = vsa.similarity(path_vec, node_vectors[node])
            print(f"   Node {node} in path: similarity = {sim:.3f}")
        else:
            print(f"   Node {node}: not in graph")
    print()


def demonstrate_image_encoding():
    """Demonstrate simplified image encoding."""
    print("=== Image Encoding (Simplified) ===\n")
    
    vsa = create_vsa(
        dimension=10000,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )
    
    # Example 1: Pixel-based encoding (small images)
    print("1. Pixel-Based Encoding (4x4 image):")
    
    # Create a simple 4x4 binary image
    image = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ])
    
    # Encode each pixel with its position
    spatial_encoder = SpatialEncoder(vsa, grid_size=(4, 4))
    image_vecs = []
    
    for i in range(4):
        for j in range(4):
            if image[i, j] == 1:
                pos_vec = spatial_encoder.encode_2d(i, j)
                pixel_vec = vsa.generate_vector()
                vsa.store('white', pixel_vec)
                pixel_at_pos = vsa.bind(pixel_vec, pos_vec)
                image_vecs.append(pixel_at_pos)
    
    # Bundle all pixel positions
    image_vec = vsa.bundle(image_vecs) if image_vecs else vsa.generate_vector()
    
    print("   Encoded 4x4 checkerboard pattern")
    print("   Image:")
    for row in image:
        print(f"   {' '.join(['█' if p else '·' for p in row])}")
    
    # Example 2: Feature-based encoding
    print("\n2. Feature-Based Image Encoding:")
    
    # Simulate image features (SIFT-like)
    features = [
        {'type': 'corner', 'x': 10, 'y': 20, 'scale': 1.5},
        {'type': 'edge', 'x': 30, 'y': 25, 'scale': 2.0},
        {'type': 'blob', 'x': 50, 'y': 50, 'scale': 3.0}
    ]
    
    feature_vecs = []
    for feat in features:
        # Encode feature type
        type_vec = vsa.generate_vector()
        vsa.store(f"feat_{feat['type']}", type_vec)
        
        # Encode position
        pos_vec = spatial_encoder.encode_2d(
            int(feat['x'] / 10),  # Discretize
            int(feat['y'] / 10)
        )
        
        # Encode scale
        scale_vec = vsa.generate_vector()
        vsa.store(f"scale_{int(feat['scale'] * 10)}", scale_vec)
        
        # Create role vectors
        pos_role = vsa.generate_vector()
        scale_role = vsa.generate_vector()
        vsa.store('POSITION', pos_role)
        vsa.store('SCALE', scale_role)
        
        # Combine
        feature = vsa.bundle([
            type_vec,
            vsa.bind(pos_role, pos_vec),
            vsa.bind(scale_role, scale_vec)
        ])
        
        feature_vecs.append(feature)
        print(f"   Feature: {feat['type']} at ({feat['x']}, {feat['y']}), scale={feat['scale']}")
    
    # Bundle all features
    feature_vec = vsa.bundle(feature_vecs)
    
    # Example 3: Histogram encoding
    print("\n3. Histogram-Based Encoding:")
    
    # Simulate color histogram
    histogram = {
        'red': 0.3,
        'green': 0.2,
        'blue': 0.4,
        'yellow': 0.1
    }
    
    hist_vecs = []
    for color, weight in histogram.items():
        color_vec = vsa.generate_vector()
        vsa.store(f"color_{color}", color_vec)
        # Weight by frequency
        weighted_color = color_vec * weight
        hist_vecs.append(weighted_color)
    
    # Bundle weighted colors
    hist_vec = vsa.bundle(hist_vecs)
    
    print("   Color histogram:")
    for color, weight in histogram.items():
        print(f"   - {color}: {weight:.1%}")
    print()


def demonstrate_record_encoding():
    """Demonstrate structured record encoding."""
    print("=== Structured Record Encoding ===\n")
    
    vsa = create_vsa(
        dimension=10000,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )
    
    # Example 1: Database record
    print("1. Database Record Encoding:")
    
    # Define a person record
    person = {
        'id': 12345,
        'name': 'John Doe',
        'age': 30,
        'city': 'New York',
        'occupation': 'Engineer'
    }
    
    # Encode each field
    record_vecs = []
    for field, value in person.items():
        field_vec = vsa.generate_vector()
        vsa.store(f'field_{field}', field_vec)
        
        if isinstance(value, int):
            value_vec = vsa.generate_vector()
            vsa.store(f'int_{value}', value_vec)
        else:
            value_vec = vsa.generate_vector()
            vsa.store(str(value), value_vec)
        
        field_value = vsa.bind(field_vec, value_vec)
        record_vecs.append(field_value)
    
    # Bundle all field-value pairs
    record_vec = vsa.bundle(record_vecs)
    
    print("   Encoded record:")
    for field, value in person.items():
        print(f"   - {field}: {value}")
    
    # Query record
    print("\n   Querying record:")
    for field in ['name', 'age', 'city']:
        # Get field vector from memory
        field_vec = vsa.memory.get(f'field_{field}')
        if field_vec is not None:
            value_vec = vsa.unbind(record_vec, field_vec)
            
            # Find best match (simplified)
            if field in person:
                if isinstance(person[field], int):
                    expected_key = f'int_{person[field]}'
                else:
                    expected_key = str(person[field])
                    
                expected = vsa.memory.get(expected_key)
                if expected is not None:
                    sim = vsa.similarity(value_vec, expected)
                    print(f"   {field}: similarity = {sim:.3f}")
    
    # Example 2: Hierarchical records
    print("\n2. Hierarchical Record Encoding:")
    
    # Company structure
    company = {
        'name': 'TechCorp',
        'departments': {
            'engineering': {
                'head': 'Alice',
                'employees': 50
            },
            'sales': {
                'head': 'Bob',
                'employees': 30
            }
        }
    }
    
    # Encode hierarchically
    company_vec = vsa.generate_vector()
    vsa.store(company['name'], company_vec)
    
    # Create role vectors
    dept_name_role = vsa.generate_vector()
    head_role = vsa.generate_vector()
    size_role = vsa.generate_vector()
    vsa.store('dept_name', dept_name_role)
    vsa.store('head', head_role)
    vsa.store('size', size_role)
    
    dept_vecs = []
    for dept_name, dept_info in company['departments'].items():
        # Create vectors for values
        dept_name_vec = vsa.generate_vector()
        head_vec = vsa.generate_vector()
        size_vec = vsa.generate_vector()
        vsa.store(dept_name, dept_name_vec)
        vsa.store(dept_info['head'], head_vec)
        vsa.store(f"size_{dept_info['employees']}", size_vec)
        
        dept = vsa.bundle([
            vsa.bind(dept_name_role, dept_name_vec),
            vsa.bind(head_role, head_vec),
            vsa.bind(size_role, size_vec)
        ])
        dept_vecs.append(dept)
    
    dept_vec = vsa.bundle(dept_vecs)
    
    # Create role vectors for company structure
    company_role = vsa.generate_vector()
    departments_role = vsa.generate_vector()
    vsa.store('COMPANY', company_role)
    vsa.store('DEPARTMENTS', departments_role)
    
    company_full = vsa.bundle([
        vsa.bind(company_role, company_vec),
        vsa.bind(departments_role, dept_vec)
    ])
    
    print("   Encoded company structure with nested departments")
    
    # Example 3: Multi-valued attributes
    print("\n3. Multi-Valued Attribute Encoding:")
    
    # Product with multiple tags
    product = {
        'name': 'Laptop',
        'tags': ['electronics', 'portable', 'computer', 'work'],
        'price': 999.99,
        'in_stock': True
    }
    
    # Create role vectors
    name_role = vsa.generate_vector()
    price_role = vsa.generate_vector()
    stock_role = vsa.generate_vector()
    vsa.store('name', name_role)
    vsa.store('price', price_role)
    vsa.store('stock', stock_role)
    
    # Create value vectors
    name_vec = vsa.generate_vector()
    price_vec = vsa.generate_vector()
    stock_vec = vsa.generate_vector()
    vsa.store(product['name'], name_vec)
    vsa.store(f"price_{int(product['price'])}", price_vec)
    vsa.store('in_stock' if product['in_stock'] else 'out_stock', stock_vec)
    
    # Encode single-valued attributes
    product_vecs = [
        vsa.bind(name_role, name_vec),
        vsa.bind(price_role, price_vec),
        vsa.bind(stock_role, stock_vec)
    ]
    
    # Encode multi-valued tags
    tag_vecs = []
    for tag in product['tags']:
        tag_vec = vsa.generate_vector()
        vsa.store(f'tag_{tag}', tag_vec)
        tag_vecs.append(tag_vec)
    
    tags_vec = vsa.bundle(tag_vecs)
    
    # Create tags role vector
    tags_role = vsa.generate_vector()
    vsa.store('tags', tags_role)
    
    product_vecs.append(vsa.bind(tags_role, tags_vec))
    product_vec = vsa.bundle(product_vecs)
    
    print(f"   Product: {product['name']}")
    print(f"   Tags: {', '.join(product['tags'])}")
    print(f"   Price: ${product['price']}")
    print(f"   In stock: {product['in_stock']}\n")


def demonstrate_mixed_encoding():
    """Demonstrate encoding mixed data types together."""
    print("=== Mixed Data Type Encoding ===\n")
    
    vsa = create_vsa(
        dimension=10000,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )
    
    # Create a complex sensor reading
    print("1. IoT Sensor Data Encoding:")
    
    sensor_data = {
        'device_id': 'sensor_42',
        'timestamp': 1234567890,
        'location': (37.7749, -122.4194),  # San Francisco
        'readings': {
            'temperature': 22.5,
            'humidity': 65.0,
            'pressure': 1013.25
        },
        'status': 'active',
        'alerts': ['low_battery', 'calibration_due']
    }
    
    # Encode components
    spatial_encoder = SpatialEncoder(vsa, grid_size=(180, 360))
    temporal_encoder = TemporalEncoder(vsa)
    level_encoder = LevelEncoder(vsa, num_levels=100, min_value=0, max_value=100)
    
    # Build encoding
    sensor_vecs = []
    
    # Device ID
    device_role = vsa.generate_vector()
    device_vec = vsa.generate_vector()
    vsa.store('device', device_role)
    vsa.store(sensor_data['device_id'], device_vec)
    sensor_vecs.append(vsa.bind(device_role, device_vec))
    
    # Timestamp
    time_vec = temporal_encoder.encode_time_point(sensor_data['timestamp'] % 86400)  # Time of day
    time_role = vsa.generate_vector()
    vsa.store('time', time_role)
    sensor_vecs.append(vsa.bind(time_role, time_vec))
    
    # Location
    lat, lon = sensor_data['location']
    loc_vec = spatial_encoder.encode_2d(
        int((lat + 90) * 2),  # Discretize latitude
        int((lon + 180) * 2)  # Discretize longitude
    )
    location_role = vsa.generate_vector()
    vsa.store('location', location_role)
    sensor_vecs.append(vsa.bind(location_role, loc_vec))
    
    # Readings
    for reading_type, value in sensor_data['readings'].items():
        if reading_type == 'temperature':
            # Use level encoding for temperature
            reading_vec = level_encoder.encode(value)
        else:
            # Simple discretization for others
            reading_vec = vsa.generate_vector()
            vsa.store(f"{reading_type}_{int(value)}", reading_vec)
        
        reading_role = vsa.generate_vector()
        vsa.store(reading_type, reading_role)
        sensor_vecs.append(vsa.bind(reading_role, reading_vec))
    
    # Status and alerts
    status_role = vsa.generate_vector()
    status_vec = vsa.generate_vector()
    vsa.store('status', status_role)
    vsa.store(sensor_data['status'], status_vec)
    sensor_vecs.append(vsa.bind(status_role, status_vec))
    
    # Bundle alerts
    alert_vecs = []
    for alert in sensor_data['alerts']:
        alert_vec = vsa.generate_vector()
        vsa.store(alert, alert_vec)
        alert_vecs.append(alert_vec)
    alerts_vec = vsa.bundle(alert_vecs)
    alerts_role = vsa.generate_vector()
    vsa.store('alerts', alerts_role)
    sensor_vecs.append(vsa.bind(alerts_role, alerts_vec))
    
    # Bundle all sensor data
    sensor_vec = vsa.bundle(sensor_vecs)
    
    print("   Encoded complex sensor reading with:")
    print("   - Device ID and timestamp")
    print("   - GPS location")
    print("   - Multiple sensor readings")
    print("   - Status and alerts")
    
    # Demonstrate querying
    print("\n2. Querying Mixed Data:")
    
    # Query temperature
    temp_role = vsa.memory.get('temperature')
    if temp_role is not None:
        temp_query = vsa.unbind(sensor_vec, temp_role)
        decoded_temp = level_encoder.decode(temp_query)
        print(f"   Temperature reading: ~{decoded_temp:.1f}°C")
    
    # Query status
    status_role = vsa.memory.get('status')
    active_vec = vsa.memory.get('active')
    if status_role is not None and active_vec is not None:
        status_query = vsa.unbind(sensor_vec, status_role)
        status_sim = vsa.similarity(status_query, active_vec)
        print(f"   Status 'active' similarity: {status_sim:.3f}")
    
    # Check for alerts
    alerts_role = vsa.memory.get('alerts')
    battery_vec = vsa.memory.get('low_battery')
    if alerts_role is not None and battery_vec is not None:
        alerts_query = vsa.unbind(sensor_vec, alerts_role)
        battery_sim = vsa.similarity(alerts_query, battery_vec)
        print(f"   'low_battery' alert present: {battery_sim > 0.3} (sim: {battery_sim:.3f})\n")


def main():
    """Run all data encoding demonstrations."""
    print("\n" + "="*60)
    print("VSA DATA ENCODING DEMONSTRATION")
    print("="*60 + "\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstrations
    demonstrate_text_encoding()
    print("-"*60 + "\n")
    
    demonstrate_numerical_encoding()
    print("-"*60 + "\n")
    
    demonstrate_spatial_encoding()
    print("-"*60 + "\n")
    
    demonstrate_temporal_encoding()
    print("-"*60 + "\n")
    
    demonstrate_graph_encoding()
    print("-"*60 + "\n")
    
    demonstrate_image_encoding()
    print("-"*60 + "\n")
    
    demonstrate_record_encoding()
    print("-"*60 + "\n")
    
    demonstrate_mixed_encoding()
    
    # Summary
    print("=== Summary ===\n")
    print("VSA can encode diverse data types:")
    print("• Text: character/word-level, n-grams, sequences")
    print("• Numbers: discrete values, continuous levels, complex numbers")
    print("• Spatial: 2D/3D coordinates, scenes, relationships")
    print("• Temporal: time series, events, periodic patterns")
    print("• Graphs: nodes, edges, paths, attributes")
    print("• Images: pixels, features, histograms")
    print("• Records: flat, hierarchical, multi-valued")
    print("• Mixed: combining multiple data types\n")
    
    print("Key encoding strategies:")
    print("• Role-filler binding for structured data")
    print("• Position encoding for sequences")
    print("• Level encoding for continuous values")
    print("• Bundling for sets and collections")
    print("• Permutation for order preservation\n")
    
    print("="*60)
    print("Data Encoding Demo Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()