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
    RandomIndexingEncoder, SequenceEncoder, SpatialEncoder,
    TemporalEncoder, LevelEncoder, GraphEncoder
)


def demonstrate_text_encoding():
    """Demonstrate text and sequence encoding."""
    print("=== Text and Sequence Encoding ===\n")
    
    # Create VSA instance
    vsa = create_vsa(VSAConfig(
        dimension=10000,
        vector_type='bipolar',
        binding_method='multiplication'
    ))
    
    # Example 1: Character-level encoding
    print("1. Character-Level Encoding:")
    
    # Create character vectors
    chars = {}
    for c in 'abcdefghijklmnopqrstuvwxyz ':
        chars[c] = vsa.encode(f'char_{c}')
    
    # Encode word "hello"
    word = "hello"
    char_vectors = []
    for i, c in enumerate(word):
        # Bind character with position
        pos_vec = vsa.encode(f'pos_{i}')
        char_vec = chars[c]
        char_vectors.append(vsa.bind(pos_vec, char_vec))
    
    word_vec = vsa.bundle(char_vectors)
    
    # Decode characters
    print(f"   Encoded word: '{word}'")
    print("   Decoded characters:")
    for i in range(len(word)):
        pos_vec = vsa.encode(f'pos_{i}')
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
    ri_encoder = RandomIndexingEncoder(vsa, n_gram_size=3)
    
    # Encode sentence
    sentence = "the quick brown fox jumps"
    words = sentence.split()
    
    # Create word vectors
    word_vecs = {}
    for word in set(words):
        word_vecs[word] = ri_encoder.encode(word)
    
    # Create sentence representation
    sentence_vec = vsa.zero()
    for i, word in enumerate(words):
        # Position encoding
        pos_vec = vsa.permute(vsa.identity(), i)
        word_positioned = vsa.bind(word_vecs[word], pos_vec)
        sentence_vec = vsa.bundle([sentence_vec, word_positioned])
    
    print(f"   Encoded sentence: '{sentence}'")
    print(f"   Sentence vector norm: {np.linalg.norm(sentence_vec.data):.3f}")
    
    # Example 3: Sequence encoding with different methods
    print("\n3. Sequence Encoding Methods:")
    
    seq_encoder = SequenceEncoder(vsa)
    sequence = ['first', 'second', 'third', 'fourth']
    
    # Positional encoding
    pos_encoded = seq_encoder.encode_sequence(
        [vsa.encode(item) for item in sequence],
        method='positional'
    )
    
    # Chaining encoding
    chain_encoded = seq_encoder.encode_sequence(
        [vsa.encode(item) for item in sequence],
        method='chaining'
    )
    
    print("   Positional encoding: preserves absolute positions")
    print("   Chaining encoding: preserves sequential relationships\n")


def demonstrate_numerical_encoding():
    """Demonstrate encoding of numerical values."""
    print("=== Numerical Value Encoding ===\n")
    
    vsa = create_vsa(VSAConfig(
        dimension=10000,
        vector_type='bipolar',
        binding_method='multiplication'
    ))
    
    # Example 1: Discrete value encoding
    print("1. Discrete Value Encoding:")
    
    # Encode integers 0-9
    int_vectors = {}
    for i in range(10):
        int_vectors[i] = vsa.encode(f'int_{i}')
    
    # Create composite number (e.g., 42)
    tens = vsa.bind(vsa.encode('tens'), int_vectors[4])
    ones = vsa.bind(vsa.encode('ones'), int_vectors[2])
    number_42 = vsa.bundle([tens, ones])
    
    # Decode
    decoded_tens = vsa.unbind(number_42, vsa.encode('tens'))
    decoded_ones = vsa.unbind(number_42, vsa.encode('ones'))
    
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
    
    level_encoder = LevelEncoder(vsa, num_levels=10, min_value=0.0, max_value=100.0)
    
    # Encode different temperatures
    temperatures = [15.5, 25.0, 35.8, 72.3]
    
    print("   Temperature encoding:")
    for temp in temperatures:
        temp_vec = level_encoder.encode(temp)
        # Decode by checking similarity to level vectors
        decoded_level = level_encoder.decode(temp_vec)
        print(f"   {temp}°C → level {decoded_level} → ~{level_encoder.level_to_value(decoded_level):.1f}°C")
    
    # Example 3: Magnitude and phase encoding
    print("\n3. Magnitude-Phase Encoding:")
    
    # For complex numbers or 2D vectors
    magnitude = 5.0
    angle = np.pi / 4  # 45 degrees
    
    # Encode magnitude (discretized)
    mag_levels = np.linspace(0, 10, 20)
    mag_idx = np.argmin(np.abs(mag_levels - magnitude))
    mag_vec = vsa.encode(f'mag_{mag_idx}')
    
    # Encode angle (discretized)
    angle_levels = np.linspace(0, 2*np.pi, 36)  # 10-degree steps
    angle_idx = np.argmin(np.abs(angle_levels - angle))
    angle_vec = vsa.encode(f'angle_{angle_idx}')
    
    # Combine
    complex_vec = vsa.bundle([
        vsa.bind(vsa.encode('MAG'), mag_vec),
        vsa.bind(vsa.encode('PHASE'), angle_vec)
    ])
    
    print(f"   Encoded complex number: {magnitude} * e^(i*{angle:.2f})")
    print(f"   Magnitude level: {mag_idx}, Angle level: {angle_idx}\n")


def demonstrate_spatial_encoding():
    """Demonstrate spatial data encoding."""
    print("=== Spatial Data Encoding ===\n")
    
    vsa = create_vsa(VSAConfig(
        dimension=10000,
        vector_type='bipolar',
        binding_method='multiplication'
    ))
    
    # Example 1: 2D coordinate encoding
    print("1. 2D Coordinate Encoding:")
    
    spatial_encoder = SpatialEncoder(vsa, grid_size=(10, 10))
    
    # Encode points
    points = [(2, 3), (5, 7), (8, 1)]
    point_vectors = []
    
    for x, y in points:
        point_vec = spatial_encoder.encode_2d(x, y)
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
    
    scene_vec = vsa.zero()
    for obj_name, (x, y) in objects.items():
        obj_vec = vsa.encode(obj_name)
        pos_vec = spatial_encoder.encode_2d(x, y)
        obj_at_pos = vsa.bind(obj_vec, pos_vec)
        scene_vec = vsa.bundle([scene_vec, obj_at_pos])
    
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
    
    vsa = create_vsa(VSAConfig(
        dimension=10000,
        vector_type='bipolar',
        binding_method='multiplication'
    ))
    
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
        value_vec = vsa.encode(f'val_{value_level}')
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
    
    event_sequence = vsa.zero()
    for timestamp, event in events:
        # Encode event
        event_vec = vsa.encode(event)
        
        # Encode timestamp (discretized)
        time_slot = int(timestamp * 10)  # 0.1 second resolution
        time_vec = temporal_encoder.encode_time_point(time_slot)
        
        # Bind and add to sequence
        event_at_time = vsa.bind(event_vec, time_vec)
        event_sequence = vsa.bundle([event_sequence, event_at_time])
        
        print(f"   Event '{event}' at time {timestamp}s")
    
    # Example 3: Periodic patterns
    print("\n3. Periodic Pattern Encoding:")
    
    # Encode day of week pattern
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_vectors = {}
    
    # Create cyclic encoding
    for i, day in enumerate(days):
        phase = 2 * np.pi * i / len(days)
        # Use phase to create smooth transitions
        day_vec = vsa.encode(f'day_{day}')
        # Add cyclic component
        cyclic_vec = vsa.permute(vsa.identity(), int(phase * 100))
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
    
    vsa = create_vsa(VSAConfig(
        dimension=10000,
        vector_type='bipolar',
        binding_method='multiplication'
    ))
    
    # Example 1: Simple graph encoding
    print("1. Simple Graph Encoding:")
    
    graph_encoder = GraphEncoder(vsa)
    
    # Define a small graph
    nodes = ['A', 'B', 'C', 'D']
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'C')]
    
    # Encode nodes
    node_vectors = {node: vsa.encode(f'node_{node}') for node in nodes}
    
    # Encode edges
    edge_vectors = []
    for source, target in edges:
        edge_vec = graph_encoder.encode_edge(
            node_vectors[source],
            node_vectors[target]
        )
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
        type_vec = vsa.encode(f"type_{attrs['type']}")
        value_vec = vsa.encode(f"value_{attrs['value']}")
        
        attributed_node = vsa.bundle([
            node_vec,
            vsa.bind(vsa.encode('TYPE'), type_vec),
            vsa.bind(vsa.encode('VALUE'), value_vec)
        ])
        
        attributed_nodes[node] = attributed_node
        print(f"   Node {node}: type={attrs['type']}, value={attrs['value']}")
    
    # Example 3: Path encoding
    print("\n3. Path Encoding in Graphs:")
    
    # Encode a path through the graph
    path = ['A', 'B', 'C', 'D']
    path_vec = graph_encoder.encode_path([node_vectors[n] for n in path])
    
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
    
    vsa = create_vsa(VSAConfig(
        dimension=10000,
        vector_type='bipolar',
        binding_method='multiplication'
    ))
    
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
    image_vec = vsa.zero()
    
    for i in range(4):
        for j in range(4):
            if image[i, j] == 1:
                pos_vec = spatial_encoder.encode_2d(i, j)
                pixel_vec = vsa.encode('white')
                pixel_at_pos = vsa.bind(pixel_vec, pos_vec)
                image_vec = vsa.bundle([image_vec, pixel_at_pos])
    
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
    
    feature_vec = vsa.zero()
    for feat in features:
        # Encode feature type
        type_vec = vsa.encode(f"feat_{feat['type']}")
        
        # Encode position
        pos_vec = spatial_encoder.encode_2d(
            int(feat['x'] / 10),  # Discretize
            int(feat['y'] / 10)
        )
        
        # Encode scale
        scale_vec = vsa.encode(f"scale_{int(feat['scale'] * 10)}")
        
        # Combine
        feature = vsa.bundle([
            type_vec,
            vsa.bind(vsa.encode('POSITION'), pos_vec),
            vsa.bind(vsa.encode('SCALE'), scale_vec)
        ])
        
        feature_vec = vsa.bundle([feature_vec, feature])
        print(f"   Feature: {feat['type']} at ({feat['x']}, {feat['y']}), scale={feat['scale']}")
    
    # Example 3: Histogram encoding
    print("\n3. Histogram-Based Encoding:")
    
    # Simulate color histogram
    histogram = {
        'red': 0.3,
        'green': 0.2,
        'blue': 0.4,
        'yellow': 0.1
    }
    
    hist_vec = vsa.zero()
    for color, weight in histogram.items():
        color_vec = vsa.encode(f"color_{color}")
        # Weight by frequency
        weighted_color = color_vec.data * weight
        hist_vec = vsa.bundle([hist_vec, vsa.encode_raw(weighted_color)])
    
    print("   Color histogram:")
    for color, weight in histogram.items():
        print(f"   - {color}: {weight:.1%}")
    print()


def demonstrate_record_encoding():
    """Demonstrate structured record encoding."""
    print("=== Structured Record Encoding ===\n")
    
    vsa = create_vsa(VSAConfig(
        dimension=10000,
        vector_type='bipolar',
        binding_method='multiplication'
    ))
    
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
    record_vec = vsa.zero()
    for field, value in person.items():
        field_vec = vsa.encode(f'field_{field}')
        
        if isinstance(value, int):
            value_vec = vsa.encode(f'int_{value}')
        else:
            value_vec = vsa.encode(str(value))
        
        field_value = vsa.bind(field_vec, value_vec)
        record_vec = vsa.bundle([record_vec, field_value])
    
    print("   Encoded record:")
    for field, value in person.items():
        print(f"   - {field}: {value}")
    
    # Query record
    print("\n   Querying record:")
    for field in ['name', 'age', 'city']:
        field_vec = vsa.encode(f'field_{field}')
        value_vec = vsa.unbind(record_vec, field_vec)
        
        # Find best match (simplified)
        if field in person:
            expected = vsa.encode(str(person[field]) if not isinstance(person[field], int) 
                                else f'int_{person[field]}')
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
    company_vec = vsa.encode(company['name'])
    
    dept_vec = vsa.zero()
    for dept_name, dept_info in company['departments'].items():
        dept = vsa.bundle([
            vsa.bind(vsa.encode('dept_name'), vsa.encode(dept_name)),
            vsa.bind(vsa.encode('head'), vsa.encode(dept_info['head'])),
            vsa.bind(vsa.encode('size'), vsa.encode(f"size_{dept_info['employees']}"))
        ])
        dept_vec = vsa.bundle([dept_vec, dept])
    
    company_full = vsa.bundle([
        vsa.bind(vsa.encode('COMPANY'), company_vec),
        vsa.bind(vsa.encode('DEPARTMENTS'), dept_vec)
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
    
    product_vec = vsa.zero()
    
    # Encode single-valued attributes
    product_vec = vsa.bundle([
        product_vec,
        vsa.bind(vsa.encode('name'), vsa.encode(product['name'])),
        vsa.bind(vsa.encode('price'), vsa.encode(f"price_{int(product['price'])}")),
        vsa.bind(vsa.encode('stock'), vsa.encode('in_stock' if product['in_stock'] else 'out_stock'))
    ])
    
    # Encode multi-valued tags
    tags_vec = vsa.zero()
    for tag in product['tags']:
        tags_vec = vsa.bundle([tags_vec, vsa.encode(f'tag_{tag}')])
    
    product_vec = vsa.bundle([
        product_vec,
        vsa.bind(vsa.encode('tags'), tags_vec)
    ])
    
    print(f"   Product: {product['name']}")
    print(f"   Tags: {', '.join(product['tags'])}")
    print(f"   Price: ${product['price']}")
    print(f"   In stock: {product['in_stock']}\n")


def demonstrate_mixed_encoding():
    """Demonstrate encoding mixed data types together."""
    print("=== Mixed Data Type Encoding ===\n")
    
    vsa = create_vsa(VSAConfig(
        dimension=10000,
        vector_type='bipolar',
        binding_method='multiplication'
    ))
    
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
    sensor_vec = vsa.zero()
    
    # Device ID
    sensor_vec = vsa.bundle([
        sensor_vec,
        vsa.bind(vsa.encode('device'), vsa.encode(sensor_data['device_id']))
    ])
    
    # Timestamp
    time_vec = temporal_encoder.encode_time_point(sensor_data['timestamp'] % 86400)  # Time of day
    sensor_vec = vsa.bundle([
        sensor_vec,
        vsa.bind(vsa.encode('time'), time_vec)
    ])
    
    # Location
    lat, lon = sensor_data['location']
    loc_vec = spatial_encoder.encode_2d(
        int((lat + 90) * 2),  # Discretize latitude
        int((lon + 180) * 2)  # Discretize longitude
    )
    sensor_vec = vsa.bundle([
        sensor_vec,
        vsa.bind(vsa.encode('location'), loc_vec)
    ])
    
    # Readings
    for reading_type, value in sensor_data['readings'].items():
        if reading_type == 'temperature':
            # Use level encoding for temperature
            temp_vec = level_encoder.encode(value)
        else:
            # Simple discretization for others
            temp_vec = vsa.encode(f"{reading_type}_{int(value)}")
        
        sensor_vec = vsa.bundle([
            sensor_vec,
            vsa.bind(vsa.encode(reading_type), temp_vec)
        ])
    
    # Status and alerts
    sensor_vec = vsa.bundle([
        sensor_vec,
        vsa.bind(vsa.encode('status'), vsa.encode(sensor_data['status']))
    ])
    
    # Bundle alerts
    alerts_vec = vsa.bundle([vsa.encode(alert) for alert in sensor_data['alerts']])
    sensor_vec = vsa.bundle([
        sensor_vec,
        vsa.bind(vsa.encode('alerts'), alerts_vec)
    ])
    
    print("   Encoded complex sensor reading with:")
    print("   - Device ID and timestamp")
    print("   - GPS location")
    print("   - Multiple sensor readings")
    print("   - Status and alerts")
    
    # Demonstrate querying
    print("\n2. Querying Mixed Data:")
    
    # Query temperature
    temp_query = vsa.unbind(sensor_vec, vsa.encode('temperature'))
    decoded_temp = level_encoder.decode(temp_query)
    print(f"   Temperature reading: ~{level_encoder.level_to_value(decoded_temp):.1f}°C")
    
    # Query status
    status_query = vsa.unbind(sensor_vec, vsa.encode('status'))
    status_sim = vsa.similarity(status_query, vsa.encode('active'))
    print(f"   Status 'active' similarity: {status_sim:.3f}")
    
    # Check for alerts
    alerts_query = vsa.unbind(sensor_vec, vsa.encode('alerts'))
    battery_sim = vsa.similarity(alerts_query, vsa.encode('low_battery'))
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