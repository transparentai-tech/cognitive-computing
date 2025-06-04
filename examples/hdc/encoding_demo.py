#!/usr/bin/env python3
"""
HDC Encoding Demo

This example demonstrates various encoding strategies:
- Scalar encoding (thermometer and level)
- Categorical encoding
- Sequence encoding (n-gram and positional)
- Spatial encoding
- Record/structured data encoding
- Text encoding with n-grams
"""

import numpy as np
from cognitive_computing.hdc import (
    ScalarEncoder,
    CategoricalEncoder,
    SequenceEncoder,
    SpatialEncoder,
    RecordEncoder,
    NGramEncoder,
    HDC,
    HDCConfig,
    plot_hypervector_comparison,
    plot_similarity_matrix,
)
import matplotlib.pyplot as plt


def scalar_encoding_demo():
    """Demonstrate scalar value encoding."""
    print("=== Scalar Encoding Demo ===\n")
    
    # Thermometer encoding
    thermo_encoder = ScalarEncoder(
        dimension=1000,
        min_value=0.0,
        max_value=100.0,
        n_levels=10,
        method="thermometer"
    )
    
    # Level encoding
    level_encoder = ScalarEncoder(
        dimension=1000,
        min_value=0.0,
        max_value=100.0,
        n_levels=10,
        method="level"
    )
    
    # Encode different values
    values = [0, 25, 50, 75, 100]
    thermo_vectors = {}
    level_vectors = {}
    
    for val in values:
        thermo_vectors[f"Thermo_{val}"] = thermo_encoder.encode(val)
        level_vectors[f"Level_{val}"] = level_encoder.encode(val)
    
    # Compare encodings
    print("Thermometer encoding preserves order:")
    for i in range(len(values) - 1):
        v1 = thermo_vectors[f"Thermo_{values[i]}"]
        v2 = thermo_vectors[f"Thermo_{values[i+1]}"]
        v_mid = thermo_vectors[f"Thermo_{values[i//2]}"]
        
        sim12 = np.dot(v1, v2) / len(v1)
        sim_mid = np.dot(v1, v_mid) / len(v1)
        
        print(f"  Similarity({values[i]}, {values[i+1]}) = {sim12:.3f}")
    
    print("\nLevel encoding creates orthogonal codes:")
    for i in range(len(values) - 1):
        v1 = level_vectors[f"Level_{values[i]}"]
        v2 = level_vectors[f"Level_{values[i+1]}"]
        
        sim = np.dot(v1, v2) / len(v1)
        print(f"  Similarity({values[i]}, {values[i+1]}) = {sim:.3f}")
    
    # Visualize
    fig, axes = plot_hypervector_comparison(
        {"Thermometer_0": thermo_vectors["Thermo_0"],
         "Thermometer_50": thermo_vectors["Thermo_50"],
         "Level_0": level_vectors["Level_0"],
         "Level_50": level_vectors["Level_50"]},
        segment_size=100
    )
    plt.suptitle("Scalar Encoding Comparison")
    
    return thermo_vectors, level_vectors


def categorical_encoding_demo():
    """Demonstrate categorical encoding."""
    print("\n=== Categorical Encoding Demo ===\n")
    
    # Create encoder with known categories
    colors = ["red", "green", "blue", "yellow"]
    encoder = CategoricalEncoder(
        dimension=1000,
        categories=colors
    )
    
    # Encode colors
    color_vectors = {}
    for color in colors:
        color_vectors[color] = encoder.encode(color)
    
    # Add new category dynamically
    color_vectors["purple"] = encoder.encode("purple")
    
    print(f"Known categories: {encoder.get_categories()}")
    
    # Check orthogonality
    print("\nColor similarity matrix:")
    all_colors = list(color_vectors.keys())
    for i, c1 in enumerate(all_colors):
        sims = []
        for c2 in all_colors:
            sim = np.dot(color_vectors[c1], color_vectors[c2]) / 1000
            sims.append(f"{sim:+.2f}")
        print(f"  {c1:8s}: {' '.join(sims)}")
    
    return color_vectors


def sequence_encoding_demo():
    """Demonstrate sequence encoding."""
    print("\n=== Sequence Encoding Demo ===\n")
    
    # Create HDC system for binding operations
    hdc = HDC(HDCConfig(dimension=1000))
    
    # N-gram encoding
    ngram_encoder = SequenceEncoder(
        dimension=1000,
        method="ngram",
        n=3
    )
    
    # Position encoding
    pos_encoder = SequenceEncoder(
        dimension=1000,
        method="position"
    )
    
    # Encode sequences
    seq1 = ["A", "B", "C", "D", "E"]
    seq2 = ["A", "B", "C", "E", "D"]  # Transposition
    seq3 = ["E", "D", "C", "B", "A"]  # Reverse
    
    print("Encoding sequences:")
    print(f"  Seq1: {seq1}")
    print(f"  Seq2: {seq2} (transposition)")
    print(f"  Seq3: {seq3} (reverse)")
    
    # Encode with both methods
    ngram_vecs = {
        "Seq1_ngram": ngram_encoder.encode(seq1),
        "Seq2_ngram": ngram_encoder.encode(seq2),
        "Seq3_ngram": ngram_encoder.encode(seq3),
    }
    
    pos_vecs = {
        "Seq1_pos": pos_encoder.encode(seq1),
        "Seq2_pos": pos_encoder.encode(seq2),
        "Seq3_pos": pos_encoder.encode(seq3),
    }
    
    # Compare similarities
    print("\nN-gram encoding similarities:")
    vecs = list(ngram_vecs.values())
    labels = list(ngram_vecs.keys())
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            sim = hdc.similarity(vecs[i], vecs[j])
            print(f"  {labels[i]} vs {labels[j]}: {sim:.3f}")
    
    print("\nPosition encoding similarities:")
    vecs = list(pos_vecs.values())
    labels = list(pos_vecs.keys())
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            sim = hdc.similarity(vecs[i], vecs[j])
            print(f"  {labels[i]} vs {labels[j]}: {sim:.3f}")
    
    return ngram_vecs, pos_vecs


def spatial_encoding_demo():
    """Demonstrate spatial coordinate encoding."""
    print("\n=== Spatial Encoding Demo ===\n")
    
    # 2D spatial encoder
    encoder_2d = SpatialEncoder(
        dimension=1000,
        bounds=((0, 10), (0, 10)),
        resolution=10
    )
    
    # Encode 2D points
    points_2d = {
        "Origin": (0, 0),
        "Center": (5, 5),
        "TopRight": (10, 10),
        "Near_Center": (4, 5),
    }
    
    vectors_2d = {}
    for name, point in points_2d.items():
        vectors_2d[name] = encoder_2d.encode(point)
        print(f"Encoded {name}: {point}")
    
    # Check distance preservation
    print("\nSpatial similarities (2D):")
    hdc = HDC(HDCConfig(dimension=1000))
    
    for name1, point1 in points_2d.items():
        for name2, point2 in points_2d.items():
            if name1 < name2:  # Avoid duplicates
                # Euclidean distance
                dist = np.sqrt(sum((p1-p2)**2 for p1, p2 in zip(point1, point2)))
                # Hypervector similarity
                sim = hdc.similarity(vectors_2d[name1], vectors_2d[name2])
                print(f"  {name1} - {name2}: distance={dist:.2f}, similarity={sim:.3f}")
    
    return vectors_2d


def record_encoding_demo():
    """Demonstrate structured record encoding."""
    print("\n=== Record Encoding Demo ===\n")
    
    # Create field-specific encoders
    age_encoder = ScalarEncoder(1000, 0, 100, 20, "thermometer")
    category_encoder = CategoricalEncoder(1000, ["student", "professor", "staff"])
    
    # Create record encoder
    encoder = RecordEncoder(
        dimension=1000,
        field_encoders={
            "age": age_encoder,
            "role": category_encoder
        }
    )
    
    # Encode records
    records = [
        {"name": "Alice", "age": 22, "role": "student", "department": "CS"},
        {"name": "Bob", "age": 45, "role": "professor", "department": "CS"},
        {"name": "Carol", "age": 30, "role": "staff", "department": "Math"},
        {"name": "Dave", "age": 23, "role": "student", "department": "CS"},
    ]
    
    record_vectors = {}
    for i, record in enumerate(records):
        vec = encoder.encode(record)
        record_vectors[f"Record_{i+1}"] = vec
        print(f"Encoded record {i+1}: {record}")
    
    # Find similar records
    print("\nRecord similarities:")
    hdc = HDC(HDCConfig(dimension=1000))
    
    vec_list = list(record_vectors.values())
    for i in range(len(vec_list)):
        for j in range(i+1, len(vec_list)):
            sim = hdc.similarity(vec_list[i], vec_list[j])
            if sim > 0.3:  # Show only significant similarities
                print(f"  Record {i+1} - Record {j+1}: {sim:.3f}")
                # Explain why they're similar
                r1, r2 = records[i], records[j]
                common = []
                if r1.get("role") == r2.get("role"):
                    common.append("same role")
                if r1.get("department") == r2.get("department"):
                    common.append("same dept")
                if abs(r1.get("age", 0) - r2.get("age", 0)) < 5:
                    common.append("similar age")
                if common:
                    print(f"    ({', '.join(common)})")
    
    return record_vectors


def text_encoding_demo():
    """Demonstrate text encoding with n-grams."""
    print("\n=== Text Encoding Demo ===\n")
    
    # Character-level encoder
    char_encoder = NGramEncoder(
        dimension=1000,
        n=3,
        level="char"
    )
    
    # Word-level encoder
    word_encoder = NGramEncoder(
        dimension=1000,
        n=2,
        level="word"
    )
    
    # Sample texts
    texts = {
        "greeting1": "hello world",
        "greeting2": "hello there",
        "greeting3": "hi world",
        "different": "goodbye world",
    }
    
    # Encode with both methods
    char_vectors = {}
    word_vectors = {}
    
    for name, text in texts.items():
        char_vectors[f"{name}_char"] = char_encoder.encode(text)
        word_vectors[f"{name}_word"] = word_encoder.encode(text)
        print(f"Encoded '{text}'")
    
    # Compare similarities
    hdc = HDC(HDCConfig(dimension=1000))
    
    print("\nCharacter n-gram similarities:")
    for n1, t1 in texts.items():
        for n2, t2 in texts.items():
            if n1 < n2:
                sim = hdc.similarity(
                    char_vectors[f"{n1}_char"],
                    char_vectors[f"{n2}_char"]
                )
                print(f"  '{t1}' vs '{t2}': {sim:.3f}")
    
    print("\nWord n-gram similarities:")
    for n1, t1 in texts.items():
        for n2, t2 in texts.items():
            if n1 < n2:
                sim = hdc.similarity(
                    word_vectors[f"{n1}_word"],
                    word_vectors[f"{n2}_word"]
                )
                print(f"  '{t1}' vs '{t2}': {sim:.3f}")
    
    return char_vectors, word_vectors


def main():
    """Run all encoding demonstrations."""
    # Scalar encoding
    thermo_vecs, level_vecs = scalar_encoding_demo()
    
    # Categorical encoding
    color_vecs = categorical_encoding_demo()
    
    # Sequence encoding
    ngram_vecs, pos_vecs = sequence_encoding_demo()
    
    # Spatial encoding
    spatial_vecs = spatial_encoding_demo()
    
    # Record encoding
    record_vecs = record_encoding_demo()
    
    # Text encoding
    char_vecs, word_vecs = text_encoding_demo()
    
    # Create combined similarity matrix
    print("\n=== Combined Similarity Analysis ===")
    
    # Select representative vectors
    selected_vectors = [
        thermo_vecs["Thermo_0"],
        thermo_vecs["Thermo_50"],
        level_vecs["Level_0"],
        level_vecs["Level_50"],
        color_vecs["red"],
        color_vecs["blue"],
        list(ngram_vecs.values())[0],
        list(spatial_vecs.values())[0],
        list(record_vecs.values())[0],
        list(char_vecs.values())[0],
    ]
    
    selected_labels = [
        "Thermo_0", "Thermo_50", "Level_0", "Level_50",
        "Color_red", "Color_blue", "Sequence", "Spatial",
        "Record", "Text"
    ]
    
    fig, ax = plot_similarity_matrix(
        selected_vectors,
        selected_labels,
        annotate=True
    )
    plt.title("Cross-Encoding Similarity Matrix")
    
    plt.show()
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()