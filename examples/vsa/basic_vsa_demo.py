#!/usr/bin/env python3
"""
Basic VSA Demo: Overview of Vector Symbolic Architecture Features

This script demonstrates the core features of VSA including:
- Different vector types (Binary, Bipolar, Ternary, Complex, Integer)
- Various binding operations (XOR, Multiplication, Convolution, MAP, Permutation)
- Common VSA operations (bundling, permutation, thinning)
- Basic symbolic reasoning capabilities
"""

import numpy as np
from cognitive_computing.vsa import (
    create_vsa, VSAConfig,
    BSC, MAP, FHRR, SparseVSA,
    BinaryVector, BipolarVector, TernaryVector, ComplexVector, IntegerVector,
    generate_random_vector
)


def demonstrate_vector_types():
    """Demonstrate different VSA vector types and their properties."""
    print("=== VSA Vector Types Demo ===\n")
    
    dimension = 1000
    
    # Binary vectors
    print("1. Binary Vectors (0/1):")
    binary_vec = generate_random_vector(dimension, BinaryVector)
    print(f"   Sample values: {binary_vec.data[:10]}")
    print(f"   Density: {np.mean(binary_vec.data):.3f}")
    print(f"   Hamming weight: {np.sum(binary_vec.data)}\n")
    
    # Bipolar vectors
    print("2. Bipolar Vectors (-1/+1):")
    bipolar_vec = generate_random_vector(dimension, BipolarVector)
    print(f"   Sample values: {bipolar_vec.data[:10]}")
    print(f"   Mean: {np.mean(bipolar_vec.data):.3f}")
    print(f"   Norm: {np.linalg.norm(bipolar_vec.data):.3f}\n")
    
    # Ternary vectors
    print("3. Ternary Vectors (-1/0/+1):")
    ternary_vec = generate_random_vector(dimension, TernaryVector, sparsity=0.1)
    print(f"   Sample values: {ternary_vec.data[:20]}")
    print(f"   Sparsity: {np.mean(ternary_vec.data == 0):.3f}")
    print(f"   Active elements: {np.sum(ternary_vec.data != 0)}\n")
    
    # Complex vectors
    print("4. Complex Vectors (unit circle):")
    complex_vec = generate_random_vector(dimension, ComplexVector)
    print(f"   Sample values: {complex_vec.data[:5]}")
    print(f"   Magnitudes: {np.abs(complex_vec.data[:5])}")
    print(f"   All unit magnitude: {np.allclose(np.abs(complex_vec.data), 1.0)}\n")
    
    # Integer vectors
    print("5. Integer Vectors (modular):")
    integer_vec = generate_random_vector(dimension, IntegerVector, modulus=256)
    print(f"   Sample values: {integer_vec.data[:10]}")
    print(f"   Range: [{np.min(integer_vec.data)}, {np.max(integer_vec.data)}]")
    print(f"   Mean: {np.mean(integer_vec.data):.3f}\n")


def demonstrate_binding_operations():
    """Demonstrate different binding operations in VSA."""
    print("=== VSA Binding Operations Demo ===\n")
    
    dimension = 1000
    
    # Create VSA with binary vectors and XOR binding
    print("1. XOR Binding (Binary Vectors):")
    vsa_xor = create_vsa(
        dimension=dimension,
        vector_type='binary',
        vsa_type='bsc'  # BSC uses XOR binding by default
    )
    
    # Create symbols
    apple = vsa_xor.generate_vector()
    red = vsa_xor.generate_vector()
    
    # Bind them
    red_apple = vsa_xor.bind(red, apple)
    
    # Unbind to retrieve
    retrieved_apple = vsa_xor.unbind(red_apple, red)
    similarity = vsa_xor.similarity(retrieved_apple, apple)
    print(f"   Binding: red ⊕ apple")
    print(f"   Unbinding: red_apple ⊕ red → apple")
    print(f"   Similarity to apple: {similarity:.3f}")
    # XOR is self-inverse: A ⊕ A = 0
    zero_vec = vsa_xor.bind(red, red)
    print(f"   Self-inverse property (red ⊕ red ≈ 0): mean = {np.mean(zero_vec):.3f}\n")
    
    # Multiplication binding
    print("2. Multiplication Binding (Bipolar Vectors):")
    vsa_mult = create_vsa(
        dimension=dimension,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )
    
    car = vsa_mult.generate_vector()
    blue = vsa_mult.generate_vector()
    blue_car = vsa_mult.bind(blue, car)
    retrieved_car = vsa_mult.unbind(blue_car, blue)
    print(f"   Binding: blue × car")
    print(f"   Similarity to car: {vsa_mult.similarity(retrieved_car, car):.3f}\n")
    
    # Convolution binding (HRR-style)
    print("3. Convolution Binding (Complex Vectors):")
    vsa_conv = create_vsa(
        dimension=dimension,
        vector_type='complex',
        vsa_type='fhrr'  # FHRR uses convolution
    )
    
    book = vsa_conv.generate_vector()
    science = vsa_conv.generate_vector()
    science_book = vsa_conv.bind(science, book)
    retrieved_book = vsa_conv.unbind(science_book, science)
    print(f"   Binding: science ⊛ book")
    print(f"   Similarity to book: {vsa_conv.similarity(retrieved_book, book):.3f}\n")
    
    # MAP binding
    print("4. MAP Binding (Multiply-Add-Permute):")
    vsa_map = create_vsa(
        dimension=dimension,
        vector_type='bipolar',
        vsa_type='map'
    )
    
    cat = vsa_map.generate_vector()
    black = vsa_map.generate_vector()
    black_cat = vsa_map.bind(black, cat)
    print(f"   MAP combines multiplication, addition, and permutation")
    print(f"   More robust for multiple bindings\n")
    
    # Permutation binding
    print("5. Permutation Binding:")
    vsa_perm = create_vsa(
        dimension=dimension,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='permutation'
    )
    
    dog = vsa_perm.generate_vector()
    brown = vsa_perm.generate_vector()
    brown_dog = vsa_perm.bind(brown, dog)
    print(f"   Uses cyclic shifts for binding")
    print(f"   Efficient for sequential data\n")


def demonstrate_vsa_operations():
    """Demonstrate common VSA operations."""
    print("=== VSA Operations Demo ===\n")
    
    # Create VSA instance
    vsa = create_vsa(
        dimension=1000,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )
    
    # Bundling (superposition)
    print("1. Bundling (Superposition):")
    fruits = ['apple', 'banana', 'orange', 'grape']
    fruit_vectors = [vsa.generate_vector() for f in fruits]
    fruit_bundle = vsa.bundle(fruit_vectors)
    
    # Check similarity to each fruit
    for fruit, vec in zip(fruits, fruit_vectors):
        sim = vsa.similarity(fruit_bundle, vec)
        print(f"   Bundle similarity to {fruit}: {sim:.3f}")
    print()
    
    # Weighted bundling
    print("2. Weighted Bundling:")
    weights = [0.5, 0.3, 0.15, 0.05]
    weighted_bundle = vsa.bundle(fruit_vectors, weights=weights)
    for fruit, vec, weight in zip(fruits, fruit_vectors, weights):
        sim = vsa.similarity(weighted_bundle, vec)
        print(f"   Weighted bundle to {fruit} (weight={weight}): {sim:.3f}")
    print()
    
    # Permutation operations
    print("3. Permutation Operations:")
    vector = vsa.generate_vector()
    perm1 = vsa.permute(vector, shift=1)
    perm2 = vsa.permute(vector, shift=2)
    inv_perm = vsa.permute(perm1, shift=-1)
    
    print(f"   Original → Permute(1): similarity = {vsa.similarity(vector, perm1):.3f}")
    print(f"   Permute(1) → Permute(2): similarity = {vsa.similarity(perm1, perm2):.3f}")
    print(f"   Permute(1) → Permute(-1): similarity = {vsa.similarity(vector, inv_perm):.3f}")
    print()
    
    # Thinning (sparsification)
    print("4. Thinning Operations:")
    dense_vec = vsa.generate_vector()
    thinned_vec = vsa.thin(dense_vec, rate=0.9)
    
    print(f"   Original non-zero elements: {np.count_nonzero(dense_vec)}")
    print(f"   Thinned non-zero elements: {np.count_nonzero(thinned_vec)}")
    print(f"   Similarity after thinning: {vsa.similarity(dense_vec, thinned_vec):.3f}\n")


def demonstrate_symbolic_reasoning():
    """Demonstrate basic symbolic reasoning with VSA."""
    print("=== Symbolic Reasoning Demo ===\n")
    
    # Create VSA for reasoning
    vsa = create_vsa(
        dimension=2000,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )
    
    # 1. Role-filler binding
    print("1. Role-Filler Binding:")
    # Roles
    color_role = vsa.generate_vector()
    size_role = vsa.generate_vector()
    type_role = vsa.generate_vector()
    
    # Fillers
    red = vsa.generate_vector()
    large = vsa.generate_vector()
    apple = vsa.generate_vector()
    
    # Create structured representation
    red_large_apple = vsa.bundle([
        vsa.bind(color_role, red),
        vsa.bind(size_role, large),
        vsa.bind(type_role, apple)
    ])
    
    # Query the structure
    queried_color = vsa.unbind(red_large_apple, color_role)
    queried_size = vsa.unbind(red_large_apple, size_role)
    queried_type = vsa.unbind(red_large_apple, type_role)
    
    print(f"   Query COLOR: similarity to 'red' = {vsa.similarity(queried_color, red):.3f}")
    print(f"   Query SIZE: similarity to 'large' = {vsa.similarity(queried_size, large):.3f}")
    print(f"   Query TYPE: similarity to 'apple' = {vsa.similarity(queried_type, apple):.3f}\n")
    
    # 2. Analogical reasoning
    print("2. Analogical Reasoning (A:B :: C:?):")
    # King:Queen :: Man:?
    king = vsa.generate_vector()
    queen = vsa.generate_vector()
    man = vsa.generate_vector()
    woman = vsa.generate_vector()
    
    # Learn the transformation
    # For multiplication binding, inverse is the same vector (element-wise division)
    transform = vsa.bind(queen, king)
    
    # Apply to man
    result = vsa.bind(transform, man)
    
    print(f"   King:Queen :: Man:?")
    print(f"   Similarity to 'woman': {vsa.similarity(result, woman):.3f}")
    print(f"   (Note: This is a simple demonstration; real word embeddings would work better)\n")
    
    # 3. Set membership
    print("3. Set Membership:")
    # Create sets
    mammals = vsa.bundle([vsa.generate_vector() for x in ['dog', 'cat', 'horse', 'cow']])
    birds = vsa.bundle([vsa.generate_vector() for x in ['eagle', 'sparrow', 'owl', 'robin']])
    
    # Test membership
    dog = vsa.generate_vector()
    eagle = vsa.generate_vector()
    fish = vsa.generate_vector()
    
    print(f"   'dog' in mammals: {vsa.similarity(mammals, dog):.3f}")
    print(f"   'eagle' in mammals: {vsa.similarity(mammals, eagle):.3f}")
    print(f"   'eagle' in birds: {vsa.similarity(birds, eagle):.3f}")
    print(f"   'fish' in mammals: {vsa.similarity(mammals, fish):.3f}\n")


def demonstrate_architectures():
    """Demonstrate different VSA architectures."""
    print("=== VSA Architectures Demo ===\n")
    
    dimension = 1000
    
    # Binary Spatter Codes (BSC)
    print("1. Binary Spatter Codes (BSC):")
    bsc = BSC(dimension=dimension)
    a = bsc.generate_vector()
    b = bsc.generate_vector()
    c = bsc.bind(a, b)
    print(f"   Vector type: Binary")
    print(f"   Binding: XOR")
    print(f"   Efficient for hardware implementation")
    # Test unbinding
    retrieved_a = bsc.unbind(c, b)
    print(f"   Unbind test: similarity(unbind(c,b), a) = {bsc.similarity(retrieved_a, a):.3f}\n")
    
    # MAP Architecture
    print("2. MAP Architecture:")
    map_arch = MAP(dimension=dimension)
    x = map_arch.generate_vector()
    y = map_arch.generate_vector()
    z = map_arch.bind(x, y)
    print(f"   Vector type: Bipolar")
    print(f"   Binding: Multiply-Add-Permute")
    print(f"   Good for multiple bindings")
    # Test unbinding (approximate)
    retrieved_x = map_arch.unbind(z, y)
    print(f"   Unbind test: similarity(unbind(z,y), x) = {map_arch.similarity(retrieved_x, x):.3f}\n")
    
    # FHRR (Fourier HRR)
    print("3. Fourier HRR (FHRR):")
    fhrr = FHRR(dimension=dimension)
    p = fhrr.generate_vector()
    q = fhrr.generate_vector()
    r = fhrr.bind(p, q)
    print(f"   Vector type: Complex")
    print(f"   Binding: Frequency domain convolution")
    print(f"   Compatible with HRR")
    # Test unbinding
    retrieved_p = fhrr.unbind(r, q)
    print(f"   Unbind test: similarity(unbind(r,q), p) = {fhrr.similarity(retrieved_p, p):.3f}\n")
    
    # Sparse VSA
    print("4. Sparse VSA:")
    sparse = SparseVSA(dimension=dimension, sparsity=0.95)
    s1 = sparse.generate_vector()
    s2 = sparse.generate_vector()
    s3 = sparse.bind(s1, s2)
    print(f"   Vector type: Ternary")
    print(f"   Very sparse representations")
    print(f"   Memory efficient")
    print(f"   Sparsity of s1: {np.mean(s1 == 0):.3f}")
    print(f"   Sparsity of bound s3: {np.mean(s3 == 0):.3f}\n")


def main():
    """Run all VSA demonstrations."""
    print("\n" + "="*60)
    print("VECTOR SYMBOLIC ARCHITECTURES (VSA) - BASIC DEMO")
    print("="*60 + "\n")
    
    # Run demonstrations
    demonstrate_vector_types()
    print("\n" + "-"*60 + "\n")
    
    demonstrate_binding_operations()
    print("\n" + "-"*60 + "\n")
    
    demonstrate_vsa_operations()
    print("\n" + "-"*60 + "\n")
    
    demonstrate_symbolic_reasoning()
    print("\n" + "-"*60 + "\n")
    
    demonstrate_architectures()
    
    print("\n" + "="*60)
    print("VSA Demo Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()