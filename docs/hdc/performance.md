# HDC Performance Guide

This guide covers performance characteristics, optimization strategies, and best practices for the Hyperdimensional Computing (HDC) module.

## Table of Contents

1. [Performance Characteristics](#performance-characteristics)
2. [Dimension Selection](#dimension-selection)
3. [Vector Type Performance](#vector-type-performance)
4. [Memory Usage](#memory-usage)
5. [Computational Complexity](#computational-complexity)
6. [Optimization Strategies](#optimization-strategies)
7. [Benchmarks](#benchmarks)
8. [GPU Acceleration](#gpu-acceleration)

## Performance Characteristics

### Key Performance Factors

1. **Dimension (D)**: Higher dimensions provide better capacity but increase computation
2. **Vector Type**: Binary vectors are fastest, complex vectors slowest
3. **Operation Type**: XOR/permutation are O(D), convolution is O(D log D)
4. **Memory Access**: Sequential access patterns are much faster than random
5. **Parallelization**: Most operations are embarrassingly parallel

### Typical Performance Metrics

| Operation | Binary | Bipolar | Ternary | Complex |
|-----------|--------|----------|----------|----------|
| Generate | 0.1ms | 0.2ms | 0.3ms | 0.5ms |
| Bind (XOR) | 0.05ms | - | - | - |
| Bind (multiply) | - | 0.1ms | 0.15ms | 0.3ms |
| Bundle | 0.1ms | 0.15ms | 0.2ms | 0.4ms |
| Similarity | 0.05ms | 0.1ms | 0.15ms | 0.3ms |

*Times for D=10,000 on modern CPU*

## Dimension Selection

### Capacity vs Performance Trade-off

```python
import numpy as np
import matplotlib.pyplot as plt
from cognitive_computing.hdc import ItemMemory
from cognitive_computing.hdc.utils import estimate_capacity

# Test different dimensions
dimensions = [1000, 2000, 5000, 10000, 20000, 50000]
capacities = []
query_times = []

for D in dimensions:
    # Create memory
    memory = ItemMemory(dimension=D)
    
    # Estimate capacity
    capacity = estimate_capacity(memory, target_accuracy=0.9)
    capacities.append(capacity)
    
    # Measure query time
    import time
    n_items = min(100, capacity // 2)
    for i in range(n_items):
        memory.add(f"item_{i}", memory.generate_random())
    
    # Time queries
    query_vector = memory.generate_random()
    start = time.time()
    for _ in range(100):
        memory.query_similar(query_vector, k=10)
    query_time = (time.time() - start) / 100
    query_times.append(query_time * 1000)  # Convert to ms

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(dimensions, capacities, 'b-o')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('Capacity (items)')
ax1.set_title('Memory Capacity vs Dimension')
ax1.grid(True)

ax2.plot(dimensions, query_times, 'r-o')
ax2.set_xlabel('Dimension')
ax2.set_ylabel('Query Time (ms)')
ax2.set_title('Query Performance vs Dimension')
ax2.grid(True)

plt.tight_layout()
```

### Recommended Dimensions

| Application | Recommended D | Rationale |
|-------------|---------------|-----------|
| Small demos | 1,000 | Fast, limited capacity |
| Prototype | 5,000 | Good balance |
| Production | 10,000 | Standard choice |
| High capacity | 20,000+ | When accuracy critical |
| Research | 50,000+ | Maximum separation |

### Dimension Guidelines

```python
def recommend_dimension(n_items, target_accuracy=0.9, safety_factor=2.0):
    """Recommend dimension based on expected number of items."""
    # Empirical formula: D ≈ n_items * log(n_items) * safety_factor
    # for target_accuracy = 0.9
    
    if n_items < 10:
        return 1000  # Minimum dimension
    elif n_items < 100:
        return 5000
    elif n_items < 1000:
        return 10000
    elif n_items < 10000:
        return 20000
    else:
        # For large sets, use formula
        import math
        D = int(n_items * math.log(n_items) * safety_factor)
        # Round to nearest 1000
        return ((D + 500) // 1000) * 1000

# Examples
print(f"50 items: D = {recommend_dimension(50)}")
print(f"500 items: D = {recommend_dimension(500)}")
print(f"5000 items: D = {recommend_dimension(5000)}")
```

## Vector Type Performance

### Performance Comparison

```python
import time
from cognitive_computing.hdc import create_hdc

dimension = 10000
n_operations = 1000

vector_types = ["binary", "bipolar", "ternary", "level"]
results = {}

for vtype in vector_types:
    hdc = create_hdc(dimension=dimension, vector_type=vtype)
    
    # Generate vectors
    vectors = [hdc.generate_random() for _ in range(100)]
    
    # Time operations
    times = {}
    
    # Generation time
    start = time.time()
    for _ in range(n_operations):
        hdc.generate_random()
    times['generate'] = time.time() - start
    
    # Bundle time
    start = time.time()
    for _ in range(n_operations):
        hdc.bundle(vectors[:10])
    times['bundle'] = time.time() - start
    
    # Bind time (if applicable)
    if hasattr(hdc, 'bind'):
        start = time.time()
        for _ in range(n_operations):
            hdc.bind(vectors[0], vectors[1])
        times['bind'] = time.time() - start
    
    # Similarity time
    start = time.time()
    for _ in range(n_operations):
        hdc.similarity(vectors[0], vectors[1])
    times['similarity'] = time.time() - start
    
    results[vtype] = times

# Display results
for vtype, times in results.items():
    print(f"\n{vtype.upper()} vectors:")
    for op, t in times.items():
        print(f"  {op}: {t/n_operations*1000:.3f} ms/op")
```

### Memory Footprint

| Vector Type | Bits/element | Memory for D=10,000 | Relative Size |
|-------------|--------------|---------------------|---------------|
| Binary | 1 | 1.25 KB | 1.0x |
| Bipolar | 8 | 10 KB | 8x |
| Ternary | 2 | 2.5 KB | 2x |
| Level-5 | 8 | 10 KB | 8x |
| Complex | 64 | 80 KB | 64x |

### Choosing Vector Types

```python
def choose_vector_type(constraints):
    """Choose optimal vector type based on constraints."""
    memory_limited = constraints.get('memory_limited', False)
    need_binding = constraints.get('need_binding', True)
    need_weighted = constraints.get('need_weighted', False)
    need_rotation = constraints.get('need_rotation', False)
    
    if need_rotation:
        return "complex"  # Only complex supports rotation
    elif memory_limited and not need_weighted:
        return "binary"  # Most memory efficient
    elif memory_limited and need_weighted:
        return "ternary"  # Sparse with weights
    elif need_binding:
        return "bipolar"  # Best general purpose
    else:
        return "binary"  # Default to fastest

# Examples
print(choose_vector_type({'memory_limited': True}))  # "binary"
print(choose_vector_type({'need_rotation': True}))   # "complex"
print(choose_vector_type({'need_weighted': True, 'memory_limited': True}))  # "ternary"
```

## Memory Usage

### Memory Profiling

```python
import sys
from cognitive_computing.hdc import ItemMemory

def profile_memory_usage():
    """Profile memory usage of HDC structures."""
    memories = {}
    
    for D in [1000, 5000, 10000, 20000]:
        memory = ItemMemory(dimension=D, vector_type="binary")
        
        # Add items
        for i in range(100):
            memory.add(f"item_{i}", memory.generate_random())
        
        # Estimate size
        size_bytes = sys.getsizeof(memory._vectors) + sys.getsizeof(memory._names)
        memories[D] = size_bytes / (1024 * 1024)  # Convert to MB
    
    return memories

# Run profiling
usage = profile_memory_usage()
for D, mb in usage.items():
    print(f"D={D}: {mb:.2f} MB for 100 items")
```

### Memory Optimization Strategies

```python
# 1. Use sparse representations
from cognitive_computing.hdc import TernaryHypervector

# Sparse ternary vector (90% zeros)
sparse_hdc = create_hdc(dimension=10000, vector_type="ternary", sparsity=0.9)

# 2. Lazy generation
class LazyItemMemory(ItemMemory):
    """Generate vectors on-demand instead of storing all."""
    def __init__(self, dimension, seed=42):
        super().__init__(dimension)
        self._seeds = {}
        self._base_seed = seed
    
    def add(self, name, vector=None):
        # Store seed instead of vector
        self._seeds[name] = len(self._seeds)
    
    def get(self, name):
        # Generate from seed
        seed = self._seeds.get(name)
        if seed is None:
            return None
        np.random.seed(self._base_seed + seed)
        return self.generate_random()

# 3. Batch processing
def process_in_batches(data, batch_size=1000):
    """Process data in batches to limit memory usage."""
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        # Process batch
        batch_results = process_batch(batch)
        results.extend(batch_results)
        # Force garbage collection
        import gc
        gc.collect()
    return results
```

## Computational Complexity

### Operation Complexity Analysis

| Operation | Time Complexity | Space Complexity | Parallelizable |
|-----------|----------------|------------------|----------------|
| Generate Random | O(D) | O(D) | Yes |
| Bind (XOR) | O(D) | O(D) | Yes |
| Bind (multiply) | O(D) | O(D) | Yes |
| Bind (convolution) | O(D log D) | O(D) | Yes (FFT) |
| Bundle | O(kD) | O(D) | Yes |
| Permute | O(D) | O(D) | No |
| Similarity | O(D) | O(1) | Yes |
| Query k-NN | O(nD + n log k) | O(n) | Partially |

### Scaling Analysis

```python
import time
import numpy as np
import matplotlib.pyplot as plt

def benchmark_scaling(max_dimension=50000, step=5000):
    """Benchmark operation scaling with dimension."""
    dimensions = list(range(1000, max_dimension + 1, step))
    
    # Operations to benchmark
    operations = {
        'generate': lambda hdc, d: hdc.generate_random(),
        'bind': lambda hdc, d: hdc.bind(np.random.randint(0, 2, d), 
                                         np.random.randint(0, 2, d)),
        'bundle': lambda hdc, d: hdc.bundle([np.random.randint(0, 2, d) 
                                            for _ in range(10)]),
        'similarity': lambda hdc, d: hdc.similarity(np.random.randint(0, 2, d),
                                                   np.random.randint(0, 2, d))
    }
    
    results = {op: [] for op in operations}
    
    for D in dimensions:
        hdc = create_hdc(dimension=D, vector_type="binary")
        
        for op_name, op_func in operations.items():
            # Time operation
            times = []
            for _ in range(100):
                start = time.perf_counter()
                op_func(hdc, D)
                times.append(time.perf_counter() - start)
            
            avg_time = np.mean(times) * 1000  # Convert to ms
            results[op_name].append(avg_time)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for op_name, times in results.items():
        plt.plot(dimensions, times, label=op_name, marker='o')
    
    plt.xlabel('Dimension')
    plt.ylabel('Time (ms)')
    plt.title('Operation Scaling with Dimension')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return dimensions, results
```

## Optimization Strategies

### 1. Vectorization

```python
# Slow: element-wise operations
def slow_bundle(vectors):
    result = np.zeros_like(vectors[0])
    for v in vectors:
        for i in range(len(v)):
            result[i] += v[i]
    return (result > len(vectors) // 2).astype(int)

# Fast: vectorized operations
def fast_bundle(vectors):
    return (np.sum(vectors, axis=0) > len(vectors) // 2).astype(int)

# Benchmark
import time
vectors = [np.random.randint(0, 2, 10000) for _ in range(100)]

start = time.time()
slow_result = slow_bundle(vectors)
slow_time = time.time() - start

start = time.time()
fast_result = fast_bundle(vectors)
fast_time = time.time() - start

print(f"Slow: {slow_time:.3f}s, Fast: {fast_time:.3f}s")
print(f"Speedup: {slow_time / fast_time:.1f}x")
```

### 2. Batch Processing

```python
from cognitive_computing.hdc import HDCClassifier

# Efficient batch prediction
class BatchHDCClassifier(HDCClassifier):
    def predict_batch(self, X, batch_size=1000):
        """Predict in batches for memory efficiency."""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        
        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]
            predictions[i:i + batch_size] = self.predict(batch)
        
        return predictions
```

### 3. Caching and Memoization

```python
from functools import lru_cache

class CachedItemMemory(ItemMemory):
    """Item memory with caching for frequent operations."""
    
    def __init__(self, dimension, cache_size=128):
        super().__init__(dimension)
        self.cache_size = cache_size
        # Create cached methods
        self._cached_similarity = lru_cache(maxsize=cache_size)(self._similarity)
        self._cached_bind = lru_cache(maxsize=cache_size)(self._bind)
    
    def _make_key(self, v1, v2):
        """Create hashable key from vectors."""
        return (v1.tobytes(), v2.tobytes())
    
    def similarity(self, v1, v2):
        """Cached similarity computation."""
        key = self._make_key(v1, v2)
        return self._cached_similarity(key)
    
    def _similarity(self, key):
        """Actual similarity computation."""
        v1 = np.frombuffer(key[0], dtype=self.dtype)
        v2 = np.frombuffer(key[1], dtype=self.dtype)
        return super().similarity(v1, v2)
```

### 4. Parallel Processing

```python
from multiprocessing import Pool
import numpy as np

def parallel_similarity_search(memory, queries, k=10, n_workers=4):
    """Parallel k-NN search for multiple queries."""
    
    def search_worker(args):
        query, vectors, names = args
        similarities = [memory.similarity(query, v) for v in vectors]
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        return [(names[i], similarities[i]) for i in top_k_idx]
    
    # Prepare data for workers
    vectors = list(memory._vectors.values())
    names = list(memory._names.values())
    worker_args = [(q, vectors, names) for q in queries]
    
    # Parallel execution
    with Pool(n_workers) as pool:
        results = pool.map(search_worker, worker_args)
    
    return results
```

### 5. Memory-Mapped Arrays

```python
import numpy as np

class MemmapItemMemory(ItemMemory):
    """Use memory-mapped arrays for large vector storage."""
    
    def __init__(self, dimension, filename='vectors.dat', max_items=100000):
        super().__init__(dimension)
        self.filename = filename
        self.max_items = max_items
        
        # Create memory-mapped array
        self.vectors = np.memmap(
            filename, 
            dtype='uint8', 
            mode='w+',
            shape=(max_items, dimension)
        )
        self.n_items = 0
    
    def add(self, name, vector):
        """Add vector to memory-mapped storage."""
        if self.n_items >= self.max_items:
            raise ValueError("Maximum capacity reached")
        
        self.vectors[self.n_items] = vector
        self._names[self.n_items] = name
        self._name_to_idx[name] = self.n_items
        self.n_items += 1
    
    def get(self, name):
        """Retrieve vector from memory-mapped storage."""
        idx = self._name_to_idx.get(name)
        if idx is None:
            return None
        return self.vectors[idx].copy()
```

## Benchmarks

### Comprehensive Benchmark Suite

```python
import time
import numpy as np
from cognitive_computing.hdc import create_hdc, ItemMemory, HDCClassifier

def run_comprehensive_benchmark():
    """Run comprehensive HDC benchmark suite."""
    results = {}
    
    # Test configurations
    dimensions = [1000, 5000, 10000, 20000]
    vector_types = ["binary", "bipolar", "ternary"]
    
    for D in dimensions:
        for vtype in vector_types:
            config_name = f"{vtype}_D{D}"
            print(f"\nBenchmarking {config_name}...")
            
            # Create HDC instance
            hdc = create_hdc(dimension=D, vector_type=vtype)
            
            # Benchmark basic operations
            n_ops = 1000
            vectors = [hdc.generate_random() for _ in range(100)]
            
            # Generation
            start = time.perf_counter()
            for _ in range(n_ops):
                hdc.generate_random()
            gen_time = (time.perf_counter() - start) / n_ops * 1000
            
            # Bundle
            start = time.perf_counter()
            for _ in range(n_ops):
                hdc.bundle(vectors[:10])
            bundle_time = (time.perf_counter() - start) / n_ops * 1000
            
            # Similarity
            start = time.perf_counter()
            for _ in range(n_ops):
                hdc.similarity(vectors[0], vectors[1])
            sim_time = (time.perf_counter() - start) / n_ops * 1000
            
            # Memory operations
            memory = ItemMemory(dimension=D, vector_type=vtype)
            for i in range(100):
                memory.add(f"item_{i}", vectors[i])
            
            # Query
            start = time.perf_counter()
            for _ in range(100):
                memory.query_similar(vectors[0], k=10)
            query_time = (time.perf_counter() - start) / 100 * 1000
            
            # Classification
            X = np.random.randn(100, 20)
            y = np.random.randint(0, 5, 100)
            clf = HDCClassifier(dimension=D, vector_type=vtype)
            
            # Training
            start = time.perf_counter()
            clf.fit(X, y)
            train_time = (time.perf_counter() - start) * 1000
            
            # Prediction
            start = time.perf_counter()
            clf.predict(X)
            pred_time = (time.perf_counter() - start) / 100 * 1000
            
            # Store results
            results[config_name] = {
                'dimension': D,
                'vector_type': vtype,
                'generate_ms': gen_time,
                'bundle_ms': bundle_time,
                'similarity_ms': sim_time,
                'query_ms': query_time,
                'train_ms': train_time,
                'predict_ms': pred_time
            }
    
    return results

# Run benchmark
benchmark_results = run_comprehensive_benchmark()

# Display results
print("\n" + "="*80)
print("BENCHMARK RESULTS")
print("="*80)
print(f"{'Config':<20} {'Gen':<8} {'Bundle':<8} {'Sim':<8} {'Query':<8} {'Train':<8} {'Pred':<8}")
print("-"*80)

for config, results in benchmark_results.items():
    print(f"{config:<20} "
          f"{results['generate_ms']:<8.3f} "
          f"{results['bundle_ms']:<8.3f} "
          f"{results['similarity_ms']:<8.3f} "
          f"{results['query_ms']:<8.3f} "
          f"{results['train_ms']:<8.1f} "
          f"{results['predict_ms']:<8.3f}")
```

### Performance vs Accuracy Trade-off

```python
def analyze_performance_accuracy_tradeoff():
    """Analyze trade-off between performance and accuracy."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    
    # Generate dataset
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_classes=5, n_informative=15)
    
    results = []
    
    for D in [500, 1000, 2000, 5000, 10000, 20000]:
        # Create classifier
        clf = HDCClassifier(dimension=D)
        
        # Measure accuracy
        scores = cross_val_score(clf, X, y, cv=5)
        accuracy = scores.mean()
        
        # Measure speed
        start = time.time()
        clf.fit(X, y)
        train_time = time.time() - start
        
        start = time.time()
        for _ in range(10):
            clf.predict(X)
        predict_time = (time.time() - start) / 10
        
        results.append({
            'dimension': D,
            'accuracy': accuracy,
            'train_time': train_time,
            'predict_time': predict_time,
            'ops_per_sec': len(X) / predict_time
        })
    
    # Plot results
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    dims = [r['dimension'] for r in results]
    accs = [r['accuracy'] for r in results]
    ops = [r['ops_per_sec'] for r in results]
    
    ax1.plot(dims, accs, 'b-o')
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Dimension')
    ax1.grid(True)
    
    ax2.plot(dims, ops, 'r-o')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Operations/sec')
    ax2.set_title('Speed vs Dimension')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results
```

## GPU Acceleration

### CuPy Backend

```python
try:
    import cupy as cp
    
    class GPUItemMemory(ItemMemory):
        """GPU-accelerated item memory using CuPy."""
        
        def __init__(self, dimension, vector_type="binary"):
            super().__init__(dimension, vector_type)
            self.use_gpu = True
            
        def generate_random(self):
            """Generate random vector on GPU."""
            if self.vector_type == "binary":
                return cp.random.randint(0, 2, self.dimension)
            elif self.vector_type == "bipolar":
                return cp.random.choice([-1, 1], self.dimension)
            
        def similarity(self, v1, v2):
            """GPU-accelerated similarity."""
            v1_gpu = cp.asarray(v1)
            v2_gpu = cp.asarray(v2)
            
            if self.vector_type == "binary":
                # Hamming similarity
                return float(cp.sum(v1_gpu == v2_gpu) / self.dimension)
            else:
                # Cosine similarity
                dot = cp.dot(v1_gpu, v2_gpu)
                norm = cp.linalg.norm(v1_gpu) * cp.linalg.norm(v2_gpu)
                return float(dot / norm)
        
        def bundle(self, vectors):
            """GPU-accelerated bundling."""
            vectors_gpu = cp.stack([cp.asarray(v) for v in vectors])
            
            if self.vector_type == "binary":
                # Majority vote
                sums = cp.sum(vectors_gpu, axis=0)
                return (sums > len(vectors) // 2).astype(cp.uint8)
            else:
                # Average and normalize
                avg = cp.mean(vectors_gpu, axis=0)
                return cp.sign(avg).astype(cp.int8)
    
    # Benchmark GPU vs CPU
    def benchmark_gpu():
        D = 10000
        n_vectors = 1000
        
        # CPU version
        cpu_memory = ItemMemory(dimension=D)
        cpu_vectors = [cpu_memory.generate_random() for _ in range(n_vectors)]
        
        start = time.time()
        for i in range(n_vectors-1):
            cpu_memory.similarity(cpu_vectors[i], cpu_vectors[i+1])
        cpu_time = time.time() - start
        
        # GPU version
        gpu_memory = GPUItemMemory(dimension=D)
        gpu_vectors = [gpu_memory.generate_random() for _ in range(n_vectors)]
        
        start = time.time()
        for i in range(n_vectors-1):
            gpu_memory.similarity(gpu_vectors[i], gpu_vectors[i+1])
        gpu_time = time.time() - start
        
        print(f"CPU time: {cpu_time:.3f}s")
        print(f"GPU time: {gpu_time:.3f}s")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x")
        
except ImportError:
    print("CuPy not available. Install with: pip install cupy")
```

### PyTorch Backend

```python
try:
    import torch
    
    class TorchHDC:
        """PyTorch-based HDC for GPU acceleration."""
        
        def __init__(self, dimension, device='cuda' if torch.cuda.is_available() else 'cpu'):
            self.dimension = dimension
            self.device = device
            
        def generate_random_batch(self, batch_size):
            """Generate batch of random vectors."""
            return torch.randint(0, 2, (batch_size, self.dimension), 
                                device=self.device, dtype=torch.uint8)
        
        def bundle_batch(self, vectors):
            """Bundle batch of vectors efficiently."""
            # vectors: (batch_size, n_vectors, dimension)
            sums = torch.sum(vectors, dim=1)
            threshold = vectors.shape[1] // 2
            return (sums > threshold).to(torch.uint8)
        
        def similarity_matrix(self, vectors1, vectors2):
            """Compute pairwise similarities between two sets."""
            # Convert to float for computation
            v1 = vectors1.float()
            v2 = vectors2.float()
            
            # Compute dot products
            dots = torch.matmul(v1, v2.t())
            
            # Normalize
            norms1 = torch.norm(v1, dim=1, keepdim=True)
            norms2 = torch.norm(v2, dim=1, keepdim=True)
            
            similarities = dots / (norms1 @ norms2.t())
            return similarities
    
    # Example usage
    if torch.cuda.is_available():
        hdc = TorchHDC(dimension=10000)
        
        # Generate batch
        batch = hdc.generate_random_batch(1000)
        
        # Compute similarity matrix
        sim_matrix = hdc.similarity_matrix(batch[:100], batch[100:200])
        print(f"Similarity matrix shape: {sim_matrix.shape}")
        print(f"Mean similarity: {sim_matrix.mean():.4f}")
        
except ImportError:
    print("PyTorch not available. Install with: pip install torch")
```

## Summary and Recommendations

### Quick Reference

1. **Default Configuration**:
   - Dimension: 10,000
   - Vector type: Binary (speed) or Bipolar (properties)
   - Encoding: Thermometer (numeric) or Random Projection (general)

2. **Performance Tips**:
   - Use batch operations whenever possible
   - Pre-allocate arrays for better memory performance
   - Consider GPU for large-scale operations (>10k vectors)
   - Cache frequently used computations

3. **Scaling Guidelines**:
   - <100 items: D=5,000 sufficient
   - 100-1,000 items: D=10,000 recommended
   - 1,000-10,000 items: D=20,000 or higher
   - >10,000 items: Consider distributed or GPU implementation

4. **Optimization Priority**:
   - Vectorize all operations (10-100x speedup)
   - Use appropriate data types (2-8x memory savings)
   - Parallelize when possible (2-8x speedup on multicore)
   - GPU acceleration for large scale (10-50x speedup)

5. **Common Bottlenecks**:
   - Element-wise operations in Python loops
   - Unnecessary type conversions
   - Not using batch operations
   - Poor memory access patterns
   - Oversized dimensions for the task