# SPA Performance Guide

This guide provides comprehensive information about optimizing Semantic Pointer Architecture (SPA) systems for performance, including benchmarks, scaling characteristics, and optimization strategies.

## Table of Contents
1. [Performance Characteristics](#performance-characteristics)
2. [Benchmarks](#benchmarks)
3. [Memory Usage](#memory-usage)
4. [Computational Complexity](#computational-complexity)
5. [Optimization Strategies](#optimization-strategies)
6. [Scaling Guidelines](#scaling-guidelines)
7. [Hardware Considerations](#hardware-considerations)
8. [Profiling and Monitoring](#profiling-and-monitoring)

## Performance Characteristics

### Core Operations Performance

| Operation | Time Complexity | Typical Time (d=512) | Typical Time (d=1024) |
|-----------|----------------|---------------------|----------------------|
| Binding (⊛) | O(d log d) | ~50 μs | ~120 μs |
| Unbinding | O(d log d) | ~50 μs | ~120 μs |
| Bundling (+) | O(d) | ~2 μs | ~4 μs |
| Similarity (·) | O(d) | ~1 μs | ~2 μs |
| Cleanup | O(N × d) | ~100 μs (N=100) | ~200 μs (N=100) |
| Normalization | O(d) | ~3 μs | ~6 μs |

### Module Performance

| Module Type | Update Time | Memory Overhead | Notes |
|------------|-------------|-----------------|-------|
| State | O(d) | 2 × d × 4 bytes | With feedback |
| Memory | O(d log d) | capacity × d × 8 bytes | Per recall |
| Buffer | O(d) | 2 × d × 4 bytes | With gating |
| Gate | O(1) | 8 bytes | Minimal overhead |
| Compare | O(d) | d × 4 bytes | Dot product |

### Action Selection Performance

| Component | Complexity | Typical Time | Scaling Factor |
|-----------|-----------|--------------|----------------|
| BasalGanglia | O(A²) | ~10 μs (A=10) | Quadratic in actions |
| Thalamus | O(A × M) | ~20 μs | A actions, M modules |
| Production System | O(P × C) | ~100 μs | P productions, C conditions |

## Benchmarks

### Semantic Pointer Operations Benchmark

```python
import time
import numpy as np
from cognitive_computing.spa import Vocabulary, SemanticPointer

def benchmark_operations(dimension, n_iterations=1000):
    """Benchmark core SPA operations."""
    vocab = Vocabulary(dimension)
    vocab.create_pointer("A")
    vocab.create_pointer("B")
    
    results = {}
    
    # Benchmark binding
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = vocab["A"] * vocab["B"]
    results['binding'] = (time.perf_counter() - start) / n_iterations
    
    # Benchmark unbinding
    bound = vocab["A"] * vocab["B"]
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = bound * ~vocab["B"]
    results['unbinding'] = (time.perf_counter() - start) / n_iterations
    
    # Benchmark bundling
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = vocab["A"] + vocab["B"]
    results['bundling'] = (time.perf_counter() - start) / n_iterations
    
    # Benchmark similarity
    start = time.perf_counter()
    for _ in range(n_iterations):
        sim = vocab["A"] @ vocab["B"]
    results['similarity'] = (time.perf_counter() - start) / n_iterations
    
    return results

# Run benchmarks
for dim in [256, 512, 1024, 2048]:
    results = benchmark_operations(dim)
    print(f"\nDimension {dim}:")
    for op, time_sec in results.items():
        print(f"  {op}: {time_sec * 1e6:.2f} μs")
```

### Vocabulary Scaling Benchmark

```python
def benchmark_vocabulary_scaling(dimension=512, max_items=1000):
    """Benchmark cleanup performance with vocabulary size."""
    vocab = Vocabulary(dimension)
    
    cleanup_times = []
    sizes = [10, 50, 100, 200, 500, 1000]
    
    for size in sizes:
        # Add items up to size
        while len(vocab.pointers) < size:
            vocab.create_pointer(f"ITEM_{len(vocab.pointers)}")
        
        # Create noisy vector
        noisy = vocab[f"ITEM_0"].vector + 0.3 * np.random.randn(dimension)
        
        # Benchmark cleanup
        start = time.perf_counter()
        n_trials = 100
        for _ in range(n_trials):
            matches = vocab.cleanup(noisy, top_n=5)
        cleanup_time = (time.perf_counter() - start) / n_trials
        
        cleanup_times.append((size, cleanup_time))
        print(f"Vocabulary size {size}: {cleanup_time * 1e6:.2f} μs")
    
    return cleanup_times
```

### Action Selection Benchmark

```python
from cognitive_computing.spa import Action, BasalGanglia

def benchmark_action_selection(n_actions_list=[5, 10, 20, 50]):
    """Benchmark action selection with different numbers of actions."""
    results = []
    
    for n_actions in n_actions_list:
        # Create dummy actions
        actions = []
        for i in range(n_actions):
            actions.append(Action(
                f"action_{i}",
                lambda i=i: float(i) / n_actions,  # Varying utilities
                lambda: None
            ))
        
        bg = BasalGanglia(actions)
        
        # Benchmark update
        start = time.perf_counter()
        n_trials = 1000
        for _ in range(n_trials):
            utilities = bg.update({})
        update_time = (time.perf_counter() - start) / n_trials
        
        results.append((n_actions, update_time))
        print(f"{n_actions} actions: {update_time * 1e6:.2f} μs")
    
    return results
```

## Memory Usage

### Memory Footprint Analysis

```python
def analyze_memory_usage(dimension=512):
    """Analyze memory usage of SPA components."""
    import sys
    
    # Semantic pointer memory
    pointer_size = dimension * 4  # float32
    print(f"Single semantic pointer: {pointer_size / 1024:.2f} KB")
    
    # Vocabulary with N items
    for n in [10, 100, 1000]:
        vocab_size = n * pointer_size + n * 50  # vectors + overhead
        print(f"Vocabulary ({n} items): {vocab_size / 1024 / 1024:.2f} MB")
    
    # Module memory
    state_size = 2 * pointer_size  # state + input
    memory_size = lambda capacity: capacity * 2 * pointer_size
    
    print(f"\nState module: {state_size / 1024:.2f} KB")
    print(f"Memory module (100 pairs): {memory_size(100) / 1024 / 1024:.2f} MB")
    print(f"Memory module (1000 pairs): {memory_size(1000) / 1024 / 1024:.2f} MB")
    
    # Full model estimate
    model_size = (
        vocab_size +  # vocabulary
        5 * state_size +  # 5 state modules
        memory_size(100) +  # 1 memory module
        n * 100  # action selection overhead
    )
    print(f"\nTypical full model: {model_size / 1024 / 1024:.2f} MB")
```

### Memory Optimization Strategies

1. **Use appropriate data types**
   ```python
   # Use float32 instead of float64 when precision allows
   config = SPAConfig(dimension=512, dtype=np.float32)
   ```

2. **Lazy initialization**
   ```python
   # Only create pointers when needed
   class LazyVocabulary(Vocabulary):
       def __getitem__(self, key):
           if key not in self.pointers:
               self.create_pointer(key)
           return super().__getitem__(key)
   ```

3. **Memory pooling**
   ```python
   # Reuse vectors for temporary computations
   class VectorPool:
       def __init__(self, dimension, pool_size=10):
           self.pool = [np.zeros(dimension) for _ in range(pool_size)]
           self.available = set(range(pool_size))
       
       def acquire(self):
           if self.available:
               idx = self.available.pop()
               return self.pool[idx]
           return np.zeros(self.dimension)
       
       def release(self, vector):
           # Return vector to pool
           pass
   ```

## Computational Complexity

### Theoretical Complexity

| Operation | Best Case | Average Case | Worst Case | Space |
|-----------|-----------|--------------|------------|-------|
| Binding | O(d log d) | O(d log d) | O(d log d) | O(d) |
| Cleanup | O(log N) | O(N × d) | O(N × d) | O(N × d) |
| Action Selection | O(A) | O(A²) | O(A²) | O(A) |
| Production Match | O(P) | O(P × C) | O(P × C × d) | O(P) |
| Memory Recall | O(d log d) | O(d log d) | O(d log d) | O(capacity × d) |

### Practical Implications

1. **Vocabulary size**: Cleanup is linear in vocabulary size
   - Use hierarchical vocabularies for very large sets
   - Implement approximate nearest neighbor methods

2. **Dimension selection**: Higher dimensions improve capacity but increase computation
   - d=512 good for most applications
   - d=1024+ for complex symbolic structures

3. **Action count**: Quadratic scaling with mutual inhibition
   - Group related actions
   - Use hierarchical action selection

## Optimization Strategies

### 1. Vectorization and SIMD

```python
# Utilize NumPy's vectorized operations
def optimized_cleanup(vocabulary_matrix, query_vector, top_n=5):
    """Optimized cleanup using matrix operations."""
    # Single matrix multiplication instead of loop
    similarities = vocabulary_matrix @ query_vector
    
    # Partial sort for top-n
    top_indices = np.argpartition(similarities, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    
    return top_indices, similarities[top_indices]
```

### 2. Caching and Memoization

```python
from functools import lru_cache

class CachedVocabulary(Vocabulary):
    """Vocabulary with cached operations."""
    
    @lru_cache(maxsize=1000)
    def parse_cached(self, expression: str) -> np.ndarray:
        """Cache parsed expressions."""
        return self.parse(expression).vector
    
    @lru_cache(maxsize=1000)
    def cleanup_cached(self, vector_hash: int, top_n: int = 5):
        """Cache cleanup results for common queries."""
        # In practice, need to handle numpy array hashing
        return self.cleanup(self._unhash_vector(vector_hash), top_n)
```

### 3. Parallel Processing

```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

def parallel_action_evaluation(actions, state, n_workers=4):
    """Evaluate action conditions in parallel."""
    
    def evaluate_action(action):
        return action.condition(state)
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        utilities = list(executor.map(evaluate_action, actions))
    
    return np.array(utilities)
```

### 4. GPU Acceleration

```python
# Using CuPy for GPU operations (if available)
try:
    import cupy as cp
    
    def gpu_binding(a, b):
        """GPU-accelerated circular convolution."""
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        
        # FFT-based convolution
        result = cp.fft.ifft(cp.fft.fft(a_gpu) * cp.fft.fft(b_gpu)).real
        
        return cp.asnumpy(result)
except ImportError:
    gpu_binding = None
```

### 5. Approximate Methods

```python
# Random projection for approximate cleanup
class ApproximateCleanup:
    def __init__(self, vocabulary, n_projections=32):
        self.vocab = vocabulary
        self.projections = np.random.randn(n_projections, vocabulary.dimension)
        self.projections /= np.linalg.norm(self.projections, axis=1, keepdims=True)
        
        # Pre-compute projections of vocabulary
        self.projected_vocab = {}
        for name, pointer in vocabulary.pointers.items():
            self.projected_vocab[name] = self.projections @ pointer.vector
    
    def cleanup(self, vector, top_n=5):
        """Fast approximate cleanup using random projections."""
        projected = self.projections @ vector
        
        # Find candidates using projections
        candidates = []
        for name, proj_vec in self.projected_vocab.items():
            score = np.dot(projected, proj_vec)
            candidates.append((name, score))
        
        # Refine top candidates with exact computation
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:top_n * 2]
        
        # Exact computation for top candidates
        exact_scores = []
        for name, _ in top_candidates:
            exact_sim = self.vocab[name] @ vector
            exact_scores.append((name, exact_sim))
        
        exact_scores.sort(key=lambda x: x[1], reverse=True)
        return exact_scores[:top_n]
```

## Scaling Guidelines

### Dimension Scaling

| Use Case | Recommended Dimension | Max Vocabulary | Notes |
|----------|---------------------|----------------|-------|
| Simple concepts | 256 | ~25 | Fast, limited capacity |
| General purpose | 512 | ~50 | Good balance |
| Complex reasoning | 1024 | ~100 | Higher capacity |
| Large vocabulary | 2048+ | ~200+ | Computational cost |

### Vocabulary Scaling Strategies

1. **Hierarchical Organization**
   ```python
   class HierarchicalVocabulary:
       def __init__(self, dimension):
           self.levels = {
               'category': Vocabulary(dimension),
               'subcategory': {},
               'item': {}
           }
       
       def add_item(self, category, subcategory, item):
           if category not in self.levels['subcategory']:
               self.levels['subcategory'][category] = Vocabulary(dimension)
           # ... hierarchical organization
   ```

2. **Sparse Representations**
   ```python
   from scipy.sparse import csr_matrix
   
   class SparseVocabulary(Vocabulary):
       def create_sparse_pointer(self, name, sparsity=0.1):
           """Create sparse semantic pointer."""
           dense = np.random.randn(self.dimension)
           mask = np.random.random(self.dimension) < sparsity
           dense[~mask] = 0
           sparse = csr_matrix(dense)
           # Store and use sparse representation
   ```

### Module Scaling

1. **Capacity Planning**
   - State modules: O(1) scaling
   - Memory modules: O(capacity) scaling
   - Use multiple specialized memories vs. one large memory

2. **Connection Optimization**
   - Minimize full connections between modules
   - Use sparse transformation matrices
   - Implement lazy evaluation

## Hardware Considerations

### CPU Optimization

```python
# Utilize all CPU cores
import os
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

# NumPy with optimized BLAS
import numpy as np
np.show_config()  # Check for optimized BLAS (MKL, OpenBLAS)
```

### Memory Bandwidth

- Semantic pointer operations are memory-bound
- Optimize data layout for cache efficiency
- Use memory pooling to reduce allocation overhead

### GPU Utilization

```python
# Check for GPU availability
def get_compute_device():
    try:
        import cupy as cp
        return 'gpu', cp
    except ImportError:
        return 'cpu', np

device, xp = get_compute_device()
print(f"Using {device} for computation")
```

## Profiling and Monitoring

### Basic Profiling

```python
import cProfile
import pstats

def profile_spa_operations():
    """Profile SPA operations."""
    vocab = Vocabulary(512)
    for i in range(100):
        vocab.create_pointer(f"ITEM_{i}")
    
    # Operations to profile
    def operations():
        for _ in range(1000):
            result = vocab["ITEM_0"] * vocab["ITEM_1"]
            cleanup = vocab.cleanup(result.vector, top_n=5)
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    operations()
    profiler.disable()
    
    # Analysis
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_operation():
    vocab = Vocabulary(1024)
    
    # Create many pointers
    for i in range(1000):
        vocab.create_pointer(f"CONCEPT_{i}")
    
    # Perform operations
    results = []
    for i in range(100):
        result = vocab[f"CONCEPT_{i}"] * vocab[f"CONCEPT_{i+1}"]
        results.append(result)
    
    return results
```

### Performance Monitoring

```python
class PerformanceMonitor:
    def __init__(self):
        self.timings = defaultdict(list)
    
    def time_operation(self, name):
        """Context manager for timing operations."""
        class Timer:
            def __enter__(timer_self):
                timer_self.start = time.perf_counter()
                return timer_self
            
            def __exit__(timer_self, *args):
                elapsed = time.perf_counter() - timer_self.start
                self.timings[name].append(elapsed)
        
        return Timer()
    
    def report(self):
        """Generate performance report."""
        for operation, times in self.timings.items():
            avg_time = np.mean(times)
            std_time = np.std(times)
            print(f"{operation}: {avg_time*1e3:.2f} ± {std_time*1e3:.2f} ms")

# Usage
monitor = PerformanceMonitor()

with monitor.time_operation("vocabulary_creation"):
    vocab = Vocabulary(512)
    for i in range(100):
        vocab.create_pointer(f"ITEM_{i}")

with monitor.time_operation("binding_operations"):
    for i in range(100):
        result = vocab["ITEM_0"] * vocab[f"ITEM_{i}"]

monitor.report()
```

## Performance Best Practices

### 1. Preprocessing
- Pre-compute frequently used bindings
- Cache cleanup results for common queries
- Pre-normalize vectors

### 2. Batch Operations
```python
def batch_cleanup(vocab, vectors, top_n=5):
    """Efficiently cleanup multiple vectors."""
    # Compute all similarities at once
    vocab_matrix = vocab._pointer_matrix
    similarities = vectors @ vocab_matrix.T
    
    results = []
    for sim_row in similarities:
        top_idx = np.argpartition(sim_row, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(sim_row[top_idx])[::-1]]
        results.append([(vocab._pointer_names[i], sim_row[i]) for i in top_idx])
    
    return results
```

### 3. Lazy Evaluation
```python
class LazySemanticPointer:
    def __init__(self, operation, *operands):
        self.operation = operation
        self.operands = operands
        self._vector = None
    
    @property
    def vector(self):
        if self._vector is None:
            self._vector = self.operation(*[op.vector for op in self.operands])
        return self._vector
```

### 4. Resource Management
- Use context managers for large operations
- Implement cleanup for temporary vectors
- Monitor memory usage in production

## Summary

SPA performance optimization involves:

1. **Choosing appropriate dimensions** for your use case
2. **Optimizing critical operations** like cleanup and binding
3. **Utilizing hardware effectively** (vectorization, parallelism, GPU)
4. **Implementing caching** for repeated operations
5. **Monitoring and profiling** to identify bottlenecks
6. **Scaling strategies** for large vocabularies and models

The key is balancing accuracy, capacity, and computational efficiency for your specific application requirements.