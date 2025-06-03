# HRR Performance Guide

This guide provides comprehensive information on optimizing HRR performance, including benchmarks, profiling techniques, and best practices for production deployments.

## Table of Contents

1. [Performance Characteristics](#performance-characteristics)
2. [Benchmarks](#benchmarks)
3. [Optimization Strategies](#optimization-strategies)
4. [Memory Management](#memory-management)
5. [Computational Complexity](#computational-complexity)
6. [Hardware Considerations](#hardware-considerations)
7. [Profiling and Monitoring](#profiling-and-monitoring)
8. [Production Best Practices](#production-best-practices)

## Performance Characteristics

### Core Operations Performance

| Operation | Time Complexity | Space Complexity | Typical Time (n=1024) |
|-----------|----------------|------------------|----------------------|
| Binding (FFT) | O(n log n) | O(n) | ~50 μs |
| Binding (Direct) | O(n²) | O(n) | ~2 ms |
| Unbinding (FFT) | O(n log n) | O(n) | ~50 μs |
| Bundling | O(k·n) | O(n) | ~10 μs per vector |
| Similarity | O(n) | O(1) | ~5 μs |
| Normalization | O(n) | O(1) | ~3 μs |

### Dimension vs Performance

```python
import time
import numpy as np
from cognitive_computing.hrr import create_hrr

def benchmark_dimensions():
    """Benchmark HRR operations across dimensions."""
    dimensions = [256, 512, 1024, 2048, 4096, 8192]
    results = {}
    
    for dim in dimensions:
        hrr = create_hrr(dimension=dim)
        a = hrr.generate_vector()
        b = hrr.generate_vector()
        
        # Benchmark binding
        start = time.perf_counter()
        for _ in range(1000):
            _ = hrr.bind(a, b)
        bind_time = (time.perf_counter() - start) / 1000
        
        # Benchmark similarity
        start = time.perf_counter()
        for _ in range(10000):
            _ = hrr.similarity(a, b)
        sim_time = (time.perf_counter() - start) / 10000
        
        results[dim] = {
            'bind_time_ms': bind_time * 1000,
            'similarity_time_us': sim_time * 1_000_000
        }
    
    return results
```

## Benchmarks

### Binding Capacity vs Dimension

```python
def benchmark_capacity():
    """Benchmark binding capacity."""
    dimensions = [512, 1024, 2048, 4096]
    max_pairs = [5, 10, 20, 30, 40, 50]
    
    results = {}
    for dim in dimensions:
        hrr = create_hrr(dimension=dim)
        dim_results = []
        
        for n_pairs in max_pairs:
            # Create role-filler pairs
            pairs = []
            for i in range(n_pairs):
                role = hrr.generate_vector(method="unitary")
                filler = hrr.generate_vector()
                pairs.append((role, filler))
            
            # Bundle all bindings
            bindings = [hrr.bind(r, f) for r, f in pairs]
            composite = hrr.bundle(bindings)
            
            # Test retrieval accuracy
            correct = 0
            for role, filler in pairs:
                retrieved = hrr.unbind(composite, role)
                sim = hrr.similarity(retrieved, filler)
                if sim > 0.3:  # Threshold
                    correct += 1
            
            accuracy = correct / n_pairs
            dim_results.append(accuracy)
        
        results[dim] = dim_results
    
    return results
```

### Real vs Complex Storage

```python
def compare_storage_methods():
    """Compare real vs complex storage performance."""
    dimension = 2048
    n_operations = 10000
    
    # Real storage
    hrr_real = create_hrr(dimension=dimension, storage_method="real")
    a_real = hrr_real.generate_vector()
    b_real = hrr_real.generate_vector()
    
    start = time.perf_counter()
    for _ in range(n_operations):
        _ = hrr_real.bind(a_real, b_real)
    real_time = time.perf_counter() - start
    
    # Complex storage
    hrr_complex = create_hrr(dimension=dimension, storage_method="complex")
    a_complex = hrr_complex.generate_vector()
    b_complex = hrr_complex.generate_vector()
    
    start = time.perf_counter()
    for _ in range(n_operations):
        _ = hrr_complex.bind(a_complex, b_complex)
    complex_time = time.perf_counter() - start
    
    print(f"Real storage: {real_time:.3f}s")
    print(f"Complex storage: {complex_time:.3f}s")
    print(f"Complex is {real_time/complex_time:.2f}x faster")
```

### Cleanup Memory Performance

```python
def benchmark_cleanup_memory():
    """Benchmark cleanup memory with varying vocabulary sizes."""
    from cognitive_computing.hrr import CleanupMemory, CleanupMemoryConfig
    
    dimension = 1024
    vocab_sizes = [100, 1000, 10000, 50000]
    hrr = create_hrr(dimension=dimension)
    
    results = {}
    for vocab_size in vocab_sizes:
        # Create cleanup memory
        cleanup = CleanupMemory(CleanupMemoryConfig(), dimension)
        
        # Add vocabulary
        for i in range(vocab_size):
            vec = hrr.generate_vector()
            cleanup.add_item(f"item_{i}", vec)
        
        # Benchmark cleanup
        test_vec = hrr.generate_vector()
        
        start = time.perf_counter()
        for _ in range(1000):
            _ = cleanup.cleanup(test_vec)
        cleanup_time = (time.perf_counter() - start) / 1000
        
        # Benchmark k-NN
        start = time.perf_counter()
        for _ in range(100):
            _ = cleanup.find_closest(test_vec, k=5)
        knn_time = (time.perf_counter() - start) / 100
        
        results[vocab_size] = {
            'cleanup_ms': cleanup_time * 1000,
            'knn_ms': knn_time * 1000
        }
    
    return results
```

## Optimization Strategies

### 1. FFT Optimization

```python
# Use FFT for dimensions > 64
def optimized_bind(hrr, a, b):
    """Optimized binding with automatic method selection."""
    if hrr.dimension <= 64:
        # Direct convolution for small dimensions
        return CircularConvolution.convolve(a, b, method="direct")
    else:
        # FFT for larger dimensions
        return CircularConvolution.convolve(a, b, method="fft")
```

### 2. Vector Caching

```python
class CachedHRR:
    """HRR with vector caching for repeated operations."""
    
    def __init__(self, hrr):
        self.hrr = hrr
        self.vector_cache = {}
        self.binding_cache = {}
        self.max_cache_size = 10000
    
    def get_or_create_vector(self, name):
        """Get vector from cache or create new."""
        if name not in self.vector_cache:
            if len(self.vector_cache) >= self.max_cache_size:
                # Evict oldest (simple FIFO)
                oldest = next(iter(self.vector_cache))
                del self.vector_cache[oldest]
            
            self.vector_cache[name] = self.hrr.generate_vector()
        
        return self.vector_cache[name]
    
    def cached_bind(self, a_name, b_name):
        """Cached binding operation."""
        cache_key = (a_name, b_name)
        
        if cache_key not in self.binding_cache:
            a = self.get_or_create_vector(a_name)
            b = self.get_or_create_vector(b_name)
            self.binding_cache[cache_key] = self.hrr.bind(a, b)
        
        return self.binding_cache[cache_key]
```

### 3. Batch Operations

```python
def batch_bind(hrr, pairs):
    """Efficiently bind multiple pairs."""
    # Pre-allocate output
    n_pairs = len(pairs)
    dimension = hrr.dimension
    results = np.zeros((n_pairs, dimension))
    
    # Use vectorized operations where possible
    for i, (a, b) in enumerate(pairs):
        results[i] = hrr.bind(a, b)
    
    return results

def batch_similarity(hrr, vectors_a, vectors_b):
    """Compute similarities in batch."""
    # Normalize if needed
    if hrr.normalize:
        norms_a = np.linalg.norm(vectors_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(vectors_b, axis=1, keepdims=True)
        vectors_a = vectors_a / norms_a
        vectors_b = vectors_b / norms_b
    
    # Batch dot product
    similarities = np.sum(vectors_a * vectors_b, axis=1)
    return similarities
```

### 4. Memory Pool

```python
class VectorPool:
    """Pre-allocated vector pool for memory efficiency."""
    
    def __init__(self, dimension, pool_size=1000):
        self.dimension = dimension
        self.pool = np.zeros((pool_size, dimension))
        self.available = list(range(pool_size))
        self.in_use = set()
    
    def acquire(self):
        """Get a vector from the pool."""
        if not self.available:
            raise RuntimeError("Vector pool exhausted")
        
        idx = self.available.pop()
        self.in_use.add(idx)
        
        # Initialize with random values
        self.pool[idx] = np.random.randn(self.dimension) / np.sqrt(self.dimension)
        return self.pool[idx]
    
    def release(self, vector):
        """Return vector to pool."""
        # Find index (in practice, track this better)
        for idx in self.in_use:
            if np.array_equal(self.pool[idx], vector):
                self.in_use.remove(idx)
                self.available.append(idx)
                break
```

## Memory Management

### Memory Usage Estimation

```python
def estimate_memory_usage(dimension, n_vectors, storage_method="real"):
    """Estimate memory usage for HRR system."""
    # Bytes per element
    if storage_method == "real":
        bytes_per_element = 8  # float64
    else:  # complex
        bytes_per_element = 16  # complex128
    
    # Vector storage
    vector_memory = n_vectors * dimension * bytes_per_element
    
    # FFT workspace (approximate)
    fft_memory = 2 * dimension * bytes_per_element
    
    # Overhead (approximately 20%)
    overhead = 0.2 * vector_memory
    
    total_bytes = vector_memory + fft_memory + overhead
    
    return {
        'vectors_mb': vector_memory / (1024 * 1024),
        'fft_mb': fft_memory / (1024 * 1024),
        'overhead_mb': overhead / (1024 * 1024),
        'total_mb': total_bytes / (1024 * 1024)
    }
```

### Memory-Efficient Cleanup

```python
class StreamingCleanup:
    """Memory-efficient cleanup for large vocabularies."""
    
    def __init__(self, dimension, chunk_size=1000):
        self.dimension = dimension
        self.chunk_size = chunk_size
        self.chunks = []
        self.current_chunk = {
            'names': [],
            'vectors': np.empty((0, dimension))
        }
    
    def add_item(self, name, vector):
        """Add item with automatic chunking."""
        if len(self.current_chunk['names']) >= self.chunk_size:
            # Save current chunk
            self.chunks.append(self.current_chunk)
            self.current_chunk = {
                'names': [],
                'vectors': np.empty((0, self.dimension))
            }
        
        self.current_chunk['names'].append(name)
        self.current_chunk['vectors'] = np.vstack([
            self.current_chunk['vectors'],
            vector.reshape(1, -1)
        ])
    
    def cleanup(self, vector):
        """Clean up using chunked search."""
        best_name = None
        best_similarity = -1
        
        # Search all chunks
        for chunk in self.chunks + [self.current_chunk]:
            if len(chunk['names']) == 0:
                continue
            
            # Compute similarities
            similarities = np.dot(chunk['vectors'], vector)
            max_idx = np.argmax(similarities)
            
            if similarities[max_idx] > best_similarity:
                best_similarity = similarities[max_idx]
                best_name = chunk['names'][max_idx]
        
        return best_name, best_similarity
```

## Computational Complexity

### Scaling Analysis

| Operation | n=256 | n=512 | n=1024 | n=2048 | n=4096 |
|-----------|-------|-------|--------|--------|--------|
| Bind (ms) | 0.02 | 0.04 | 0.09 | 0.20 | 0.45 |
| Unbind (ms) | 0.02 | 0.04 | 0.09 | 0.20 | 0.45 |
| Bundle 10 items (ms) | 0.05 | 0.10 | 0.20 | 0.40 | 0.80 |
| Cleanup 1k vocab (ms) | 0.25 | 0.50 | 1.00 | 2.00 | 4.00 |
| Memory (MB) | 0.5 | 2 | 8 | 32 | 128 |

### Optimization Guidelines

1. **Dimension Selection**
   - Use smallest dimension that meets accuracy requirements
   - Sweet spot often 1024-2048 for most applications
   - Consider 4096+ only for complex hierarchical structures

2. **Storage Method**
   - Real: Lower memory, standard performance
   - Complex: 2x memory, potentially faster operations

3. **Batch Size**
   - Batch operations when possible
   - Optimal batch size depends on cache size
   - Typically 100-1000 items per batch

## Hardware Considerations

### CPU Optimization

```python
# Enable multi-threading for NumPy
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

# Use optimized BLAS
# Install: pip install intel-scipy
```

### GPU Acceleration (Future)

```python
# Placeholder for GPU support
class GPUAcceleratedHRR:
    """GPU-accelerated HRR operations."""
    
    def __init__(self, dimension, device='cuda'):
        self.dimension = dimension
        self.device = device
        # Initialize CUDA/OpenCL context
    
    def bind_gpu(self, a, b):
        """GPU-accelerated binding."""
        # Transfer to GPU
        # Perform FFT on GPU
        # Transfer back
        pass
```

### Memory Hierarchy

```python
def optimize_for_cache(dimension):
    """Adjust operations for CPU cache."""
    L1_cache = 32 * 1024  # 32KB typical L1
    L2_cache = 256 * 1024  # 256KB typical L2
    L3_cache = 8 * 1024 * 1024  # 8MB typical L3
    
    vector_bytes = dimension * 8  # float64
    
    if vector_bytes <= L1_cache:
        return "L1_optimized"
    elif vector_bytes <= L2_cache:
        return "L2_optimized"
    elif vector_bytes <= L3_cache:
        return "L3_optimized"
    else:
        return "memory_bound"
```

## Profiling and Monitoring

### Performance Profiling

```python
import cProfile
import pstats
from io import StringIO

def profile_hrr_operations():
    """Profile HRR operations."""
    hrr = create_hrr(dimension=2048)
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Profile operations
    profiler.enable()
    
    # Simulate workload
    for _ in range(1000):
        a = hrr.generate_vector()
        b = hrr.generate_vector()
        c = hrr.bind(a, b)
        _ = hrr.unbind(c, a)
    
    profiler.disable()
    
    # Get results
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(10)  # Top 10 functions
    
    return s.getvalue()
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_operations():
    """Track memory usage."""
    hrr = create_hrr(dimension=4096)
    
    # Create many vectors
    vectors = []
    for i in range(1000):
        vectors.append(hrr.generate_vector())
    
    # Bundle operations
    result = hrr.bundle(vectors[:100])
    
    # Cleanup
    cleanup = CleanupMemory(CleanupMemoryConfig(), 4096)
    for i, vec in enumerate(vectors[:500]):
        cleanup.add_item(f"item_{i}", vec)
    
    return result
```

### Custom Metrics

```python
class PerformanceMonitor:
    """Monitor HRR performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'bind_times': [],
            'unbind_times': [],
            'bundle_times': [],
            'cleanup_times': []
        }
    
    def time_operation(self, operation, *args, **kwargs):
        """Time an operation."""
        start = time.perf_counter()
        result = operation(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        op_name = operation.__name__
        if op_name in self.metrics:
            self.metrics[op_name].append(elapsed)
        
        return result
    
    def get_statistics(self):
        """Get performance statistics."""
        stats = {}
        for op, times in self.metrics.items():
            if times:
                stats[op] = {
                    'mean_ms': np.mean(times) * 1000,
                    'std_ms': np.std(times) * 1000,
                    'min_ms': np.min(times) * 1000,
                    'max_ms': np.max(times) * 1000,
                    'count': len(times)
                }
        return stats
```

## Production Best Practices

### 1. Configuration Tuning

```python
def production_config(use_case="general"):
    """Get optimized configuration for production."""
    configs = {
        "general": {
            "dimension": 1024,
            "normalize": True,
            "cleanup_threshold": 0.3,
            "storage_method": "real"
        },
        "high_capacity": {
            "dimension": 4096,
            "normalize": True,
            "cleanup_threshold": 0.25,
            "storage_method": "complex"
        },
        "low_latency": {
            "dimension": 512,
            "normalize": False,
            "cleanup_threshold": 0.35,
            "storage_method": "real"
        },
        "memory_constrained": {
            "dimension": 256,
            "normalize": True,
            "cleanup_threshold": 0.4,
            "storage_method": "real"
        }
    }
    
    return HRRConfig(**configs.get(use_case, configs["general"]))
```

### 2. Error Handling

```python
class RobustHRR:
    """Production-ready HRR with error handling."""
    
    def __init__(self, config):
        self.hrr = HRR(config)
        self.error_count = 0
        self.max_retries = 3
    
    def safe_bind(self, a, b):
        """Binding with error handling."""
        for attempt in range(self.max_retries):
            try:
                # Validate inputs
                if not self._validate_vector(a) or not self._validate_vector(b):
                    raise ValueError("Invalid vector input")
                
                # Perform binding
                result = self.hrr.bind(a, b)
                
                # Validate output
                if not self._validate_vector(result):
                    raise RuntimeError("Invalid binding result")
                
                return result
                
            except Exception as e:
                self.error_count += 1
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Binding failed after {self.max_retries} attempts: {e}")
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
    
    def _validate_vector(self, vector):
        """Validate vector integrity."""
        if vector is None or len(vector) != self.hrr.dimension:
            return False
        if np.isnan(vector).any() or np.isinf(vector).any():
            return False
        return True
```

### 3. Monitoring and Alerting

```python
class HRRHealthCheck:
    """Health monitoring for HRR system."""
    
    def __init__(self, hrr, thresholds=None):
        self.hrr = hrr
        self.thresholds = thresholds or {
            'bind_time_ms': 1.0,
            'memory_usage_mb': 1000,
            'error_rate': 0.01
        }
        self.metrics = {
            'operations': 0,
            'errors': 0,
            'total_time': 0
        }
    
    def check_health(self):
        """Perform health check."""
        health_status = {
            'healthy': True,
            'issues': []
        }
        
        # Check operation time
        test_vec = self.hrr.generate_vector()
        start = time.perf_counter()
        _ = self.hrr.bind(test_vec, test_vec)
        bind_time = (time.perf_counter() - start) * 1000
        
        if bind_time > self.thresholds['bind_time_ms']:
            health_status['healthy'] = False
            health_status['issues'].append(
                f"Slow binding: {bind_time:.2f}ms"
            )
        
        # Check error rate
        if self.metrics['operations'] > 0:
            error_rate = self.metrics['errors'] / self.metrics['operations']
            if error_rate > self.thresholds['error_rate']:
                health_status['healthy'] = False
                health_status['issues'].append(
                    f"High error rate: {error_rate:.2%}"
                )
        
        return health_status
```

### 4. Deployment Considerations

```python
# Environment variables for production
PRODUCTION_CONFIG = {
    'HRR_DIMENSION': int(os.getenv('HRR_DIMENSION', '1024')),
    'HRR_NORMALIZE': os.getenv('HRR_NORMALIZE', 'true').lower() == 'true',
    'HRR_STORAGE': os.getenv('HRR_STORAGE', 'real'),
    'HRR_CACHE_SIZE': int(os.getenv('HRR_CACHE_SIZE', '10000')),
    'HRR_THREAD_POOL': int(os.getenv('HRR_THREAD_POOL', '4'))
}

# Production initialization
def create_production_hrr():
    """Create HRR instance for production."""
    config = HRRConfig(
        dimension=PRODUCTION_CONFIG['HRR_DIMENSION'],
        normalize=PRODUCTION_CONFIG['HRR_NORMALIZE'],
        storage_method=PRODUCTION_CONFIG['HRR_STORAGE']
    )
    
    hrr = HRR(config)
    
    # Warm up FFT cache
    warmup_vec = hrr.generate_vector()
    for _ in range(10):
        _ = hrr.bind(warmup_vec, warmup_vec)
    
    return hrr
```

## Optimization Checklist

- [ ] Choose appropriate dimension for use case
- [ ] Enable FFT for dimensions > 64
- [ ] Use unitary vectors for roles/relations
- [ ] Batch operations when possible
- [ ] Implement vector caching for repeated operations
- [ ] Monitor memory usage and implement cleanup
- [ ] Profile performance bottlenecks
- [ ] Configure NumPy/BLAS for multi-threading
- [ ] Implement health checks and monitoring
- [ ] Set up proper error handling and retries
- [ ] Warm up caches before production use
- [ ] Document expected performance characteristics

## Conclusion

Optimizing HRR performance requires attention to:
1. **Algorithm selection** - FFT vs direct convolution
2. **Memory management** - Caching and pooling strategies
3. **Batch processing** - Vectorized operations
4. **Hardware utilization** - CPU cache and threading
5. **Production readiness** - Monitoring and error handling

Key performance tips:
- Start with dimension 1024 for most applications
- Use batch operations for multiple items
- Implement caching for repeated operations
- Monitor performance metrics in production
- Plan for graceful degradation under load