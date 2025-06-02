# SDM API Reference

This document provides a complete API reference for the Sparse Distributed Memory (SDM) module.

## Table of Contents

- [Core Classes](#core-classes)
  - [SDM](#sdm)
  - [SDMConfig](#sdmconfig)
- [Memory Components](#memory-components)
  - [HardLocation](#hardlocation)
  - [MemoryContents](#memorycontents)
  - [MemoryStatistics](#memorystatistics)
  - [MemoryOptimizer](#memoryoptimizer)
- [Address Decoders](#address-decoders)
  - [AddressDecoder](#addressdecoder)
  - [HammingDecoder](#hammingdecoder)
  - [JaccardDecoder](#jaccardecoder)
  - [RandomDecoder](#randomdecoder)
  - [AdaptiveDecoder](#adaptivedecoder)
  - [HierarchicalDecoder](#hierarchicaldecoder)
  - [LSHDecoder](#lshdecoder)
- [Utility Functions](#utility-functions)
- [Visualization Functions](#visualization-functions)
- [Quick Reference Functions](#quick-reference-functions)

---

## Core Classes

### SDM

```python
class SDM(config: SDMConfig)
```

Main Sparse Distributed Memory implementation.

#### Parameters
- **config** : `SDMConfig`
  - Configuration object containing SDM parameters

#### Attributes
- **hard_locations** : `np.ndarray`
  - Array of hard location addresses, shape (num_hard_locations, dimension)
- **counters** : `np.ndarray` or `None`
  - Counter array for storage method 'counters'
- **binary_storage** : `np.ndarray` or `None`
  - Binary storage for method 'binary'
- **location_usage** : `np.ndarray`
  - Count of how many times each location has been activated
- **metrics** : `MemoryPerformanceMetrics`
  - Performance tracking metrics

#### Methods

##### store
```python
store(address: np.ndarray, data: np.ndarray) -> None
```
Store a data pattern at the given address.

**Parameters:**
- **address** : `np.ndarray` - Address vector of shape (dimension,)
- **data** : `np.ndarray` - Data vector of shape (dimension,)

**Example:**
```python
address = np.random.randint(0, 2, 1000)
data = np.random.randint(0, 2, 1000)
sdm.store(address, data)
```

##### recall
```python
recall(address: np.ndarray) -> Optional[np.ndarray]
```
Recall data from the given address.

**Parameters:**
- **address** : `np.ndarray` - Address vector of shape (dimension,)

**Returns:**
- `np.ndarray` or `None` - Recalled data vector or None if no data found

**Example:**
```python
recalled_data = sdm.recall(address)
if recalled_data is not None:
    accuracy = np.mean(recalled_data == original_data)
```

##### recall_with_confidence
```python
recall_with_confidence(address: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]
```
Recall data with confidence scores for each bit.

**Returns:**
- **data** : `np.ndarray` or `None` - Recalled data vector
- **confidence** : `np.ndarray` - Confidence scores (0 to 1) for each bit

##### clear
```python
clear() -> None
```
Clear all stored data from memory.

##### get_memory_stats
```python
get_memory_stats() -> Dict[str, float]
```
Get statistics about memory usage and distribution.

**Returns:**
- Dictionary containing:
  - `num_patterns_stored` : Number of stored patterns
  - `locations_used` : Number of locations with non-zero usage
  - `avg_location_usage` : Average usage count
  - `max_location_usage` : Maximum usage count
  - And more...

##### analyze_crosstalk
```python
analyze_crosstalk(num_samples: int = 100) -> Dict[str, float]
```
Analyze interference between stored patterns.

**Parameters:**
- **num_samples** : `int` - Number of pattern pairs to sample

**Returns:**
- Dictionary with crosstalk analysis results

#### Properties

##### size
```python
@property
size() -> int
```
Return the number of stored patterns. This includes all patterns passed to the `store()` method, even those that failed to activate any locations.

---

### SDMConfig

```python
@dataclass
class SDMConfig(MemoryConfig)
```

Configuration for Sparse Distributed Memory.

#### Parameters
- **dimension** : `int` - Dimensionality of address and data vectors
- **num_hard_locations** : `int` - Number of hard memory locations
- **activation_radius** : `int` - Hamming radius for location activation
- **threshold** : `float` = 0.0 - Threshold for reading from counters
- **storage_method** : `str` = "counters" - Method: 'counters' or 'binary'
- **parallel** : `bool` = False - Whether to use parallel processing
- **num_workers** : `int` = 4 - Number of worker threads
- **counter_bits** : `int` = 8 - Number of bits per counter
- **saturation_value** : `int` = 127 - Maximum absolute counter value

#### Computed Attributes
- **capacity** : `int` - Theoretical capacity estimate
- **critical_distance** : `int` - Critical distance for the dimension

**Example:**
```python
config = SDMConfig(
    dimension=1000,
    num_hard_locations=1000,
    activation_radius=451,
    storage_method="counters",
    parallel=True
)
```

---

## Memory Components

### HardLocation

```python
@dataclass
class HardLocation
```

Represents a single hard location in SDM.

#### Parameters
- **index** : `int` - Index in the SDM
- **address** : `np.ndarray` - Binary address vector
- **dimension** : `int` - Dimensionality
- **storage_type** : `str` - 'counters' or 'binary'

#### Methods

##### write
```python
write(data: np.ndarray, timestamp: int = 0) -> None
```
Write data to this location.

##### read
```python
read(timestamp: int = 0) -> np.ndarray
```
Read data from this location.

##### get_saturation_level
```python
get_saturation_level(max_value: int = 127) -> float
```
Calculate saturation level for counter-based storage.

##### get_entropy
```python
get_entropy() -> float
```
Calculate Shannon entropy of stored data.

---

### MemoryContents

```python
class MemoryContents(sdm: SDM)
```

Analyzer for SDM memory contents.

#### Methods

##### get_memory_map
```python
get_memory_map() -> Dict[str, np.ndarray]
```
Generate memory maps showing usage patterns.

**Returns:**
- Dictionary containing:
  - `usage_map` : Location usage frequencies
  - `entropy_map` : Information entropy per location
  - `saturation_map` : Saturation levels (counters)
  - `density_map` : Bit densities (binary)

##### analyze_pattern_distribution
```python
analyze_pattern_distribution(sample_size: int = 1000) -> Dict[str, float]
```
Analyze how patterns are distributed across memory.

##### find_similar_locations
```python
find_similar_locations(threshold: float = 0.8) -> List[Tuple[int, int, float]]
```
Find pairs of locations with similar contents.

##### get_capacity_estimate
```python
get_capacity_estimate() -> Dict[str, float]
```
Estimate current and maximum capacity.

---

### MemoryStatistics

```python
class MemoryStatistics(sdm: SDM)
```

Advanced statistical analysis for SDM.

#### Methods

##### analyze_temporal_patterns
```python
analyze_temporal_patterns(window_size: int = 100) -> Dict[str, np.ndarray]
```
Analyze temporal patterns in memory usage.

##### compute_correlation_matrix
```python
compute_correlation_matrix(sample_size: int = 100) -> np.ndarray
```
Compute correlation matrix between memory locations.

##### analyze_recall_quality
```python
analyze_recall_quality(test_size: int = 100, 
                      noise_levels: List[float] = None) -> Dict[str, List[float]]
```
Analyze recall quality under different noise conditions.

##### generate_report
```python
generate_report() -> Dict[str, any]
```
Generate comprehensive statistical report.

##### plot_analysis
```python
plot_analysis(figsize: Tuple[int, int] = (15, 10)) -> plt.Figure
```
Generate visualization plots for memory analysis.

---

### MemoryOptimizer

```python
class MemoryOptimizer
```

Static utilities for optimizing SDM parameters.

#### Static Methods

##### find_optimal_radius
```python
@staticmethod
find_optimal_radius(dimension: int, num_locations: int, 
                   target_activation: int = None) -> int
```
Find optimal activation radius for given parameters.

##### estimate_required_locations
```python
@staticmethod
estimate_required_locations(dimension: int, capacity: int, 
                          activation_radius: int = None) -> int
```
Estimate number of hard locations needed for desired capacity.

##### analyze_parameter_space
```python
@staticmethod
analyze_parameter_space(dimension_range: Tuple[int, int],
                       location_range: Tuple[int, int],
                       samples: int = 10) -> List[Dict]
```
Analyze SDM performance across parameter space.

---

## Address Decoders

### AddressDecoder

```python
class AddressDecoder(ABC)
```

Abstract base class for address decoders.

#### Abstract Methods

##### decode
```python
@abstractmethod
decode(address: np.ndarray) -> np.ndarray
```
Decode an address to activated location indices.

##### expected_activations
```python
@abstractmethod
expected_activations() -> float
```
Return expected number of activations per address.

#### Common Methods

##### decode_batch
```python
decode_batch(addresses: np.ndarray) -> List[np.ndarray]
```
Decode multiple addresses in batch.

##### get_activation_stats
```python
get_activation_stats(address: np.ndarray) -> Dict[str, float]
```
Get statistics about activation pattern.

---

### HammingDecoder

```python
class HammingDecoder(AddressDecoder)
```

Classic Hamming distance-based decoder.

#### Parameters
- **config** : `DecoderConfig`
- **hard_locations** : `np.ndarray`
- **use_fast_hamming** : `bool` = True

#### Methods

##### get_activation_distribution
```python
get_activation_distribution(num_samples: int = 1000) -> Dict[str, np.ndarray]
```
Analyze activation distribution through sampling.

---

### JaccardDecoder

```python
class JaccardDecoder(AddressDecoder)
```

Jaccard similarity-based decoder for sparse data.

#### Parameters
- **config** : `DecoderConfig` - The `activation_radius` is interpreted as similarity threshold × 1000. For example, `activation_radius=200` means a Jaccard similarity threshold of 0.2
- **hard_locations** : `np.ndarray`
- **min_similarity** : `float` = None - If provided, overrides the threshold calculated from activation_radius

#### Usage Note
For sparse binary data (e.g., 30% density), typical Jaccard similarities between random patterns are around 0.15-0.25. Therefore, use activation_radius values like 200-300 for reasonable activation rates.

---

### RandomDecoder

```python
class RandomDecoder(AddressDecoder)
```

Random hash-based decoder with O(1) complexity.

#### Parameters
- **config** : `DecoderConfig`
- **hard_locations** : `np.ndarray`
- **num_hashes** : `int` = None

---

### AdaptiveDecoder

```python
class AdaptiveDecoder(AddressDecoder)
```

Self-adjusting decoder that adapts to memory state.

#### Parameters
- **config** : `DecoderConfig`
- **hard_locations** : `np.ndarray`
- **target_activations** : `int` = None
- **adaptation_rate** : `float` = 0.1

#### Methods

##### adapt_radii
```python
adapt_radii() -> None
```
Globally adapt radii based on activation history.

##### get_adaptation_stats
```python
get_adaptation_stats() -> Dict[str, float]
```
Get statistics about adaptation behavior.

---

### HierarchicalDecoder

```python
class HierarchicalDecoder(AddressDecoder)
```

Multi-level hierarchical decoder.

#### Parameters
- **config** : `DecoderConfig`
- **hard_locations** : `np.ndarray`
- **num_levels** : `int` = 3
- **branching_factor** : `int` = 4

#### Methods

##### visualize_hierarchy
```python
visualize_hierarchy() -> Dict[str, np.ndarray]
```
Get hierarchy structure for visualization.

---

### LSHDecoder

```python
class LSHDecoder(AddressDecoder)
```

Locality-Sensitive Hashing based decoder.

#### Parameters
- **config** : `DecoderConfig`
- **hard_locations** : `np.ndarray`
- **num_tables** : `int` = 10
- **hash_size** : `int` = 8

#### Methods

##### get_hash_statistics
```python
get_hash_statistics() -> Dict[str, float]
```
Get statistics about hash table distribution.

---

### create_decoder

```python
create_decoder(decoder_type: str, config: DecoderConfig, 
              hard_locations: np.ndarray, **kwargs) -> AddressDecoder
```

Factory function to create address decoders.

**Parameters:**
- **decoder_type** : `str` - Type: 'hamming', 'jaccard', 'random', 'adaptive', 'hierarchical', or 'lsh'
- **config** : `DecoderConfig`
- **hard_locations** : `np.ndarray`
- **kwargs** : Additional decoder-specific parameters

**Example:**
```python
decoder = create_decoder('adaptive', config, hard_locations, 
                        target_activations=100)
```

---

## Utility Functions

### Pattern Generation and Noise

#### add_noise
```python
add_noise(pattern: np.ndarray, noise_level: float, 
         noise_type: str = 'flip', seed: Optional[int] = None) -> np.ndarray
```
Add noise to a binary pattern.

**Parameters:**
- **pattern** : `np.ndarray` - Binary pattern
- **noise_level** : `float` - Amount of noise (0 to 1)
- **noise_type** : `str` - Type: 'flip', 'swap', 'burst', 'salt_pepper'
- **seed** : `int` - Random seed

**Example:**
```python
noisy_pattern = add_noise(pattern, 0.1, 'flip')
```

#### generate_random_patterns
```python
generate_random_patterns(num_patterns: int, dimension: int,
                       sparsity: float = 0.5, 
                       correlation: float = 0.0,
                       seed: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]
```
Generate random binary patterns for testing.

**Parameters:**
- **correlation** : float - Controls correlation between address and data patterns. With correlation=c, the expected bit matching is: c + (1-c) × 0.5. For example, correlation=0.7 results in ~85% bit matching.

**Returns:**
- **addresses** : List of address patterns
- **data** : List of data patterns

#### create_orthogonal_patterns
```python
create_orthogonal_patterns(num_patterns: int, dimension: int,
                         min_distance: Optional[int] = None) -> List[np.ndarray]
```
Create approximately orthogonal binary patterns.

### Capacity and Analysis

#### compute_memory_capacity
```python
compute_memory_capacity(dimension: int, num_locations: int,
                      activation_radius: int, 
                      error_tolerance: float = 0.01) -> Dict[str, float]
```
Compute theoretical capacity of SDM configuration.

**Returns:**
- Dictionary containing:
  - `kanerva_estimate` : Kanerva's capacity formula
  - `information_theoretic` : Information theory bound
  - `sphere_packing` : Sphere packing bound
  - `coverage_based` : Coverage-based estimate

#### analyze_activation_patterns
```python
analyze_activation_patterns(sdm: SDM, sample_size: int = 1000,
                          visualize: bool = False) -> Dict[str, any]
```
Analyze activation patterns in an SDM instance.

### Performance Testing

#### test_sdm_performance
```python
test_sdm_performance(sdm: SDM, test_patterns: int = 100,
                    noise_levels: List[float] = None,
                    progress: bool = True) -> PerformanceTestResult
```
Comprehensive performance test for SDM.

**Returns:**
- `PerformanceTestResult` dataclass with:
  - Write/read time statistics
  - Recall accuracy
  - Noise tolerance
  - Capacity utilization

### Pattern Encoding

#### PatternEncoder
```python
class PatternEncoder(dimension: int)
```

Encode various data types into binary patterns.

##### encode_integer
```python
encode_integer(value: int, bits: Optional[int] = None) -> np.ndarray
```
Encode integer to binary pattern.

##### encode_float
```python
encode_float(value: float, precision: int = 16) -> np.ndarray
```
Encode float to binary pattern.

##### encode_string
```python
encode_string(text: str, method: str = 'hash') -> np.ndarray
```
Encode string to binary pattern.

**Parameters:**
- **method** : `str` - Encoding method: 'hash', 'char'

##### encode_vector
```python
encode_vector(vector: np.ndarray, method: str = 'threshold') -> np.ndarray
```
Encode continuous vector to binary pattern.

**Parameters:**
- **method** : `str` - Method: 'threshold', 'rank', 'random_projection'

### Persistence

#### save_sdm_state
```python
save_sdm_state(sdm: SDM, filepath: str, include_patterns: bool = True) -> None
```
Save SDM state to file.

#### load_sdm_state
```python
load_sdm_state(filepath: str, sdm_class=None) -> SDM
```
Load SDM state from file.

**Example:**
```python
# Save
save_sdm_state(sdm, 'my_sdm.pkl')

# Load
loaded_sdm = load_sdm_state('my_sdm.pkl')
```

### Similarity Metrics

#### calculate_pattern_similarity
```python
calculate_pattern_similarity(pattern1: np.ndarray, pattern2: np.ndarray,
                           metric: str = 'hamming') -> float
```
Calculate similarity between two binary patterns.

**Parameters:**
- **metric** : `str` - Metric: 'hamming', 'jaccard', 'cosine', 'mutual_info'

---

## Visualization Functions

### plot_memory_distribution
```python
plot_memory_distribution(sdm: SDM, figsize: Tuple[int, int] = (15, 10),
                       save_path: Optional[str] = None) -> plt.Figure
```
Plot comprehensive memory distribution analysis.

Creates multi-panel figure showing:
- Location usage distribution
- Usage heatmap
- Saturation/density distribution
- Address space coverage
- Memory statistics
- Activation overlap matrix

### plot_activation_pattern
```python
plot_activation_pattern(sdm: SDM, address: np.ndarray, 
                      comparison_addresses: Optional[List[np.ndarray]] = None,
                      figsize: Tuple[int, int] = (12, 8),
                      save_path: Optional[str] = None) -> plt.Figure
```
Visualize activation pattern for a given address.

### plot_recall_accuracy
```python
plot_recall_accuracy(test_results: Union[Dict, List[Dict]], 
                    figsize: Tuple[int, int] = (12, 8),
                    save_path: Optional[str] = None) -> plt.Figure
```
Plot recall accuracy under various conditions.

### visualize_memory_contents
```python
visualize_memory_contents(sdm: SDM, num_samples: int = 100,
                        method: str = 'tsne',
                        color_by: str = 'usage',
                        interactive: bool = False,
                        figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[str] = None) -> Union[plt.Figure, go.Figure]
```
Visualize memory contents using dimensionality reduction.

**Parameters:**
- **method** : `str` - Method: 'pca', 'tsne', 'mds'
- **color_by** : `str` - Color scheme: 'usage', 'saturation', 'cluster'
- **interactive** : `bool` - Create interactive plotly visualization

### plot_decoder_comparison
```python
plot_decoder_comparison(sdm_instances: Dict[str, Any],
                       test_size: int = 100,
                       figsize: Tuple[int, int] = (15, 10),
                       save_path: Optional[str] = None) -> plt.Figure
```
Compare different decoder strategies.

### create_recall_animation
```python
create_recall_animation(sdm: SDM, address: np.ndarray, 
                       noise_levels: List[float] = None,
                       interval: int = 500,
                       save_path: Optional[str] = None) -> FuncAnimation
```
Create animation showing recall process with increasing noise.

### plot_theoretical_analysis
```python
plot_theoretical_analysis(dimension_range: Tuple[int, int] = (100, 2000),
                        num_points: int = 20,
                        figsize: Tuple[int, int] = (15, 10),
                        save_path: Optional[str] = None) -> plt.Figure
```
Plot theoretical SDM properties across dimensions.

---

## Quick Reference Functions

### create_sdm
```python
create_sdm(dimension: int, num_locations: int = None, 
          activation_radius: int = None) -> SDM
```
Quick function to create an SDM instance with sensible defaults.

**Parameters:**
- **dimension** : `int` - Dimensionality of the address/data space
- **num_locations** : `int` - Number of hard locations (default: sqrt(2^dimension))
- **activation_radius** : `int` - Hamming radius (default: dimension * 0.451)

**Example:**
```python
# Quick creation with defaults
sdm = create_sdm(dimension=1000)

# Custom parameters
sdm = create_sdm(dimension=2000, num_locations=5000, activation_radius=900)
```

---

## Common Usage Patterns

### Basic Storage and Recall
```python
from cognitive_computing.sdm import create_sdm
import numpy as np

# Create SDM
sdm = create_sdm(dimension=1000)

# Store pattern
address = np.random.randint(0, 2, 1000)
data = np.random.randint(0, 2, 1000)
sdm.store(address, data)

# Recall pattern
recalled = sdm.recall(address)
```

### Custom Configuration
```python
from cognitive_computing.sdm import SDM, SDMConfig

config = SDMConfig(
    dimension=2000,
    num_hard_locations=5000,
    activation_radius=900,
    storage_method="counters",
    parallel=True,
    num_workers=4
)
sdm = SDM(config)
```

### Using Different Decoders
```python
from cognitive_computing.sdm.address_decoder import create_decoder

# Create SDM with custom decoder
sdm = create_sdm(dimension=1000)
decoder = create_decoder('adaptive', sdm.config, sdm.hard_locations,
                        target_activations=100)
```

### Data Encoding
```python
from cognitive_computing.sdm.utils import PatternEncoder

encoder = PatternEncoder(dimension=1000)

# Encode different data types
int_pattern = encoder.encode_integer(42)
float_pattern = encoder.encode_float(3.14159)
text_pattern = encoder.encode_string("Hello SDM")
vector_pattern = encoder.encode_vector(np.array([0.1, 0.5, 0.9]))
```

### Performance Testing
```python
from cognitive_computing.sdm.utils import test_sdm_performance

results = test_sdm_performance(sdm, test_patterns=100)
print(f"Recall accuracy: {results.recall_accuracy_mean:.2%}")
print(f"Noise tolerance at 20%: {results.noise_tolerance[0.2]:.2%}")
```

### Visualization
```python
from cognitive_computing.sdm.visualizations import (
    plot_memory_distribution,
    plot_recall_accuracy,
    visualize_memory_contents
)

# Analyze memory state
fig = plot_memory_distribution(sdm)

# Test performance
from cognitive_computing.sdm.memory import MemoryStatistics
stats = MemoryStatistics(sdm)
results = stats.analyze_recall_quality()
fig = plot_recall_accuracy(results)

# Interactive 3D visualization
fig = visualize_memory_contents(sdm, interactive=True)
```