"""
Hyperdimensional Computing (HDC) module.

This module implements brain-inspired computing using high-dimensional vectors
for robust and efficient information processing, particularly suited for
classification, sensor fusion, and edge computing applications.
"""

from cognitive_computing.hdc.core import (
    HDC,
    HDCConfig,
    HypervectorType,
    create_hdc,
)

from cognitive_computing.hdc.hypervectors import (
    BinaryHypervector,
    BipolarHypervector,
    TernaryHypervector,
    LevelHypervector,
    generate_orthogonal_hypervectors,
    fractional_binding,
    protect_hypervector,
    unprotect_hypervector,
)

from cognitive_computing.hdc.operations import (
    BundlingMethod,
    PermutationMethod,
    bind_hypervectors,
    bundle_hypervectors,
    permute_hypervector,
    similarity,
    noise_hypervector,
    thin_hypervector,
    segment_hypervector,
    concatenate_hypervectors,
    power_hypervector,
    normalize_hypervector,
    protect_sequence,
)

from cognitive_computing.hdc.item_memory import ItemMemory

from cognitive_computing.hdc.encoding import (
    Encoder,
    ScalarEncoder,
    CategoricalEncoder,
    SequenceEncoder,
    SpatialEncoder,
    RecordEncoder,
    NGramEncoder,
)

from cognitive_computing.hdc.classifiers import (
    HDClassifier,
    OneShotClassifier,
    AdaptiveClassifier,
    EnsembleClassifier,
    HierarchicalClassifier,
)

from cognitive_computing.hdc.utils import (
    HDCPerformanceMetrics,
    measure_capacity,
    benchmark_operations,
    analyze_binding_properties,
    compare_hypervector_types,
    generate_similarity_matrix,
    measure_associativity,
    estimate_required_dimension,
    create_codebook,
    measure_classifier_performance,
)

from cognitive_computing.hdc.visualizations import (
    plot_hypervector,
    plot_similarity_matrix,
    plot_binding_operation,
    plot_capacity_analysis,
    plot_classifier_performance,
    plot_hypervector_comparison,
    create_interactive_similarity_plot,
    save_plots,
)

# Version information
__version__ = "0.1.0"

# Public API
__all__ = [
    # Core classes
    "HDC",
    "HDCConfig",
    "HypervectorType",
    
    # Hypervector types
    "BinaryHypervector",
    "BipolarHypervector", 
    "TernaryHypervector",
    "LevelHypervector",
    
    # Operations
    "BundlingMethod",
    "PermutationMethod",
    "bind_hypervectors",
    "bundle_hypervectors",
    "permute_hypervector",
    "similarity",
    "noise_hypervector",
    "thin_hypervector",
    "segment_hypervector",
    "concatenate_hypervectors",
    "power_hypervector",
    "normalize_hypervector",
    "protect_sequence",
    
    # Memory
    "ItemMemory",
    
    # Encoders
    "Encoder",
    "ScalarEncoder",
    "CategoricalEncoder",
    "SequenceEncoder",
    "SpatialEncoder",
    "RecordEncoder",
    "NGramEncoder",
    
    # Classifiers
    "HDClassifier",
    "OneShotClassifier",
    "AdaptiveClassifier",
    "EnsembleClassifier",
    "HierarchicalClassifier",
    
    # Factory and utility functions
    "create_hdc",
    "generate_orthogonal_hypervectors",
    "fractional_binding",
    "protect_hypervector",
    "unprotect_hypervector",
    
    # Utils
    "HDCPerformanceMetrics",
    "measure_capacity",
    "benchmark_operations",
    "analyze_binding_properties",
    "compare_hypervector_types",
    "generate_similarity_matrix",
    "measure_associativity",
    "estimate_required_dimension",
    "create_codebook",
    "measure_classifier_performance",
    
    # Visualizations
    "plot_hypervector",
    "plot_similarity_matrix",
    "plot_binding_operation",
    "plot_capacity_analysis",
    "plot_classifier_performance",
    "plot_hypervector_comparison",
    "create_interactive_similarity_plot",
    "save_plots",
]