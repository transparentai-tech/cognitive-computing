"""
Semantic Pointer Architecture (SPA) implementation.

SPA provides a cognitive architecture that extends Holographic Reduced 
Representations (HRR) with structured control flow, action selection, 
and neural implementation principles.
"""

# Version will be imported when we have core components
__version__ = "0.1.0"

# Core imports
from .core import (
    SPAConfig,
    SemanticPointer,
    Vocabulary,
    SPA,
    create_spa,
    create_vocabulary,
)

# Module imports
from .modules import (
    Module,
    State,
    Memory,
    AssociativeMemory,
    Buffer,
    Gate,
    Compare,
    DotProduct,
    Connection,
)

# Action imports
from .actions import (
    Action,
    ActionRule,
    ActionSet,
    BasalGanglia,
    Thalamus,
    Cortex,
    ActionSelection,
)

# Network imports
from .networks import (
    NeuronParams,
    Ensemble,
    EnsembleArray,
    Connection as NetworkConnection,
    Probe,
    Network,
    CircularConvolution,
)

# Production imports
from .production import (
    Condition,
    MatchCondition,
    CompareCondition,
    CompoundCondition,
    Effect,
    SetEffect,
    BindEffect,
    CompoundEffect,
    Production,
    ProductionSystem,
    ConditionalModule,
    parse_production_rules,
)

# Control imports
from .control import (
    CognitiveControl,
    Routing,
    Gating,
    Sequencing,
)

# Compiler imports
from .compiler import (
    ModuleSpec,
    ConnectionSpec,
    ActionSpec,
    SPAModel,
    ModelBuilder,
    compile_model,
    parse_actions,
    optimize_network,
)

# Utils imports
from .utils import (
    make_unitary,
    similarity,
    normalize_semantic_pointer,
    generate_pointers,
    analyze_vocabulary,
    measure_binding_capacity,
    create_transformation_matrix,
    estimate_module_capacity,
    analyze_production_system,
    optimize_action_thresholds,
)

__all__ = [
    # Core
    "SPAConfig",
    "SemanticPointer",
    "Vocabulary", 
    "SPA",
    "create_spa",
    "create_vocabulary",
    # Modules
    "Module",
    "State",
    "Memory",
    "AssociativeMemory",
    "Buffer",
    "Gate",
    "Compare",
    "DotProduct",
    "Connection",
    # Actions
    "Action",
    "ActionRule",
    "ActionSet",
    "BasalGanglia",
    "Thalamus",
    "Cortex",
    "ActionSelection",
    # Networks
    "NeuronParams",
    "Ensemble",
    "EnsembleArray",
    "NetworkConnection",
    "Probe",
    "Network",
    "CircularConvolution",
    # Production
    "Condition",
    "MatchCondition",
    "CompareCondition",
    "CompoundCondition",
    "Effect",
    "SetEffect",
    "BindEffect",
    "CompoundEffect",
    "Production",
    "ProductionSystem",
    "ConditionalModule",
    "parse_production_rules",
    # Control
    "CognitiveControl",
    "Routing",
    "Gating",
    "Sequencing",
    # Compiler
    "ModuleSpec",
    "ConnectionSpec",
    "ActionSpec",
    "SPAModel",
    "ModelBuilder",
    "compile_model",
    "parse_actions",
    "optimize_network",
    # Utils
    "make_unitary",
    "similarity",
    "normalize_semantic_pointer",
    "generate_pointers",
    "analyze_vocabulary",
    "measure_binding_capacity",
    "create_transformation_matrix",
    "estimate_module_capacity",
    "analyze_production_system",
    "optimize_action_thresholds",
]