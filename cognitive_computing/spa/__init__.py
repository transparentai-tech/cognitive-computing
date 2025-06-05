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
]