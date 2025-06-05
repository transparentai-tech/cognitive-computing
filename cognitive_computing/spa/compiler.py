"""
High-level model specification and compilation for SPA.

This module provides a declarative API for building SPA models, parsing
action specifications, and compiling high-level descriptions into
executable networks.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np

from .core import SemanticPointer, Vocabulary, SPAConfig, SPA
from .modules import Module, State, Memory, Buffer, Gate, Compare, Connection
from .actions import Action, ActionRule, ActionSet, ActionSelection
from .networks import Network, Ensemble, EnsembleArray
from .production import Production, ProductionSystem, parse_production_rules
from .control import CognitiveControl, Routing, Gating, Sequencing

logger = logging.getLogger(__name__)


@dataclass
class ModuleSpec:
    """
    Specification for a module in the model.
    
    Parameters
    ----------
    name : str
        Module name
    type : str
        Module type (state, memory, buffer, etc.)
    dimensions : int
        Dimensionality
    vocab : str, optional
        Vocabulary name to use
    params : dict
        Additional parameters
    """
    name: str
    type: str
    dimensions: int
    vocab: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionSpec:
    """
    Specification for a connection between modules.
    
    Parameters
    ----------
    source : str
        Source module name
    target : str
        Target module name
    transform : str or array, optional
        Transformation (identity, semantic pointer, or matrix)
    gate : str, optional
        Gate module name
    """
    source: str
    target: str
    transform: Optional[Union[str, np.ndarray]] = None
    gate: Optional[str] = None


@dataclass
class ActionSpec:
    """
    Specification for an action rule.
    
    Parameters
    ----------
    name : str
        Action name
    condition : str
        Condition expression
    effect : str
        Effect expression
    priority : float
        Priority for conflict resolution
    """
    name: str
    condition: str
    effect: str
    priority: float = 0.0


class SPAModel:
    """
    High-level SPA model specification.
    
    Provides a declarative interface for building SPA models with
    modules, connections, vocabularies, and action rules.
    
    Parameters
    ----------
    name : str
        Model name
    config : SPAConfig
        Configuration parameters
    """
    
    def __init__(self, name: str, config: SPAConfig):
        """Initialize model specification."""
        self.name = name
        self.config = config
        
        # Specifications
        self.module_specs: Dict[str, ModuleSpec] = {}
        self.connection_specs: List[ConnectionSpec] = []
        self.vocab_specs: Dict[str, int] = {}  # name -> dimensions
        self.action_specs: List[ActionSpec] = []
        self.production_specs: List[str] = []
        
        # Default vocabularies
        self.vocab_specs["default"] = config.dimension
        
    def add_module(self, name: str, module_type: str, 
                   dimensions: Optional[int] = None,
                   vocab: Optional[str] = None, **params):
        """
        Add a module to the model.
        
        Parameters
        ----------
        name : str
            Module name
        module_type : str
            Type: state, memory, buffer, gate, compare, control
        dimensions : int, optional
            Dimensions (defaults to config.dimension)
        vocab : str, optional
            Vocabulary to use
        **params
            Additional module parameters
        """
        if dimensions is None:
            dimensions = self.config.dimension
            
        if vocab is None and module_type != "compare":
            vocab = "default"
            
        spec = ModuleSpec(name, module_type, dimensions, vocab, params)
        self.module_specs[name] = spec
        
    def connect(self, source: str, target: str, 
                transform: Optional[Union[str, np.ndarray]] = None,
                gate: Optional[str] = None):
        """
        Connect two modules.
        
        Parameters
        ----------
        source : str
            Source module name
        target : str
            Target module name
        transform : str or array, optional
            Transformation to apply
        gate : str, optional
            Gate controlling the connection
        """
        spec = ConnectionSpec(source, target, transform, gate)
        self.connection_specs.append(spec)
        
    def add_vocabulary(self, name: str, dimensions: Optional[int] = None):
        """
        Add a vocabulary.
        
        Parameters
        ----------
        name : str
            Vocabulary name
        dimensions : int, optional
            Dimensions (defaults to config.dimension)
        """
        if dimensions is None:
            dimensions = self.config.dimension
        self.vocab_specs[name] = dimensions
        
    def add_action(self, name: str, condition: str, effect: str, 
                   priority: float = 0.0):
        """
        Add an action rule.
        
        Parameters
        ----------
        name : str
            Action name
        condition : str
            Condition expression
        effect : str
            Effect expression
        priority : float
            Priority for conflict resolution
        """
        spec = ActionSpec(name, condition, effect, priority)
        self.action_specs.append(spec)
        
    def add_production(self, rule: str):
        """
        Add a production rule.
        
        Parameters
        ----------
        rule : str
            Production rule in format "IF condition THEN effect"
        """
        self.production_specs.append(rule)
        
    def get_module_dependencies(self) -> Dict[str, Set[str]]:
        """
        Get module dependencies from connections.
        
        Returns
        -------
        dict
            Mapping from module to set of modules it depends on
        """
        deps: Dict[str, Set[str]] = {name: set() for name in self.module_specs}
        
        for conn in self.connection_specs:
            if conn.target in deps:
                deps[conn.target].add(conn.source)
                if conn.gate:
                    deps[conn.target].add(conn.gate)
                    
        return deps


class ModelBuilder:
    """
    Builds executable models from specifications.
    
    Takes high-level model specifications and creates the corresponding
    modules, connections, and control structures.
    
    Parameters
    ----------
    model : SPAModel
        Model specification to build
    """
    
    def __init__(self, model: SPAModel):
        """Initialize builder."""
        self.model = model
        self.config = model.config
        
        # Built components
        self.vocabularies: Dict[str, Vocabulary] = {}
        self.modules: Dict[str, Module] = {}
        self.spa: Optional[SPA] = None
        self.action_selection: Optional[ActionSelection] = None
        self.production_system: Optional[ProductionSystem] = None
        
    def build(self) -> SPA:
        """
        Build the complete model.
        
        Returns
        -------
        SPA
            Built SPA system
        """
        # Build in order
        self._build_vocabularies()
        self._build_modules()
        self._build_connections()
        self._build_actions()
        self._build_productions()
        
        # Create SPA system
        self.spa = SPA(self.config)
        self.spa.modules = self.modules
        
        # Set default vocabulary
        if "default" in self.vocabularies:
            self.spa.vocabulary = self.vocabularies["default"]
        
        if self.action_selection:
            self.spa.action_selection = self.action_selection
            
        if self.production_system:
            self.spa.production_system = self.production_system
            
        return self.spa
        
    def _build_vocabularies(self):
        """Build vocabularies from specifications."""
        for name, dimensions in self.model.vocab_specs.items():
            # Create config for vocab
            vocab_config = SPAConfig(
                dimension=dimensions,
                subdimensions=self.config.subdimensions,
                normalize_pointers=self.config.normalize_pointers,
                strict_vocab=self.config.strict_vocab
            )
            vocab = Vocabulary(dimensions, vocab_config)
            self.vocabularies[name] = vocab
            
    def _build_modules(self):
        """Build modules from specifications."""
        # Sort by dependencies for proper initialization order
        deps = self.model.get_module_dependencies()
        built: Set[str] = set()
        
        def build_module(name: str):
            if name in built:
                return
                
            # Build dependencies first
            for dep in deps.get(name, set()):
                if dep in self.model.module_specs:
                    build_module(dep)
                    
            # Build this module
            spec = self.model.module_specs[name]
            vocab = self.vocabularies.get(spec.vocab) if spec.vocab else None
            
            if spec.type == "state":
                module = State(spec.name, spec.dimensions, vocab)
            elif spec.type == "memory":
                module = Memory(spec.name, spec.dimensions, vocab)
            elif spec.type == "buffer":
                module = Buffer(spec.name, spec.dimensions, self.config, vocab)
            elif spec.type == "gate":
                module = Gate(spec.name, spec.dimensions, vocab)
            elif spec.type == "compare":
                module = Compare(spec.name, spec.dimensions, vocab,
                               spec.params.get("output_dimensions", 1))
            elif spec.type == "control":
                module = CognitiveControl(spec.dimensions, self.config, vocab)
            elif spec.type == "routing":
                module = Routing(spec.dimensions, self.config)
            elif spec.type == "gating":
                module = Gating(spec.dimensions, self.config)
            elif spec.type == "sequencing":
                module = Sequencing(spec.dimensions, self.config, vocab)
            else:
                raise ValueError(f"Unknown module type: {spec.type}")
                
            # Apply additional parameters
            for key, value in spec.params.items():
                if hasattr(module, key):
                    setattr(module, key, value)
                    
            self.modules[spec.name] = module
            built.add(name)
            
        # Build all modules
        for name in self.model.module_specs:
            build_module(name)
            
    def _build_connections(self):
        """Build connections between modules."""
        for spec in self.model.connection_specs:
            source = self.modules.get(spec.source)
            target = self.modules.get(spec.target)
            
            if not source or not target:
                logger.warning(f"Cannot connect {spec.source} -> {spec.target}: "
                             f"module not found")
                continue
                
            # Determine transform
            transform = None
            if spec.transform is not None:
                if isinstance(spec.transform, str):
                    # Look up semantic pointer
                    for vocab in self.vocabularies.values():
                        if spec.transform in vocab.pointers:
                            transform = vocab.pointers[spec.transform].vector
                            # Make it a proper transform matrix
                            transform = np.outer(transform, transform)
                            break
                else:
                    # Direct matrix transform
                    transform = spec.transform
                    
            # Create connection with transform
            conn = Connection(source, target, transform)
                    
            # Apply gate
            if spec.gate:
                gate = self.modules.get(spec.gate)
                if gate:
                    # Store gate reference for gated connections
                    conn.gate = gate
                    
            # Add connection
            target.inputs[f"{source.name}_conn"] = conn
            
    def _build_actions(self):
        """Build action selection system from specifications."""
        if not self.model.action_specs:
            return
            
        # Create action selection system
        self.action_selection = ActionSelection(self.config)
        
        # Add action rules
        for spec in self.model.action_specs:
            # Add rule to system
            action = self.action_selection.add_rule(
                spec.name,
                spec.condition,
                spec.effect,
                self.modules,
                self.vocabularies.get("default", Vocabulary(self.config.dimension))
            )
            # Set priority
            action.priority = spec.priority
            
        # Connect modules
        for module in self.modules.values():
            self.action_selection.cortex.add_module(module.name, module)
            
    def _build_productions(self):
        """Build production system from specifications."""
        if not self.model.production_specs:
            return
            
        # Parse all rules
        rules_text = "\n".join(self.model.production_specs)
        productions = parse_production_rules(
            rules_text,
            self.vocabularies.get("default", Vocabulary(self.config.dimension))
        )
        
        # Create production system
        self.production_system = ProductionSystem()
        for prod in productions:
            self.production_system.add_production(prod)
            
        # Set module context
        self.production_system.set_context(
            self.modules,
            self.vocabularies.get("default", Vocabulary(self.config.dimension))
        )


def compile_model(model: SPAModel) -> Network:
    """
    Compile high-level model to neural network.
    
    This is a simplified version that returns a placeholder network.
    Full neural compilation would require significant additional
    implementation.
    
    Parameters
    ----------
    model : SPAModel
        Model specification
        
    Returns
    -------
    Network
        Compiled neural network
    """
    builder = ModelBuilder(model)
    spa = builder.build()
    
    # Create network
    network = Network("compiled_" + model.name)
    
    # Add placeholder ensembles for each module
    for name, module in spa.modules.items():
        if hasattr(module, 'dimensions'):
            ens = EnsembleArray(
                name + "_array",
                1,  # n_ensembles
                module.dimensions,  # dimensions_per_ensemble
                model.config.neurons_per_dimension
            )
            network.add_ensemble(name, ens)
            
    logger.info(f"Compiled model '{model.name}' to neural network "
               f"with {len(network.ensembles)} ensemble arrays")
    
    return network


def parse_actions(action_spec: str) -> List[ActionSpec]:
    """
    Parse action rules from string specification.
    
    Supports simple rule format:
    - "action_name: IF condition THEN effect"
    - "action_name[priority]: IF condition THEN effect"
    
    Parameters
    ----------
    action_spec : str
        Multi-line string with action rules
        
    Returns
    -------
    list of ActionSpec
        Parsed action specifications
    """
    actions = []
    
    # Pattern for action rules
    # Matches: name[priority]: IF condition THEN effect
    pattern = r'(\w+)(?:\[([0-9.]+)\])?\s*:\s*IF\s+(.+?)\s+THEN\s+(.+?)(?:$|\n)'
    
    for match in re.finditer(pattern, action_spec, re.MULTILINE | re.DOTALL):
        name = match.group(1)
        priority = float(match.group(2)) if match.group(2) else 0.0
        condition = match.group(3).strip()
        effect = match.group(4).strip()
        
        actions.append(ActionSpec(name, condition, effect, priority))
        
    return actions


def optimize_network(network: Network) -> Network:
    """
    Optimize a compiled network.
    
    Applies various optimizations like connection pruning,
    ensemble merging, and parameter tuning.
    
    Parameters
    ----------
    network : Network
        Network to optimize
        
    Returns
    -------
    Network
        Optimized network
    """
    # This is a placeholder for network optimization
    # Real implementation would include:
    # - Dead code elimination
    # - Connection strength optimization
    # - Ensemble parameter tuning
    # - Parallel pathway identification
    
    logger.info(f"Optimized network '{network.label}'")
    return network