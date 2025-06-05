"""
Production system capabilities for Semantic Pointer Architecture.

This module implements production rules (if-then rules) that provide
structured control flow and decision making in SPA models. Productions
can match patterns in module states and execute effects when conditions
are satisfied.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np

from .core import SemanticPointer, Vocabulary, SPAConfig
from .modules import Module, State
from .actions import Action, ActionRule, ActionSet

logger = logging.getLogger(__name__)


class Condition(ABC):
    """
    Abstract base class for production conditions.
    
    Conditions evaluate to a numeric value indicating confidence
    or match strength (0 = no match, 1 = perfect match).
    """
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> float:
        """
        Evaluate the condition.
        
        Parameters
        ----------
        context : dict
            Evaluation context with modules, vocabulary, etc.
            
        Returns
        -------
        float
            Match strength in [0, 1]
        """
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """String representation."""
        pass


class MatchCondition(Condition):
    """
    Condition that matches module state against a pattern.
    
    Parameters
    ----------
    module_name : str
        Name of module to check
    pattern : Union[str, SemanticPointer, np.ndarray]
        Pattern to match against
    threshold : float
        Minimum similarity for match (default: 0.7)
    """
    
    def __init__(self, module_name: str, 
                 pattern: Union[str, SemanticPointer, np.ndarray],
                 threshold: float = 0.7):
        """Initialize match condition."""
        self.module_name = module_name
        self.pattern = pattern
        self.threshold = threshold
        
    def evaluate(self, context: Dict[str, Any]) -> float:
        """Evaluate similarity match."""
        modules = context.get('modules', {})
        vocab = context.get('vocab')
        
        if self.module_name not in modules:
            logger.warning(f"Module '{self.module_name}' not found in context")
            return 0.0
            
        module = modules[self.module_name]
        
        # Get pattern vector
        if isinstance(self.pattern, str):
            if vocab is None:
                logger.warning("No vocabulary provided for pattern lookup")
                return 0.0
            try:
                pattern_vec = vocab[self.pattern].vector
            except KeyError:
                logger.warning(f"Pattern '{self.pattern}' not in vocabulary")
                return 0.0
        elif isinstance(self.pattern, SemanticPointer):
            pattern_vec = self.pattern.vector
        else:
            pattern_vec = self.pattern
            
        # Compute similarity
        module_vec = module.state
        similarity = np.dot(module_vec, pattern_vec) / (
            np.linalg.norm(module_vec) * np.linalg.norm(pattern_vec) + 1e-10)
        
        # Return normalized match strength
        if similarity >= self.threshold:
            # Scale from [threshold, 1] to [0, 1]
            return (similarity - self.threshold) / (1 - self.threshold)
        else:
            return 0.0
    
    def __repr__(self) -> str:
        """String representation."""
        pattern_str = self.pattern if isinstance(self.pattern, str) else "vector"
        return f"Match({self.module_name} ~ {pattern_str}, threshold={self.threshold})"


class CompareCondition(Condition):
    """
    Condition that compares two module states.
    
    Parameters
    ----------
    module1_name : str
        First module name
    module2_name : str
        Second module name
    threshold : float
        Minimum similarity for match
    """
    
    def __init__(self, module1_name: str, module2_name: str,
                 threshold: float = 0.7):
        """Initialize compare condition."""
        self.module1_name = module1_name
        self.module2_name = module2_name
        self.threshold = threshold
        
    def evaluate(self, context: Dict[str, Any]) -> float:
        """Evaluate similarity between modules."""
        modules = context.get('modules', {})
        
        if self.module1_name not in modules or self.module2_name not in modules:
            return 0.0
            
        vec1 = modules[self.module1_name].state
        vec2 = modules[self.module2_name].state
        
        similarity = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
        
        if similarity >= self.threshold:
            return (similarity - self.threshold) / (1 - self.threshold)
        else:
            return 0.0
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Compare({self.module1_name} ~ {self.module2_name}, threshold={self.threshold})"


class CompoundCondition(Condition):
    """
    Compound condition combining multiple conditions.
    
    Parameters
    ----------
    conditions : List[Condition]
        Sub-conditions to combine
    operation : str
        How to combine: "and", "or", "not"
    """
    
    def __init__(self, conditions: List[Condition], operation: str = "and"):
        """Initialize compound condition."""
        self.conditions = conditions
        self.operation = operation.lower()
        
        if self.operation not in ["and", "or", "not"]:
            raise ValueError(f"Unknown operation: {self.operation}")
            
        if self.operation == "not" and len(conditions) != 1:
            raise ValueError("NOT operation requires exactly one condition")
            
    def evaluate(self, context: Dict[str, Any]) -> float:
        """Evaluate compound condition."""
        if not self.conditions:
            return 0.0
            
        values = [cond.evaluate(context) for cond in self.conditions]
        
        if self.operation == "and":
            # Minimum value (all must be satisfied)
            return min(values)
        elif self.operation == "or":
            # Maximum value (any can be satisfied)
            return max(values)
        elif self.operation == "not":
            # Invert single condition
            return 1.0 - values[0]
        else:
            return 0.0
    
    def __repr__(self) -> str:
        """String representation."""
        op_str = f" {self.operation.upper()} "
        cond_strs = [repr(c) for c in self.conditions]
        return f"({op_str.join(cond_strs)})"


class Effect(ABC):
    """
    Abstract base class for production effects.
    
    Effects modify the state of the system when executed.
    """
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]):
        """
        Execute the effect.
        
        Parameters
        ----------
        context : dict
            Execution context with modules, vocabulary, etc.
        """
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """String representation."""
        pass


class SetEffect(Effect):
    """
    Effect that sets a module's state.
    
    Parameters
    ----------
    module_name : str
        Module to modify
    value : Union[str, SemanticPointer, np.ndarray]
        Value to set
    """
    
    def __init__(self, module_name: str,
                 value: Union[str, SemanticPointer, np.ndarray]):
        """Initialize set effect."""
        self.module_name = module_name
        self.value = value
        
    def execute(self, context: Dict[str, Any]):
        """Set module state."""
        modules = context.get('modules', {})
        vocab = context.get('vocab')
        
        if self.module_name not in modules:
            logger.warning(f"Module '{self.module_name}' not found")
            return
            
        module = modules[self.module_name]
        
        # Get value vector
        if isinstance(self.value, str):
            if vocab is None:
                logger.warning("No vocabulary provided for value lookup")
                return
            try:
                value_vec = vocab[self.value].vector
            except KeyError:
                logger.warning(f"Value '{self.value}' not in vocabulary")
                return
        elif isinstance(self.value, SemanticPointer):
            value_vec = self.value.vector
        else:
            value_vec = self.value
            
        # Set state
        module.state = value_vec
    
    def __repr__(self) -> str:
        """String representation."""
        value_str = self.value if isinstance(self.value, str) else "vector"
        return f"Set({self.module_name} = {value_str})"


class BindEffect(Effect):
    """
    Effect that binds two values and stores in a module.
    
    Parameters
    ----------
    module_name : str
        Module to store result
    source1 : Union[str, Module]
        First source (module name or direct reference)
    source2 : Union[str, Module]
        Second source
    """
    
    def __init__(self, module_name: str,
                 source1: Union[str, Module],
                 source2: Union[str, Module]):
        """Initialize bind effect."""
        self.module_name = module_name
        self.source1 = source1
        self.source2 = source2
        
    def execute(self, context: Dict[str, Any]):
        """Execute binding operation."""
        modules = context.get('modules', {})
        vocab = context.get('vocab')
        
        if self.module_name not in modules:
            logger.warning(f"Target module '{self.module_name}' not found")
            return
            
        # Get source vectors
        vec1 = self._get_vector(self.source1, modules, vocab)
        vec2 = self._get_vector(self.source2, modules, vocab)
        
        if vec1 is None or vec2 is None:
            return
            
        # Perform binding (circular convolution)
        result = np.fft.ifft(np.fft.fft(vec1) * np.fft.fft(vec2)).real
        
        # Store result
        modules[self.module_name].state = result
        
    def _get_vector(self, source: Union[str, Module], 
                    modules: Dict[str, Module],
                    vocab: Optional[Vocabulary]) -> Optional[np.ndarray]:
        """Get vector from source."""
        if isinstance(source, str):
            # Could be module name or vocabulary item
            if source in modules:
                return modules[source].state
            elif vocab and source in vocab.pointers:
                return vocab[source].vector
            else:
                logger.warning(f"Source '{source}' not found")
                return None
        elif isinstance(source, Module):
            return source.state
        else:
            return None
    
    def __repr__(self) -> str:
        """String representation."""
        s1 = self.source1 if isinstance(self.source1, str) else "module"
        s2 = self.source2 if isinstance(self.source2, str) else "module"
        return f"Bind({self.module_name} = {s1} * {s2})"


class CompoundEffect(Effect):
    """
    Effect that executes multiple sub-effects.
    
    Parameters
    ----------
    effects : List[Effect]
        Effects to execute in sequence
    """
    
    def __init__(self, effects: List[Effect]):
        """Initialize compound effect."""
        self.effects = effects
        
    def execute(self, context: Dict[str, Any]):
        """Execute all sub-effects."""
        for effect in self.effects:
            try:
                effect.execute(context)
            except Exception as e:
                logger.error(f"Error executing effect {effect}: {e}")
                
    def __repr__(self) -> str:
        """String representation."""
        effect_strs = [repr(e) for e in self.effects]
        return f"Compound([{', '.join(effect_strs)}])"


@dataclass
class Production:
    """
    Single if-then production rule.
    
    Parameters
    ----------
    name : str
        Name of the production
    condition : Condition
        Condition for firing
    effect : Effect
        Effect when fired
    priority : float
        Priority for conflict resolution
    """
    name: str
    condition: Condition
    effect: Effect
    priority: float = 0.0
    
    # Tracking
    _strength: float = field(init=False, default=0.0)
    _fired_count: int = field(init=False, default=0)
    
    def evaluate(self, context: Dict[str, Any]) -> float:
        """
        Evaluate production strength.
        
        Parameters
        ----------
        context : dict
            Evaluation context
            
        Returns
        -------
        float
            Production strength
        """
        self._strength = self.condition.evaluate(context)
        return self._strength
    
    def fire(self, context: Dict[str, Any]):
        """
        Fire the production.
        
        Parameters
        ----------
        context : dict
            Execution context
        """
        logger.debug(f"Firing production '{self.name}'")
        self.effect.execute(context)
        self._fired_count += 1
        
    def __repr__(self) -> str:
        """String representation."""
        return (f"Production('{self.name}': {self.condition} -> {self.effect}, "
                f"strength={self._strength:.3f}, fired={self._fired_count})")


class ProductionSystem:
    """
    Collection of productions with conflict resolution.
    
    Parameters
    ----------
    productions : List[Production]
        Initial productions
    conflict_resolution : str
        How to resolve conflicts: "first", "priority", "specificity"
    """
    
    def __init__(self, productions: Optional[List[Production]] = None,
                 conflict_resolution: str = "priority"):
        """Initialize production system."""
        self.productions = productions or []
        self.conflict_resolution = conflict_resolution
        
        # Execution context
        self._context: Dict[str, Any] = {}
        self._fired_productions: List[str] = []
        
    def add_production(self, production: Production):
        """Add a production to the system."""
        self.productions.append(production)
        
    def set_context(self, modules: Dict[str, Module],
                    vocab: Optional[Vocabulary] = None,
                    **kwargs):
        """
        Set execution context.
        
        Parameters
        ----------
        modules : dict
            Module name to module mapping
        vocab : Vocabulary, optional
            Vocabulary for symbol lookup
        **kwargs
            Additional context variables
        """
        self._context = {
            'modules': modules,
            'vocab': vocab,
            **kwargs
        }
        
    def evaluate_all(self) -> List[Tuple[Production, float]]:
        """
        Evaluate all productions.
        
        Returns
        -------
        list
            List of (production, strength) tuples
        """
        results = []
        for prod in self.productions:
            strength = prod.evaluate(self._context)
            if strength > 0:
                results.append((prod, strength))
        return results
    
    def select_production(self) -> Optional[Production]:
        """
        Select production to fire based on conflict resolution.
        
        Returns
        -------
        Production or None
            Selected production
        """
        # Get active productions
        active = self.evaluate_all()
        
        if not active:
            return None
            
        if self.conflict_resolution == "first":
            # First matching production
            return active[0][0]
            
        elif self.conflict_resolution == "priority":
            # Highest priority, then highest strength
            active.sort(key=lambda x: (x[0].priority, x[1]), reverse=True)
            return active[0][0]
            
        elif self.conflict_resolution == "specificity":
            # Most specific condition (highest strength)
            active.sort(key=lambda x: x[1], reverse=True)
            return active[0][0]
            
        else:
            raise ValueError(f"Unknown conflict resolution: {self.conflict_resolution}")
    
    def step(self) -> bool:
        """
        Execute one production cycle.
        
        Returns
        -------
        bool
            True if a production fired
        """
        selected = self.select_production()
        
        if selected is not None:
            selected.fire(self._context)
            self._fired_productions.append(selected.name)
            return True
        return False
    
    def run(self, max_cycles: int = 100) -> int:
        """
        Run production system until no productions fire.
        
        Parameters
        ----------
        max_cycles : int
            Maximum cycles to prevent infinite loops
            
        Returns
        -------
        int
            Number of cycles executed
        """
        self._fired_productions.clear()
        
        cycles = 0
        for cycle in range(max_cycles):
            if not self.step():
                break
            cycles = cycle + 1
                
        return cycles
    
    def get_fired_productions(self) -> List[str]:
        """Get list of fired production names."""
        return self._fired_productions.copy()
    
    def reset(self):
        """Reset tracking information."""
        self._fired_productions.clear()
        for prod in self.productions:
            prod._strength = 0.0
            prod._fired_count = 0
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ProductionSystem({len(self.productions)} productions, {self.conflict_resolution})"


class ConditionalModule(Module):
    """
    Module that only updates when condition is met.
    
    This provides a way to gate module updates based on
    production system conditions.
    
    Parameters
    ----------
    name : str
        Module name
    dimensions : int
        Vector dimensions
    condition : Condition
        Update condition
    base_module : Module
        Underlying module to wrap
    vocab : Vocabulary, optional
        Vocabulary for condition evaluation
    """
    
    def __init__(self, name: str, dimensions: int,
                 condition: Condition, base_module: Module,
                 vocab: Optional[Vocabulary] = None):
        """Initialize conditional module."""
        super().__init__(name, dimensions, vocab)
        self.condition = condition
        self.base_module = base_module
        self._gated = False
        
    def update(self, dt: float):
        """Update only if condition is met."""
        # Evaluate condition
        context = {
            'modules': {self.name: self},
            'vocab': self.vocabulary if hasattr(self, 'vocabulary') else None
        }
        strength = self.condition.evaluate(context)
        
        self._gated = strength > 0.5
        
        if self._gated:
            # Pass through to base module
            self.base_module.update(dt)
            self._state = self.base_module.state
            
    @property
    def is_gated(self) -> bool:
        """Check if module is currently gated open."""
        return self._gated


def parse_production_rules(rules_text: str, 
                          vocab: Optional[Vocabulary] = None) -> List[Production]:
    """
    Parse production rules from text specification.
    
    Simple syntax:
    - IF <condition> THEN <effect>
    - Conditions: <module> MATCHES <pattern>
    - Effects: SET <module> TO <value>
    
    Parameters
    ----------
    rules_text : str
        Rules specification
    vocab : Vocabulary, optional
        Vocabulary for symbol lookup
        
    Returns
    -------
    list
        List of productions
    """
    productions = []
    
    lines = rules_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Simple IF-THEN parsing
        if ' THEN ' in line:
            condition_str, effect_str = line.split(' THEN ', 1)
            condition_str = condition_str.replace('IF ', '', 1).strip()
            
            # Parse condition (simple MATCHES for now)
            if ' MATCHES ' in condition_str:
                module_name, pattern = condition_str.split(' MATCHES ', 1)
                condition = MatchCondition(module_name.strip(), pattern.strip())
            else:
                logger.warning(f"Unknown condition format: {condition_str}")
                continue
                
            # Parse effect (simple SET TO for now)
            if effect_str.startswith('SET ') and ' TO ' in effect_str:
                parts = effect_str[4:].split(' TO ', 1)
                module_name = parts[0].strip()
                value = parts[1].strip()
                effect = SetEffect(module_name, value)
            else:
                logger.warning(f"Unknown effect format: {effect_str}")
                continue
                
            # Create production
            prod_name = f"rule_{len(productions) + 1}"
            productions.append(Production(prod_name, condition, effect))
            
    return productions