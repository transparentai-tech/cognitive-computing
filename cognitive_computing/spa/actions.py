"""
Action selection mechanisms for Semantic Pointer Architecture.

This module implements the basal ganglia-inspired action selection system,
thalamus for routing, and cortical action execution. These components work
together to provide cognitive control and decision making in SPA models.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np

from .core import SemanticPointer, Vocabulary, SPAConfig
from .modules import Module, State, Connection

logger = logging.getLogger(__name__)


@dataclass
class Action:
    """
    Single action with condition and effect.
    
    An action represents a conditional rule that can modify the state
    of SPA modules when its condition is satisfied.
    
    Parameters
    ----------
    name : str
        Name of the action
    condition : Callable
        Function that returns utility value for this action
    effect : Callable
        Function that executes when action is selected
    priority : float
        Priority for tie-breaking (higher is preferred)
    """
    name: str
    condition: Callable[[], float]
    effect: Callable[[], None]
    priority: float = 0.0
    
    # Tracking
    _utility: float = field(init=False, default=0.0)
    _selected_count: int = field(init=False, default=0)
    
    def evaluate(self) -> float:
        """
        Evaluate the action's utility.
        
        Returns
        -------
        float
            Utility value (higher means more likely to be selected)
        """
        try:
            self._utility = float(self.condition())
        except Exception as e:
            logger.warning(f"Error evaluating condition for action {self.name}: {e}")
            self._utility = 0.0
        return self._utility
    
    def execute(self):
        """Execute the action's effect."""
        try:
            self.effect()
            self._selected_count += 1
        except Exception as e:
            logger.error(f"Error executing effect for action {self.name}: {e}")
            raise
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"Action('{self.name}', utility={self._utility:.3f}, "
                f"selected={self._selected_count})")


@dataclass
class ActionRule:
    """
    Rule-based action specification.
    
    ActionRule provides a higher-level way to specify actions using
    string expressions that get compiled into condition and effect functions.
    
    Parameters
    ----------
    name : str
        Name of the action
    condition_expr : str
        Expression for condition (e.g., "dot(state.vision, CIRCLE) > 0.5")
    effect_expr : str
        Expression for effect (e.g., "motor.set(GRASP)")
    modules : Dict[str, Module]
        Module references for evaluation
    vocab : Vocabulary
        Vocabulary for symbol lookup
    """
    name: str
    condition_expr: str
    effect_expr: str
    modules: Dict[str, Module]
    vocab: Vocabulary
    
    def compile(self) -> Action:
        """
        Compile the rule into an Action.
        
        Returns
        -------
        Action
            Compiled action with condition and effect functions
        """
        # Create evaluation context
        context = {
            'dot': self._dot_product,
            'sim': self._similarity,
            **self.modules,
            **{name: self.vocab[name] for name in self.vocab.pointers}
        }
        
        # Compile condition
        def condition():
            try:
                return eval(self.condition_expr, {"__builtins__": {}}, context)
            except Exception as e:
                logger.warning(f"Condition evaluation error in {self.name}: {e}")
                return 0.0
        
        # Compile effect
        def effect():
            exec(self.effect_expr, {"__builtins__": {}}, context)
        
        return Action(self.name, condition, effect)
    
    def _dot_product(self, a: Union[Module, SemanticPointer, np.ndarray],
                     b: Union[Module, SemanticPointer, np.ndarray]) -> float:
        """Helper for dot product in expressions."""
        # Extract vectors
        if isinstance(a, Module):
            vec_a = a.state
        elif isinstance(a, SemanticPointer):
            vec_a = a.vector
        else:
            vec_a = a
            
        if isinstance(b, Module):
            vec_b = b.state
        elif isinstance(b, SemanticPointer):
            vec_b = b.vector
        else:
            vec_b = b
            
        return np.dot(vec_a, vec_b)
    
    def _similarity(self, a: Any, b: Any) -> float:
        """Helper for similarity in expressions."""
        dot = self._dot_product(a, b)
        # Get norms
        vec_a = a.state if isinstance(a, Module) else (
            a.vector if isinstance(a, SemanticPointer) else a)
        vec_b = b.state if isinstance(b, Module) else (
            b.vector if isinstance(b, SemanticPointer) else b)
        
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a > 0 and norm_b > 0:
            return dot / (norm_a * norm_b)
        return 0.0


class ActionSet:
    """
    Collection of actions with utility-based selection.
    
    Parameters
    ----------
    actions : List[Action]
        Initial set of actions
    """
    
    def __init__(self, actions: Optional[List[Action]] = None):
        """Initialize action set."""
        self.actions = actions or []
        self._utilities = np.zeros(len(self.actions))
        
    def add_action(self, action: Action):
        """Add an action to the set."""
        self.actions.append(action)
        self._utilities = np.zeros(len(self.actions))
        
    def evaluate_all(self) -> np.ndarray:
        """
        Evaluate all action utilities.
        
        Returns
        -------
        np.ndarray
            Array of utility values
        """
        self._utilities = np.array([action.evaluate() for action in self.actions])
        return self._utilities
    
    def select_action(self, method: str = "max") -> Optional[Action]:
        """
        Select an action based on utilities.
        
        Parameters
        ----------
        method : str
            Selection method: "max", "softmax", or "epsilon-greedy"
            
        Returns
        -------
        Action or None
            Selected action or None if no action has positive utility
        """
        if not self.actions:
            return None
            
        self.evaluate_all()
        
        if method == "max":
            # Select action with highest utility
            if np.max(self._utilities) <= 0:
                return None
            idx = np.argmax(self._utilities)
            return self.actions[idx]
            
        elif method == "softmax":
            # Probabilistic selection based on utilities
            if np.max(self._utilities) <= 0:
                return None
            # Shift to avoid negative utilities
            shifted = self._utilities - np.min(self._utilities) + 1e-10
            probs = np.exp(shifted) / np.sum(np.exp(shifted))
            idx = np.random.choice(len(self.actions), p=probs)
            return self.actions[idx]
            
        elif method == "epsilon-greedy":
            # With small probability, select random action
            epsilon = 0.1
            if np.random.random() < epsilon:
                idx = np.random.choice(len(self.actions))
            else:
                if np.max(self._utilities) <= 0:
                    return None
                idx = np.argmax(self._utilities)
            return self.actions[idx]
            
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def __len__(self) -> int:
        """Number of actions."""
        return len(self.actions)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ActionSet({len(self.actions)} actions)"


class BasalGanglia:
    """
    Basal ganglia model for action selection.
    
    The basal ganglia implements winner-take-all action selection through
    mutual inhibition, inspired by the neurobiological circuits involved
    in motor control and decision making.
    
    Parameters
    ----------
    action_set : ActionSet
        Set of actions to select from
    config : SPAConfig
        Configuration with parameters like threshold and mutual_inhibition
    """
    
    def __init__(self, action_set: ActionSet, config: SPAConfig):
        """Initialize basal ganglia."""
        self.name = "BasalGanglia"
        self.dimensions = len(action_set.actions)
        self.action_set = action_set
        self.config = config
        
        # Internal state for action selection
        self._state = np.zeros(self.dimensions)
        self._utilities = np.zeros(self.dimensions)
        self._activations = np.zeros(self.dimensions)
        
    @property
    def state(self) -> np.ndarray:
        """Get current state."""
        return self._state.copy()
        
    def update(self, dt: float):
        """
        Update basal ganglia state.
        
        Implements mutual inhibition for action selection.
        """
        # Evaluate all utilities
        self._utilities = self.action_set.evaluate_all()
        
        # Apply threshold
        above_threshold = self._utilities > self.config.threshold
        
        # Compute activations with mutual inhibition
        for i in range(len(self.action_set.actions)):
            if above_threshold[i]:
                # Activation = utility - inhibition from other actions
                inhibition = 0.0
                for j in range(len(self.action_set.actions)):
                    if i != j and above_threshold[j]:
                        # Inhibition proportional to other utilities
                        inhibition += self.config.mutual_inhibition * self._utilities[j]
                
                self._activations[i] = self._utilities[i] - inhibition + self.config.bg_bias
            else:
                self._activations[i] = 0.0
        
        # Ensure non-negative activations
        self._activations = np.maximum(self._activations, 0.0)
        
        # Normalize to [0, 1] if any activation
        if np.max(self._activations) > 0:
            self._state = self._activations / np.max(self._activations)
        else:
            self._state = np.zeros_like(self._activations)
    
    def get_selected_action(self) -> Optional[Action]:
        """
        Get the currently selected action.
        
        Returns
        -------
        Action or None
            Action with highest activation or None if no action selected
        """
        if np.max(self._activations) <= 0:
            return None
            
        idx = np.argmax(self._activations)
        return self.action_set.actions[idx]
    
    @property
    def utilities(self) -> np.ndarray:
        """Get current utility values."""
        return self._utilities.copy()
    
    @property
    def activations(self) -> np.ndarray:
        """Get current activation values."""
        return self._activations.copy()


class Thalamus:
    """
    Thalamus model for routing selected actions to target modules.
    
    The thalamus takes the output from basal ganglia and routes
    information flow to implement the selected action's effects.
    
    Parameters
    ----------
    basal_ganglia : BasalGanglia
        Basal ganglia providing action selection
    config : SPAConfig
        Configuration parameters
    """
    
    def __init__(self, basal_ganglia: BasalGanglia, config: SPAConfig):
        """Initialize thalamus."""
        self.name = "Thalamus"
        self.dimensions = basal_ganglia.dimensions
        self.basal_ganglia = basal_ganglia
        self.config = config
        
        # Internal state
        self._state = np.zeros(self.dimensions)
        # Routing gates for each action
        self._gates = np.zeros(self.dimensions)
        
    @property
    def state(self) -> np.ndarray:
        """Get current state."""
        return self._state.copy()
        
    def update(self, dt: float):
        """
        Update thalamus routing.
        
        Creates gates based on basal ganglia output.
        """
        # Get basal ganglia output
        bg_output = self.basal_ganglia.state
        
        # Apply routing inhibition to create gates
        # Only the winning action gets full routing
        max_idx = np.argmax(bg_output)
        
        for i in range(len(self._gates)):
            if bg_output[i] > 0.5:  # Action is selected
                if i == max_idx:
                    # Winner gets full routing
                    self._gates[i] = 1.0
                else:
                    # Others are inhibited
                    self._gates[i] = 1.0 / (1.0 + self.config.routing_inhibition)
            else:
                # No routing for inactive actions
                self._gates[i] = 0.0
        
        self._state = self._gates
    
    def get_gate(self, action_index: int) -> float:
        """
        Get gate value for specific action.
        
        Parameters
        ----------
        action_index : int
            Index of the action
            
        Returns
        -------
        float
            Gate value in [0, 1]
        """
        if 0 <= action_index < len(self._gates):
            return self._gates[action_index]
        return 0.0
    
    def route(self, source: Module, target: Module, action_index: int,
              transform: Optional[np.ndarray] = None):
        """
        Route information from source to target for specific action.
        
        Parameters
        ----------
        source : Module
            Source of information
        target : Module
            Target for routed information
        action_index : int
            Index of action controlling this route
        transform : np.ndarray, optional
            Transformation to apply
        """
        gate = self.get_gate(action_index)
        
        if gate > 0:
            # Get source output
            source_output = source.state
            
            # Apply transform if provided
            if transform is not None:
                output = np.dot(transform, source_output)
            else:
                output = source_output
            
            # Apply gating and add to target
            target.state += gate * output


class Cortex:
    """
    Cortical module for implementing action effects.
    
    The cortex executes the effects of selected actions, modifying
    the states of various modules according to the action rules.
    
    Parameters
    ----------
    modules : Dict[str, Module]
        Dictionary of modules that can be affected
    config : SPAConfig
        Configuration parameters
    """
    
    def __init__(self, modules: Dict[str, Module], config: SPAConfig):
        """Initialize cortex."""
        self.name = "Cortex"
        self.dimensions = 1  # Single activity indicator
        self.modules = modules
        self.config = config
        self.basal_ganglia: Optional[BasalGanglia] = None
        self.thalamus: Optional[Thalamus] = None
        
        # Internal state
        self._state = np.zeros(self.dimensions)
        
    @property
    def state(self) -> np.ndarray:
        """Get current state."""
        return self._state.copy()
        
    def connect_control(self, basal_ganglia: BasalGanglia, thalamus: Thalamus):
        """
        Connect control structures.
        
        Parameters
        ----------
        basal_ganglia : BasalGanglia
            Action selection system
        thalamus : Thalamus
            Routing system
        """
        self.basal_ganglia = basal_ganglia
        self.thalamus = thalamus
        
    def update(self, dt: float):
        """
        Update cortex by executing selected action.
        
        The cortex monitors the basal ganglia output and executes
        the effect of the currently selected action.
        """
        if self.basal_ganglia is None:
            return
            
        # Get selected action
        action = self.basal_ganglia.get_selected_action()
        
        if action is not None:
            # Check if action is sufficiently active
            action_idx = self.basal_ganglia.action_set.actions.index(action)
            if self.thalamus is not None:
                gate = self.thalamus.get_gate(action_idx)
                if gate > 0.5:  # Only execute if gate is open
                    action.execute()
            else:
                # No thalamus, execute directly if selected
                if self.basal_ganglia.state[action_idx] > 0.5:
                    action.execute()
        
        # Cortex output is just activity indicator
        self._state[0] = 1.0 if action is not None else 0.0
    
    def add_module(self, name: str, module: Module):
        """Add a module that can be controlled."""
        self.modules[name] = module
    
    def get_module(self, name: str) -> Optional[Module]:
        """Get a module by name."""
        return self.modules.get(name)


class ActionSelection:
    """
    High-level action selection system combining BG, Thalamus, and Cortex.
    
    This class provides a convenient interface for setting up a complete
    action selection system with all necessary components.
    
    Parameters
    ----------
    config : SPAConfig
        Configuration parameters
    """
    
    def __init__(self, config: SPAConfig):
        """Initialize action selection system."""
        self.config = config
        
        # Create components
        self.action_set = ActionSet()
        self.basal_ganglia = BasalGanglia(self.action_set, config)
        self.thalamus = Thalamus(self.basal_ganglia, config)
        self.cortex = Cortex({}, config)
        
        # Connect control
        self.cortex.connect_control(self.basal_ganglia, self.thalamus)
        
    def add_action(self, action: Action):
        """Add an action to the system."""
        self.action_set.add_action(action)
        
        # Recreate BG and Thalamus with new dimensions
        self.basal_ganglia = BasalGanglia(self.action_set, self.config)
        self.thalamus = Thalamus(self.basal_ganglia, self.config)
        self.cortex.connect_control(self.basal_ganglia, self.thalamus)
        
    def add_rule(self, name: str, condition: str, effect: str,
                 modules: Dict[str, Module], vocab: Vocabulary) -> Action:
        """
        Add an action from rule specification.
        
        Parameters
        ----------
        name : str
            Action name
        condition : str
            Condition expression
        effect : str
            Effect expression
        modules : Dict[str, Module]
            Available modules
        vocab : Vocabulary
            Symbol vocabulary
            
        Returns
        -------
        Action
            Created action
        """
        rule = ActionRule(name, condition, effect, modules, vocab)
        action = rule.compile()
        self.add_action(action)
        return action
    
    def update(self, dt: float):
        """Update all components."""
        self.basal_ganglia.update(dt)
        self.thalamus.update(dt)
        self.cortex.update(dt)
    
    def get_selected_action(self) -> Optional[Action]:
        """Get currently selected action."""
        return self.basal_ganglia.get_selected_action()
    
    @property
    def utilities(self) -> np.ndarray:
        """Get current utility values."""
        return self.basal_ganglia.utilities
    
    @property
    def activations(self) -> np.ndarray:
        """Get current activation values."""
        return self.basal_ganglia.activations