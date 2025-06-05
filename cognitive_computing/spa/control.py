"""
Cognitive control mechanisms for Semantic Pointer Architecture.

This module implements executive control functions including working memory
control, attention routing, task switching, and sequential behavior. These
build on top of the action selection system to provide higher-level control.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np

from .core import SemanticPointer, Vocabulary, SPAConfig
from .modules import Module, State, Buffer, Gate, Connection
from .actions import Action, ActionSet, BasalGanglia, Thalamus

logger = logging.getLogger(__name__)


class CognitiveControl(Module):
    """
    Executive control over SPA modules.
    
    Provides high-level control mechanisms including attention, 
    working memory management, and task coordination.
    
    Parameters
    ----------
    dimensions : int
        Dimensionality of control vectors
    config : SPAConfig
        Configuration parameters
    vocab : Vocabulary, optional
        Vocabulary for control signals
    """
    
    def __init__(self, dimensions: int, config: SPAConfig, 
                 vocab: Optional[Vocabulary] = None):
        """Initialize cognitive control."""
        super().__init__("CognitiveControl", dimensions)
        self.config = config
        self.vocab = vocab if vocab is not None else Vocabulary(dimensions)
        
        # Control signals
        self._attention = np.zeros(dimensions)
        self._task_state = np.zeros(dimensions)
        self._goal_state = np.zeros(dimensions)
        
        # Working memory buffers
        self.working_memory: Dict[str, Buffer] = {}
        
        # Task switching
        self.current_task: Optional[str] = None
        self.task_stack: List[str] = []
        
        # Performance monitoring
        self.error_signal = 0.0
        self.conflict_level = 0.0
        
    @property
    def attention(self) -> np.ndarray:
        """Get current attention state."""
        return self._attention.copy()
        
    @property
    def task_state(self) -> np.ndarray:
        """Get current task state."""
        return self._task_state.copy()
        
    @property
    def goal_state(self) -> np.ndarray:
        """Get current goal state."""
        return self._goal_state.copy()
    
    def set_attention(self, target: Union[str, np.ndarray]):
        """
        Set attention to a specific target.
        
        Parameters
        ----------
        target : str or array
            Symbol name or vector to attend to
        """
        if isinstance(target, str):
            self._attention = self.vocab.parse(target).vector
        else:
            self._attention = np.array(target)
            
    def set_goal(self, goal: Union[str, np.ndarray]):
        """
        Set the current goal state.
        
        Parameters
        ----------
        goal : str or array
            Goal symbol or vector
        """
        if isinstance(goal, str):
            self._goal_state = self.vocab.parse(goal).vector
        else:
            self._goal_state = np.array(goal)
            
    def push_task(self, task: str):
        """
        Push a new task onto the task stack.
        
        Parameters
        ----------
        task : str
            Task identifier
        """
        if self.current_task is not None:
            self.task_stack.append(self.current_task)
        self.current_task = task
        self._task_state = self.vocab.parse(task).vector
        
    def pop_task(self) -> Optional[str]:
        """
        Pop a task from the task stack and make it current.
        
        Returns
        -------
        str or None
            Previous task if any
        """
        previous = self.current_task
        if self.task_stack:
            self.current_task = self.task_stack.pop()
            self._task_state = self.vocab.parse(self.current_task).vector
        else:
            self.current_task = None
            self._task_state = np.zeros(self.dimensions)
        return previous
        
    def add_working_memory(self, name: str, dimensions: Optional[int] = None):
        """
        Add a working memory buffer.
        
        Parameters
        ----------
        name : str
            Buffer name
        dimensions : int, optional
            Buffer dimensions (defaults to control dimensions)
        """
        if dimensions is None:
            dimensions = self.dimensions
        self.working_memory[name] = Buffer(name, dimensions, self.config)
        
    def update_conflict(self, utilities: np.ndarray):
        """
        Update conflict monitoring based on action utilities.
        
        Parameters
        ----------
        utilities : array
            Utility values from action selection
        """
        # Conflict is high when multiple actions have similar high utilities
        if len(utilities) > 1:
            sorted_utils = np.sort(utilities)[::-1]
            if sorted_utils[0] > self.config.threshold:
                # Conflict = ratio of second best to best
                self.conflict_level = sorted_utils[1] / sorted_utils[0]
            else:
                self.conflict_level = 0.0
        else:
            self.conflict_level = 0.0
            
    def update_error(self, expected: np.ndarray, actual: np.ndarray):
        """
        Update error monitoring.
        
        Parameters
        ----------
        expected : array
            Expected state
        actual : array
            Actual state
        """
        # Error is distance between expected and actual
        diff = expected - actual
        self.error_signal = np.linalg.norm(diff) / np.sqrt(len(diff))
        
    def update(self, dt: float):
        """Update cognitive control state."""
        # Update working memory buffers
        for buffer in self.working_memory.values():
            buffer.update(dt)
            
        # Decay error and conflict signals
        self.error_signal *= (1.0 - dt / self.config.synapse)
        self.conflict_level *= (1.0 - dt / self.config.synapse)
        
        # Update state based on attention, task, and goal
        self._state = (self._attention + self._task_state + self._goal_state) / 3.0


class Routing(Module):
    """
    Dynamic routing of information between modules.
    
    Implements flexible routing that can be controlled by cognitive control
    signals, allowing dynamic reconfiguration of information flow.
    
    Parameters
    ----------
    dimensions : int
        Dimensionality of routed signals
    config : SPAConfig
        Configuration parameters
    """
    
    def __init__(self, dimensions: int, config: SPAConfig):
        """Initialize routing."""
        super().__init__("Routing", dimensions)
        self.config = config
        
        # Routing matrix (from x to)
        self.routes: Dict[Tuple[str, str], Gate] = {}
        
        # Default routes
        self.default_routes: Dict[str, str] = {}
        
    def add_route(self, source: str, target: str, 
                  gate: Optional[Gate] = None) -> Gate:
        """
        Add a route between modules.
        
        Parameters
        ----------
        source : str
            Source module name
        target : str
            Target module name
        gate : Gate, optional
            Gate controlling this route
            
        Returns
        -------
        Gate
            The gate controlling this route
        """
        if gate is None:
            gate = Gate(f"{source}_to_{target}", self.dimensions, self.config)
        self.routes[(source, target)] = gate
        return gate
        
    def set_default_route(self, source: str, target: str):
        """Set default routing for a source."""
        self.default_routes[source] = target
        
    def route(self, source: str, signal: np.ndarray, 
              target: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Route a signal from source to target(s).
        
        Parameters
        ----------
        source : str
            Source module name
        signal : array
            Signal to route
        target : str, optional
            Specific target (uses default if not specified)
            
        Returns
        -------
        dict
            Routed signals by target name
        """
        outputs = {}
        
        if target is not None:
            # Route to specific target
            key = (source, target)
            if key in self.routes:
                gate = self.routes[key]
                # Gate applies its signal to the input
                if hasattr(gate, '_gate_signal'):
                    if isinstance(gate._gate_signal, float):
                        outputs[target] = signal * gate._gate_signal
                    else:
                        outputs[target] = signal * gate._gate_signal
                else:
                    outputs[target] = signal
        else:
            # Route to all connected targets
            for (src, tgt), gate in self.routes.items():
                if src == source:
                    # Gate applies its signal to the input
                    if hasattr(gate, '_gate_signal'):
                        if isinstance(gate._gate_signal, float):
                            outputs[tgt] = signal * gate._gate_signal
                        else:
                            outputs[tgt] = signal * gate._gate_signal
                    else:
                        outputs[tgt] = signal
                    
            # Use default if no specific routes
            if not outputs and source in self.default_routes:
                target = self.default_routes[source]
                outputs[target] = signal
                
        return outputs
        
    def update(self, dt: float):
        """Update routing gates."""
        for gate in self.routes.vectoralues():
            gate.update(dt)


class Gating(Module):
    """
    Gating control for information flow.
    
    Provides mechanisms for controlling when and how information
    flows between modules based on control signals.
    
    Parameters
    ----------
    dimensions : int
        Dimensionality of gated signals
    config : SPAConfig
        Configuration parameters
    """
    
    def __init__(self, dimensions: int, config: SPAConfig):
        """Initialize gating control."""
        super().__init__("Gating", dimensions)
        self.config = config
        
        # Gates by name
        self.gates: Dict[str, Gate] = {}
        
        # Gate groups for coordinated control
        self.gate_groups: Dict[str, List[str]] = {}
        
    def add_gate(self, name: str, gate: Optional[Gate] = None) -> Gate:
        """
        Add a gate.
        
        Parameters
        ----------
        name : str
            Gate name
        gate : Gate, optional
            Gate object (creates new if not provided)
            
        Returns
        -------
        Gate
            The gate object
        """
        if gate is None:
            gate = Gate(name, self.dimensions, self.config)
        # Add control property for testing
        gate.control = 0.0
        self.gates[name] = gate
        return gate
        
    def add_gate_group(self, group_name: str, gate_names: List[str]):
        """
        Create a group of gates for coordinated control.
        
        Parameters
        ----------
        group_name : str
            Name of the group
        gate_names : list of str
            Names of gates in the group
        """
        self.gate_groups[group_name] = gate_names
        
    def open_gate(self, name: str, amount: float = 1.0):
        """
        Open a gate or gate group.
        
        Parameters
        ----------
        name : str
            Gate or group name
        amount : float
            How much to open (0-1)
        """
        if name in self.gates:
            self.gates[name].set_gate(amount)
            # Store control value for testing
            self.gates[name].control = amount
        elif name in self.gate_groups:
            for gate_name in self.gate_groups[name]:
                if gate_name in self.gates:
                    self.gates[gate_name].set_gate(amount)
                    self.gates[gate_name].control = amount
                    
    def close_gate(self, name: str):
        """
        Close a gate or gate group.
        
        Parameters
        ----------
        name : str
            Gate or group name
        """
        if name in self.gates:
            self.gates[name].set_gate(0.0)
            self.gates[name].control = 0.0
        elif name in self.gate_groups:
            for gate_name in self.gate_groups[name]:
                if gate_name in self.gates:
                    self.gates[gate_name].set_gate(0.0)
                    self.gates[gate_name].control = 0.0
                    
    def update(self, dt: float):
        """Update all gates."""
        for gate in self.gates.values():
            gate.update(dt)


class Sequencing(Module):
    """
    Sequential behavior control.
    
    Implements mechanisms for executing sequences of actions,
    including loops, conditionals, and interruption handling.
    
    Parameters
    ----------
    dimensions : int
        Dimensionality of sequence states
    config : SPAConfig
        Configuration parameters
    vocab : Vocabulary
        Vocabulary for sequence elements
    """
    
    def __init__(self, dimensions: int, config: SPAConfig, vocab: Vocabulary):
        """Initialize sequencing."""
        super().__init__("Sequencing", dimensions)
        self.config = config
        self.vocab = vocab
        
        # Sequence definitions
        self.sequences: Dict[str, List[Union[str, Callable]]] = {}
        
        # Current sequence state
        self.current_sequence: Optional[str] = None
        self.sequence_index = 0
        self.sequence_state = np.zeros(dimensions)
        
        # Sequence control
        self.paused = False
        self.loop_count = 0
        self.max_loops = 10
        
        # Interruption handling
        self.interrupt_stack: List[Tuple[str, int]] = []
        
    def define_sequence(self, name: str, steps: List[Union[str, Callable]]):
        """
        Define a sequence of steps.
        
        Parameters
        ----------
        name : str
            Sequence name
        steps : list
            List of step names or callables
        """
        self.sequences[name] = steps
        
    def start_sequence(self, name: str):
        """
        Start executing a sequence.
        
        Parameters
        ----------
        name : str
            Sequence name
        """
        if name not in self.sequences:
            raise ValueError(f"Unknown sequence: {name}")
            
        # Save current if needed
        if self.current_sequence is not None:
            self.interrupt_stack.append((self.current_sequence, self.sequence_index))
            
        self.current_sequence = name
        self.sequence_index = 0
        self.loop_count = 0
        self.paused = False
        self._update_sequence_state()
        
    def pause_sequence(self):
        """Pause the current sequence."""
        self.paused = True
        
    def resume_sequence(self):
        """Resume the current sequence."""
        self.paused = False
        
    def stop_sequence(self):
        """Stop the current sequence."""
        self.current_sequence = None
        self.sequence_index = 0
        self.sequence_state = np.zeros(self.dimensions)
        
        # Resume interrupted sequence if any
        if self.interrupt_stack:
            name, index = self.interrupt_stack.pop()
            self.current_sequence = name
            self.sequence_index = index
            self._update_sequence_state()
            
    def next_step(self) -> Optional[Union[str, Callable]]:
        """
        Get the next step in the sequence.
        
        Returns
        -------
        str or callable or None
            Next step or None if sequence complete
        """
        if self.current_sequence is None or self.paused:
            return None
            
        steps = self.sequences[self.current_sequence]
        
        if self.sequence_index >= len(steps):
            # Check for looping
            if self.loop_count < self.max_loops:
                self.sequence_index = 0
                self.loop_count += 1
            else:
                self.stop_sequence()
                return None
                
        step = steps[self.sequence_index]
        self.sequence_index += 1
        self._update_sequence_state()
        
        return step
        
    def _update_sequence_state(self):
        """Update the sequence state vector."""
        if self.current_sequence is None:
            self.sequence_state = np.zeros(self.dimensions)
        else:
            # Encode current sequence and position
            # Create pointer if it doesn't exist
            if self.current_sequence not in self.vocab.pointers:
                self.vocab.create_pointer(self.current_sequence)
            seq_ptr = self.vocab.parse(self.current_sequence).vector
            
            if self.sequence_index > 0:
                step_name = f"STEP_{self.sequence_index}"
                if step_name not in self.vocab.pointers:
                    self.vocab.create_pointer(step_name)
                pos_ptr = self.vocab.parse(step_name).vector
                self.sequence_state = 0.7 * seq_ptr + 0.3 * pos_ptr
            else:
                self.sequence_state = seq_ptr
                
    def update(self, dt: float):
        """Update sequence state."""
        self._state = self.sequence_state