"""
Neural network implementation for Semantic Pointer Architecture.

This module provides NEF (Neural Engineering Framework) style networks
for implementing SPA operations with biologically plausible neural populations.
These are not traditional deep learning networks, but rather principled
neural implementations of vector operations.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np
from scipy import signal as scipy_signal

from .core import SemanticPointer, Vocabulary, SPAConfig
from .modules import Module, Connection as ModuleConnection

logger = logging.getLogger(__name__)


@dataclass
class NeuronParams:
    """
    Parameters for a population of neurons.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons in the population
    tau_rc : float
        Membrane RC time constant in seconds (default: 0.02)
    tau_ref : float
        Refractory period in seconds (default: 0.002)
    max_rates : Tuple[float, float]
        Range of maximum firing rates in Hz (default: (200, 400))
    intercepts : Tuple[float, float]
        Range of x-intercepts for tuning curves (default: (-1, 1))
    noise : float
        Amount of noise to add to neural activities (default: 0.1)
    """
    n_neurons: int
    tau_rc: float = 0.02
    tau_ref: float = 0.002
    max_rates: Tuple[float, float] = (200.0, 400.0)
    intercepts: Tuple[float, float] = (-1.0, 1.0)
    noise: float = 0.1
    
    def __post_init__(self):
        """Validate neuron parameters."""
        if self.n_neurons <= 0:
            raise ValueError(f"n_neurons must be positive, got {self.n_neurons}")
        if self.tau_rc <= 0:
            raise ValueError(f"tau_rc must be positive, got {self.tau_rc}")
        if self.tau_ref <= 0:
            raise ValueError(f"tau_ref must be positive, got {self.tau_ref}")
        if self.noise < 0:
            raise ValueError(f"noise must be non-negative, got {self.noise}")


class Ensemble:
    """
    A population of neurons representing a value or vector.
    
    In NEF, ensembles are groups of neurons that collectively represent
    values through their firing patterns. Each neuron has a preferred
    direction vector (encoder) and responds based on the dot product
    of its encoder with the represented value.
    
    Parameters
    ----------
    name : str
        Name of the ensemble
    dimensions : int
        Dimensionality of the represented value
    neurons : NeuronParams
        Parameters for the neural population
    radius : float
        Radius of the representation space (default: 1.0)
    seed : Optional[int]
        Random seed for reproducibility
    """
    
    def __init__(self, name: str, dimensions: int, neurons: NeuronParams,
                 radius: float = 1.0, seed: Optional[int] = None):
        """Initialize ensemble."""
        self.name = name
        self.dimensions = dimensions
        self.neurons = neurons
        self.radius = radius
        
        # Random number generator
        self.rng = np.random.RandomState(seed)
        
        # Neural parameters
        self.n_neurons = neurons.n_neurons
        
        # Generate encoders (preferred direction vectors)
        self.encoders = self._generate_encoders()
        
        # Generate gains and biases for neurons
        self.gains, self.biases = self._generate_tuning_curves()
        
        # Current state
        self._value = np.zeros(dimensions)
        self._activities = np.zeros(self.n_neurons)
        self._rates = np.zeros(self.n_neurons)
        
    def _generate_encoders(self) -> np.ndarray:
        """
        Generate encoder vectors (preferred directions) for neurons.
        
        Returns
        -------
        np.ndarray
            Encoder matrix of shape (n_neurons, dimensions)
        """
        # Generate random vectors
        encoders = self.rng.randn(self.n_neurons, self.dimensions)
        
        # Normalize to unit length
        norms = np.linalg.norm(encoders, axis=1, keepdims=True)
        encoders = encoders / norms
        
        return encoders
    
    def _generate_tuning_curves(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate gains and biases for neural tuning curves.
        
        Returns
        -------
        gains : np.ndarray
            Gain for each neuron
        biases : np.ndarray
            Bias current for each neuron
        """
        # Sample intercepts and max rates
        intercepts = self.rng.uniform(
            self.neurons.intercepts[0],
            self.neurons.intercepts[1],
            self.n_neurons
        )
        max_rates = self.rng.uniform(
            self.neurons.max_rates[0],
            self.neurons.max_rates[1],
            self.n_neurons
        )
        
        # Convert to gains and biases using NEF formulas
        # These formulas come from solving for neural parameters
        # that give desired intercepts and max rates
        tau_rc = self.neurons.tau_rc
        tau_ref = self.neurons.tau_ref
        
        # Calculate gains and biases
        # For LIF neurons, we need to handle the cases where intercepts might cause issues
        gains = np.zeros(self.n_neurons)
        biases = np.zeros(self.n_neurons)
        
        for i in range(self.n_neurons):
            intercept = intercepts[i]
            max_rate = max_rates[i]
            
            # Calculate rate at intercept
            if intercept > 1.0:
                # No firing at intercept
                rate_at_intercept = 0.0
            else:
                # Calculate current at intercept (when neuron starts firing)
                j_threshold = 1.0 / (1 - intercept) if intercept < 1.0 else np.inf
                if j_threshold > 1.0:
                    # Calculate firing rate using LIF formula
                    rate_at_intercept = 1.0 / (tau_ref + tau_rc * np.log(1 + 1.0 / (j_threshold - 1)))
                else:
                    rate_at_intercept = 0.0
            
            # Calculate gain and bias
            # We need to ensure the neuron fires at max_rate when input is radius
            # For LIF neurons: rate = 1/(tau_ref + tau_rc * log(1 + 1/(J-1)))
            # where J = gain * x + bias
            # At x = radius, we want rate = max_rate
            
            if self.radius > intercept:
                # Binary search for gain that gives desired max rate
                gain_low = 0.0
                gain_high = 1000.0
                
                for _ in range(20):  # Binary search iterations
                    gain_mid = (gain_low + gain_high) / 2.0
                    bias_mid = 1.0 - gain_mid * intercept  # Bias to make J=1 at intercept
                    
                    # Calculate rate at radius
                    j_at_radius = gain_mid * self.radius + bias_mid
                    if j_at_radius > 1.0:
                        rate_at_radius = 1.0 / (tau_ref + tau_rc * np.log(1 + 1.0 / (j_at_radius - 1)))
                    else:
                        rate_at_radius = 0.0
                    
                    if rate_at_radius < max_rate:
                        gain_low = gain_mid
                    else:
                        gain_high = gain_mid
                
                gains[i] = (gain_low + gain_high) / 2.0
                biases[i] = 1.0 - gains[i] * intercept
            else:
                # Shouldn't happen with proper intercepts, but handle gracefully
                gains[i] = max_rate / self.radius
                biases[i] = 0.0
        
        return gains, biases
    
    def encode(self, value: np.ndarray) -> np.ndarray:
        """
        Encode a value into neural activities.
        
        Parameters
        ----------
        value : np.ndarray
            Value to encode
            
        Returns
        -------
        np.ndarray
            Neural activities
        """
        if value.shape != (self.dimensions,):
            raise ValueError(f"Value shape {value.shape} doesn't match "
                           f"ensemble dimensions {self.dimensions}")
        
        # Store value
        self._value = value.copy()
        
        # Calculate input currents (dot product with encoders)
        currents = np.dot(self.encoders, value) * self.gains + self.biases
        
        # Apply neural nonlinearity (LIF neuron model)
        self._rates = self._lif_rate(currents)
        
        # Add noise
        if self.neurons.noise > 0:
            noise = self.rng.normal(0, self.neurons.noise * np.max(self._rates),
                                  size=self.n_neurons)
            self._rates = np.maximum(0, self._rates + noise)
        
        # Convert rates to activities (normalized by max rate)
        max_rate = self.neurons.max_rates[1]
        self._activities = self._rates / max_rate
        
        return self._activities
    
    def _lif_rate(self, current: np.ndarray) -> np.ndarray:
        """
        Leaky Integrate-and-Fire neuron rate response.
        
        Parameters
        ----------
        current : np.ndarray
            Input current to each neuron
            
        Returns
        -------
        np.ndarray
            Firing rates in Hz
        """
        tau_rc = self.neurons.tau_rc
        tau_ref = self.neurons.tau_ref
        
        # LIF rate formula
        rate = np.zeros_like(current)
        mask = current > 1
        rate[mask] = 1.0 / (tau_ref + tau_rc * np.log(1 + 1.0 / (current[mask] - 1)))
        
        return rate
    
    def decode(self, activities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decode neural activities into a value.
        
        Parameters
        ----------
        activities : np.ndarray, optional
            Activities to decode. If None, uses current activities
            
        Returns
        -------
        np.ndarray
            Decoded value
        """
        if activities is None:
            activities = self._activities
            
        # Simple linear decoding (could be optimized with decoders)
        # For now, use least-squares approximation
        # In full NEF, decoders are precomputed to minimize error
        
        # Convert activities back to rates
        rates = activities * self.neurons.max_rates[1]
        
        # Weighted sum of encoders by rates, normalized by total rate
        total_rate = np.sum(rates)
        if total_rate > 0:
            value = np.dot(rates, self.encoders) / total_rate * self.radius
        else:
            value = np.zeros(self.dimensions)
        
        return value
    
    @property
    def activities(self) -> np.ndarray:
        """Get current neural activities."""
        return self._activities.copy()
    
    @property
    def rates(self) -> np.ndarray:
        """Get current firing rates."""
        return self._rates.copy()
    
    @property
    def value(self) -> np.ndarray:
        """Get current represented value."""
        return self._value.copy()


class EnsembleArray:
    """
    Array of ensembles representing a high-dimensional vector.
    
    For high-dimensional vectors, we use multiple ensembles each
    representing a subset of dimensions. This is more efficient
    than having one ensemble with many dimensions.
    
    Parameters
    ----------
    name : str
        Name of the ensemble array
    n_ensembles : int
        Number of ensembles in the array
    dimensions_per_ensemble : int
        Dimensions represented by each ensemble
    neurons_per_ensemble : int
        Number of neurons in each ensemble
    radius : float
        Radius of representation space
    seed : Optional[int]
        Random seed
    """
    
    def __init__(self, name: str, n_ensembles: int,
                 dimensions_per_ensemble: int,
                 neurons_per_ensemble: int = 50,
                 radius: float = 1.0,
                 seed: Optional[int] = None):
        """Initialize ensemble array."""
        self.name = name
        self.n_ensembles = n_ensembles
        self.dimensions_per_ensemble = dimensions_per_ensemble
        self.total_dimensions = n_ensembles * dimensions_per_ensemble
        
        # Create ensembles
        self.ensembles = []
        for i in range(n_ensembles):
            ens_seed = None if seed is None else seed + i
            neurons = NeuronParams(neurons_per_ensemble)
            ensemble = Ensemble(
                f"{name}_{i}",
                dimensions_per_ensemble,
                neurons,
                radius,
                ens_seed
            )
            self.ensembles.append(ensemble)
    
    def encode(self, value: np.ndarray) -> np.ndarray:
        """
        Encode a value into neural activities.
        
        Parameters
        ----------
        value : np.ndarray
            Value to encode
            
        Returns
        -------
        np.ndarray
            Concatenated neural activities from all ensembles
        """
        if value.shape != (self.total_dimensions,):
            raise ValueError(f"Value shape {value.shape} doesn't match "
                           f"total dimensions {self.total_dimensions}")
        
        # Split value among ensembles
        activities = []
        for i, ensemble in enumerate(self.ensembles):
            start = i * self.dimensions_per_ensemble
            end = (i + 1) * self.dimensions_per_ensemble
            sub_value = value[start:end]
            
            ens_activities = ensemble.encode(sub_value)
            activities.append(ens_activities)
        
        return np.concatenate(activities)
    
    def decode(self, activities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decode neural activities into a value.
        
        Parameters
        ----------
        activities : np.ndarray, optional
            Activities to decode
            
        Returns
        -------
        np.ndarray
            Decoded value
        """
        if activities is None:
            # Get current activities from all ensembles
            activities = self.get_activities()
        
        # Decode each ensemble
        values = []
        neurons_per_ens = self.ensembles[0].n_neurons
        
        for i, ensemble in enumerate(self.ensembles):
            start = i * neurons_per_ens
            end = (i + 1) * neurons_per_ens
            ens_activities = activities[start:end]
            
            sub_value = ensemble.decode(ens_activities)
            values.append(sub_value)
        
        return np.concatenate(values)
    
    def get_activities(self) -> np.ndarray:
        """Get all neural activities."""
        activities = []
        for ensemble in self.ensembles:
            activities.append(ensemble.activities)
        return np.concatenate(activities)
    
    @property
    def value(self) -> np.ndarray:
        """Get current represented value."""
        values = []
        for ensemble in self.ensembles:
            values.append(ensemble.value)
        return np.concatenate(values)


class Connection:
    """
    Connection between neural ensembles with optional transformation.
    
    Connections in NEF compute functions between ensembles. The connection
    weights are derived from the desired transformation function.
    
    Parameters
    ----------
    pre : Union[Ensemble, EnsembleArray]
        Pre-synaptic ensemble
    post : Union[Ensemble, EnsembleArray]
        Post-synaptic ensemble
    transform : Optional[np.ndarray]
        Transformation matrix (if None, identity)
    synapse : float
        Synaptic time constant
    function : Optional[Callable]
        Function to compute (if None, linear transform)
    """
    
    def __init__(self, pre: Union[Ensemble, EnsembleArray],
                 post: Union[Ensemble, EnsembleArray],
                 transform: Optional[np.ndarray] = None,
                 synapse: float = 0.005,
                 function: Optional[Callable] = None):
        """Initialize connection."""
        self.pre = pre
        self.post = post
        self.synapse = synapse
        self.function = function
        
        # Get dimensions
        self.pre_dims = (pre.dimensions if isinstance(pre, Ensemble) 
                        else pre.total_dimensions)
        self.post_dims = (post.dimensions if isinstance(post, Ensemble)
                         else post.total_dimensions)
        
        # Set transform
        if transform is None:
            # Check dimension compatibility
            if self.pre_dims != self.post_dims:
                if function is not None:
                    raise ValueError(f"Transform required when using function with "
                                   f"dimension change: {self.pre_dims} -> {self.post_dims}")
                else:
                    raise ValueError(f"Dimensions mismatch without transform: "
                                   f"{self.pre_dims} != {self.post_dims}")
            self.transform = np.eye(self.post_dims)
        else:
            # With function, transform applies to function output
            if function is not None:
                # Transform should map from function output to post dimensions
                if transform.shape[0] != self.post_dims:
                    raise ValueError(f"Transform output dimension {transform.shape[0]} "
                                   f"doesn't match post dimension {self.post_dims}")
            else:
                # Without function, transform maps from pre to post
                if transform.shape != (self.post_dims, self.pre_dims):
                    raise ValueError(f"Transform shape {transform.shape} doesn't "
                                   f"match expected ({self.post_dims}, {self.pre_dims})")
            self.transform = transform
        
        # Synaptic filter state
        self._filtered = np.zeros(self.post_dims)
    
    def compute(self, dt: float = 0.001) -> np.ndarray:
        """
        Compute connection output.
        
        Parameters
        ----------
        dt : float
            Time step
            
        Returns
        -------
        np.ndarray
            Filtered and transformed output
        """
        # Get pre-synaptic value
        pre_value = self.pre.value
        
        # Apply function if specified
        if self.function is not None:
            pre_value = self.function(pre_value)
        
        # Apply transformation
        output = np.dot(self.transform, pre_value)
        
        # Apply synaptic filter
        if self.synapse > 0:
            decay = np.exp(-dt / self.synapse)
            self._filtered = decay * self._filtered + (1 - decay) * output
            return self._filtered
        else:
            return output


class Probe:
    """
    Probe for recording values from network components.
    
    Parameters
    ----------
    target : Union[Ensemble, EnsembleArray, Connection]
        Object to probe
    attr : str
        Attribute to record ('value', 'activities', 'rates', etc.)
    synapse : float
        Filter time constant for probe
    sample_every : float
        Sampling period in seconds
    """
    
    def __init__(self, target: Any, attr: str = "value",
                 synapse: float = 0.01, sample_every: float = 0.001):
        """Initialize probe."""
        self.target = target
        self.attr = attr
        self.synapse = synapse
        self.sample_every = sample_every
        
        # Recording storage
        self.times = []
        self.data = []
        
        # Filter state
        self._filtered = None
        self._last_sample_time = 0.0
    
    def record(self, time: float, dt: float = 0.001):
        """
        Record a sample if enough time has passed.
        
        Parameters
        ----------
        time : float
            Current simulation time
        dt : float
            Time step
        """
        # Check if we should sample (always sample at t=0)
        if time > 0 and time - self._last_sample_time < self.sample_every:
            return
        
        # Get value to record
        if hasattr(self.target, self.attr):
            value = getattr(self.target, self.attr)
            if callable(value):
                value = value()
        else:
            raise ValueError(f"Target has no attribute '{self.attr}'")
        
        # Apply filter if needed
        if self.synapse > 0:
            if self._filtered is None:
                self._filtered = np.zeros_like(value)
            decay = np.exp(-dt / self.synapse)
            self._filtered = decay * self._filtered + (1 - decay) * value
            value = self._filtered.copy()
        
        # Store sample
        self.times.append(time)
        self.data.append(value.copy())
        self._last_sample_time = time
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get recorded data.
        
        Returns
        -------
        times : np.ndarray
            Time points
        data : np.ndarray
            Recorded data
        """
        return np.array(self.times), np.array(self.data)
    
    def clear(self):
        """Clear recorded data."""
        self.times.clear()
        self.data.clear()
        self._filtered = None
        self._last_sample_time = 0.0


class Network:
    """
    Container for ensembles, connections, and probes.
    
    The Network class manages the simulation of neural populations
    and their interactions.
    
    Parameters
    ----------
    label : str
        Name of the network
    dt : float
        Simulation timestep
    seed : Optional[int]
        Random seed
    """
    
    def __init__(self, label: str = "Network", dt: float = 0.001,
                 seed: Optional[int] = None):
        """Initialize network."""
        self.label = label
        self.dt = dt
        self.seed = seed
        
        # Components
        self.ensembles: Dict[str, Union[Ensemble, EnsembleArray]] = {}
        self.connections: List[Connection] = []
        self.probes: List[Probe] = []
        
        # Simulation time
        self.time = 0.0
    
    def add_ensemble(self, name: str, ensemble: Union[Ensemble, EnsembleArray]):
        """Add an ensemble to the network."""
        if name in self.ensembles:
            raise ValueError(f"Ensemble '{name}' already exists")
        self.ensembles[name] = ensemble
    
    def connect(self, pre: Union[str, Ensemble, EnsembleArray],
                post: Union[str, Ensemble, EnsembleArray],
                transform: Optional[np.ndarray] = None,
                synapse: float = 0.005,
                function: Optional[Callable] = None) -> Connection:
        """
        Create a connection between ensembles.
        
        Parameters
        ----------
        pre : str or Ensemble/EnsembleArray
            Pre-synaptic ensemble
        post : str or Ensemble/EnsembleArray
            Post-synaptic ensemble
        transform : np.ndarray, optional
            Transformation matrix
        synapse : float
            Synaptic time constant
        function : Callable, optional
            Function to compute
            
        Returns
        -------
        Connection
            Created connection
        """
        # Resolve string names
        if isinstance(pre, str):
            pre = self.ensembles[pre]
        if isinstance(post, str):
            post = self.ensembles[post]
        
        # Create connection
        conn = Connection(pre, post, transform, synapse, function)
        self.connections.append(conn)
        
        return conn
    
    def probe(self, target: Union[str, Any], attr: str = "value",
              synapse: float = 0.01, sample_every: float = None) -> Probe:
        """
        Create a probe to record data.
        
        Parameters
        ----------
        target : str or object
            Target to probe
        attr : str
            Attribute to record
        synapse : float
            Filter time constant
        sample_every : float
            Sampling period (if None, uses dt)
            
        Returns
        -------
        Probe
            Created probe
        """
        # Resolve string names
        if isinstance(target, str):
            target = self.ensembles[target]
        
        if sample_every is None:
            sample_every = self.dt
        
        # Create probe
        probe = Probe(target, attr, synapse, sample_every)
        self.probes.append(probe)
        
        return probe
    
    def step(self):
        """Run one timestep of simulation."""
        # Update connections
        for conn in self.connections:
            output = conn.compute(self.dt)
            
            # Apply to post ensemble
            # In full NEF, this would go through decoders/encoders
            # For simplicity, we directly set the value
            if isinstance(conn.post, Ensemble):
                conn.post.encode(output)
            else:  # EnsembleArray
                conn.post.encode(output)
        
        # Record probes
        for probe in self.probes:
            probe.record(self.time, self.dt)
        
        # Advance time
        self.time += self.dt
    
    def run(self, duration: float):
        """
        Run simulation for specified duration.
        
        Parameters
        ----------
        duration : float
            Simulation duration in seconds
        """
        steps = int(duration / self.dt)
        for _ in range(steps):
            self.step()
    
    def reset(self):
        """Reset network to initial state."""
        self.time = 0.0
        
        # Reset ensembles
        for ensemble in self.ensembles.values():
            if isinstance(ensemble, Ensemble):
                ensemble.encode(np.zeros(ensemble.dimensions))
            else:  # EnsembleArray
                ensemble.encode(np.zeros(ensemble.total_dimensions))
        
        # Clear probes
        for probe in self.probes:
            probe.clear()
        
        # Reset connection filters
        for conn in self.connections:
            conn._filtered = np.zeros(conn.post_dims)


class CircularConvolution:
    """
    Neural implementation of circular convolution for binding.
    
    This provides a way to implement the binding operation of HRR/SPA
    using neural populations instead of direct computation.
    
    Parameters
    ----------
    dimensions : int
        Dimensionality of vectors
    neurons_per_dimension : int
        Neurons per dimension in the ensembles
    radius : float
        Representation radius
    """
    
    def __init__(self, dimensions: int, neurons_per_dimension: int = 50,
                 radius: float = 1.0):
        """Initialize circular convolution network."""
        self.dimensions = dimensions
        self.neurons_per_dimension = neurons_per_dimension
        self.radius = radius
        
        # We'll use the Fourier transform approach:
        # conv(a,b) = IFFT(FFT(a) * FFT(b))
        # This requires complex-valued representations
        
        # For simplicity, we'll provide a functional implementation
        # In a full neural implementation, this would use complex-valued
        # ensembles or pairs of real ensembles for real/imaginary parts
    
    def convolve(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute circular convolution.
        
        Parameters
        ----------
        a, b : np.ndarray
            Input vectors
            
        Returns
        -------
        np.ndarray
            Convolution result
        """
        if a.shape != (self.dimensions,) or b.shape != (self.dimensions,):
            raise ValueError(f"Input dimensions must be {self.dimensions}")
        
        # Use FFT-based convolution
        fft_a = np.fft.fft(a)
        fft_b = np.fft.fft(b)
        fft_result = fft_a * fft_b
        result = np.real(np.fft.ifft(fft_result))
        
        return result
    
    def create_network(self, network: Network, 
                      input_a: str, input_b: str, output: str):
        """
        Create neural network for convolution.
        
        This is a simplified version. A full implementation would create
        ensembles for FFT computation, complex multiplication, and IFFT.
        
        Parameters
        ----------
        network : Network
            Network to add components to
        input_a, input_b : str
            Names of input ensemble arrays
        output : str
            Name of output ensemble array
        """
        # Get input ensembles
        ens_a = network.ensembles[input_a]
        ens_b = network.ensembles[input_b]
        
        # For now, create a functional connection
        # In full implementation, this would be multiple stages
        def conv_func(x):
            # x is concatenated [a; b]
            mid = len(x) // 2
            a = x[:mid]
            b = x[mid:]
            return self.convolve(a, b)
        
        # Would need intermediate ensembles in practice
        logger.info("CircularConvolution.create_network is simplified - "
                   "full neural implementation would require complex-valued "
                   "ensembles for FFT operations")