"""
Binding operations for Vector Symbolic Architectures.

This module implements various binding operations including XOR for binary vectors,
element-wise multiplication, circular convolution, and the MAP (Multiply-Add-Permute)
operation.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
from scipy import signal

from .core import VectorType

logger = logging.getLogger(__name__)


class BindingOperation(ABC):
    """
    Abstract base class for VSA binding operations.
    
    Each binding operation must be reversible (have an unbind operation)
    and should preserve the distributional properties of vectors.
    """
    
    def __init__(self, vector_type: VectorType, dimension: int):
        """
        Initialize binding operation.
        
        Parameters
        ----------
        vector_type : VectorType
            Type of vectors this operation works with
        dimension : int
            Vector dimension
        """
        self.vector_type = vector_type
        self.dimension = dimension
    
    @abstractmethod
    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bind two vectors together."""
        pass
    
    @abstractmethod
    def unbind(self, xy: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Unbind a vector from a bound pair."""
        pass
    
    @abstractmethod
    def is_commutative(self) -> bool:
        """Return True if bind(x,y) == bind(y,x)."""
        pass
    
    @abstractmethod
    def identity(self) -> np.ndarray:
        """Return the identity vector for this operation."""
        pass


class XORBinding(BindingOperation):
    """
    XOR binding for binary vectors.
    
    Properties:
    - Self-inverse: unbind is the same as bind
    - Commutative
    - Distributes over bundling
    - Hardware-friendly
    """
    
    def __init__(self, vector_type: VectorType, dimension: int):
        """Initialize XOR binding."""
        super().__init__(vector_type, dimension)
        if vector_type != VectorType.BINARY:
            logger.warning(f"XOR binding is designed for binary vectors, "
                         f"got {vector_type}. Will convert.")
    
    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """XOR two binary vectors."""
        # Ensure binary
        x_binary = (x > 0.5).astype(np.uint8) if x.dtype != np.uint8 else x
        y_binary = (y > 0.5).astype(np.uint8) if y.dtype != np.uint8 else y
        return np.bitwise_xor(x_binary, y_binary)
    
    def unbind(self, xy: np.ndarray, y: np.ndarray) -> np.ndarray:
        """XOR is self-inverse."""
        return self.bind(xy, y)
    
    def is_commutative(self) -> bool:
        """XOR is commutative."""
        return True
    
    def identity(self) -> np.ndarray:
        """Identity for XOR is all zeros."""
        return np.zeros(self.dimension, dtype=np.uint8)


class MultiplicationBinding(BindingOperation):
    """
    Element-wise multiplication binding.
    
    Properties:
    - Works with bipolar, ternary, complex, and real vectors
    - Commutative
    - For bipolar: self-inverse
    - For complex: unbind uses conjugate
    """
    
    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Element-wise multiplication."""
        return x * y
    
    def unbind(self, xy: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Unbind by multiplication with inverse.
        
        For bipolar/ternary: multiply (self-inverse)
        For complex: multiply by conjugate
        For real: element-wise division
        """
        if self.vector_type in [VectorType.BIPOLAR, VectorType.TERNARY]:
            # Self-inverse for {-1, +1} elements
            return xy * y
        elif self.vector_type == VectorType.COMPLEX:
            # Multiply by conjugate
            return xy * np.conj(y)
        else:
            # Element-wise division with protection
            with np.errstate(divide='ignore', invalid='ignore'):
                result = xy / y
                result[~np.isfinite(result)] = 0
            return result
    
    def is_commutative(self) -> bool:
        """Multiplication is commutative."""
        return True
    
    def identity(self) -> np.ndarray:
        """Identity for multiplication is all ones."""
        if self.vector_type == VectorType.COMPLEX:
            return np.ones(self.dimension, dtype=np.complex64)
        elif self.vector_type == VectorType.BINARY:
            return np.ones(self.dimension, dtype=np.uint8)
        else:
            return np.ones(self.dimension, dtype=np.float32)


class ConvolutionBinding(BindingOperation):
    """
    Circular convolution binding (HRR-style).
    
    Properties:
    - Non-commutative (unless made commutative)
    - Unbind uses circular correlation
    - Preserves similarity structure
    - Can be made efficient with FFT
    """
    
    def __init__(self, vector_type: VectorType, dimension: int, 
                 use_fft: bool = True):
        """
        Initialize convolution binding.
        
        Parameters
        ----------
        vector_type : VectorType
            Vector type (works best with real/bipolar)
        dimension : int
            Vector dimension
        use_fft : bool
            Whether to use FFT for efficiency
        """
        super().__init__(vector_type, dimension)
        self.use_fft = use_fft and dimension > 128
    
    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Circular convolution."""
        if self.use_fft:
            # FFT-based convolution
            return np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(y)))
        else:
            # Direct convolution
            return signal.fftconvolve(x, y, mode='same')
    
    def unbind(self, xy: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Circular correlation (reverse y and convolve)."""
        y_reversed = np.concatenate([y[:1], y[1:][::-1]])
        return self.bind(xy, y_reversed)
    
    def is_commutative(self) -> bool:
        """Convolution is not generally commutative."""
        return False
    
    def identity(self) -> np.ndarray:
        """Identity is a single spike at position 0."""
        identity = np.zeros(self.dimension, dtype=np.float32)
        identity[0] = 1
        return identity


class MAPBinding(BindingOperation):
    """
    Multiply-Add-Permute binding operation.
    
    Combines multiplication, addition, and permutation for
    enhanced binding properties. Used in Gayler's MAP architecture.
    """
    
    def __init__(self, vector_type: VectorType, dimension: int,
                 permutation: Optional[np.ndarray] = None):
        """
        Initialize MAP binding.
        
        Parameters
        ----------
        vector_type : VectorType
            Vector type
        dimension : int
            Vector dimension
        permutation : np.ndarray, optional
            Permutation to use. If None, generates random permutation.
        """
        super().__init__(vector_type, dimension)
        
        if permutation is None:
            # Generate random permutation
            self.permutation = np.random.RandomState(42).permutation(dimension)
        else:
            self.permutation = permutation
        
        # Compute inverse permutation
        self.inverse_permutation = np.argsort(self.permutation)
    
    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        MAP binding: (x * y) + permute(x) + permute(y).
        
        This creates a richer binding with better properties.
        """
        # Element-wise multiplication
        product = x * y
        
        # Add permuted versions
        x_permuted = x[self.permutation]
        y_permuted = y[self.permutation]
        
        result = product + x_permuted + y_permuted
        
        # Normalize based on vector type
        if self.vector_type == VectorType.BINARY:
            return (result > 1.5).astype(np.uint8)
        elif self.vector_type in [VectorType.BIPOLAR, VectorType.TERNARY]:
            return np.sign(result)
        else:
            return result / np.sqrt(3)  # Preserve variance
    
    def unbind(self, xy: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Approximate unbinding for MAP.
        
        This is not exact but provides good approximation.
        """
        # Subtract permuted y
        y_permuted = y[self.permutation]
        partial = xy - y_permuted
        
        # Approximate x from the result
        # This is simplified; full MAP unbinding is complex
        if self.vector_type in [VectorType.BIPOLAR, VectorType.TERNARY]:
            # For discrete values, use majority voting
            return np.sign(partial - y)
        else:
            # For continuous, approximate
            return (partial - y) / (1 + y)
    
    def is_commutative(self) -> bool:
        """MAP is commutative."""
        return True
    
    def identity(self) -> np.ndarray:
        """No simple identity for MAP."""
        raise NotImplementedError("MAP binding has no simple identity vector")


class PermutationBinding(BindingOperation):
    """
    Permutation-based binding.
    
    Binds by permuting one vector based on the other.
    Used in some VSA architectures for non-commutative binding.
    """
    
    def __init__(self, vector_type: VectorType, dimension: int):
        """Initialize permutation binding."""
        super().__init__(vector_type, dimension)
        # Pre-compute some random permutations
        rng = np.random.RandomState(42)
        self.n_perms = 256
        self.permutations = [rng.permutation(dimension) for _ in range(self.n_perms)]
    
    def _vector_to_permutation(self, vector: np.ndarray) -> np.ndarray:
        """Convert a vector to a permutation index."""
        # Hash vector to select permutation
        if self.vector_type == VectorType.BINARY:
            index = int(np.sum(vector * np.arange(len(vector))) % self.n_perms)
        else:
            index = int(abs(np.sum(vector)) * 1000) % self.n_perms
        return self.permutations[index]
    
    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bind by permuting x according to y."""
        perm = self._vector_to_permutation(y)
        return x[perm]
    
    def unbind(self, xy: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Unbind by inverse permutation."""
        perm = self._vector_to_permutation(y)
        inverse_perm = np.argsort(perm)
        return xy[inverse_perm]
    
    def is_commutative(self) -> bool:
        """Permutation binding is not commutative."""
        return False
    
    def identity(self) -> np.ndarray:
        """Identity would be a vector that maps to identity permutation."""
        raise NotImplementedError("Permutation binding has no simple identity")


def create_binding(binding_type: str,
                   vector_type: Union[VectorType, str],
                   dimension: int,
                   **kwargs) -> BindingOperation:
    """
    Create a binding operation of the specified type.
    
    Parameters
    ----------
    binding_type : str
        Type of binding: 'xor', 'multiplication', 'convolution', 'map', 'permutation'
    vector_type : VectorType or str
        Type of vectors
    dimension : int
        Vector dimension
    **kwargs
        Additional arguments for specific binding types
        
    Returns
    -------
    BindingOperation
        Binding operation instance
        
    Examples
    --------
    >>> # XOR binding for binary vectors
    >>> xor_bind = create_binding('xor', 'binary', 1024)
    
    >>> # Multiplication binding for bipolar
    >>> mult_bind = create_binding('multiplication', 'bipolar', 1024)
    
    >>> # MAP binding with custom permutation
    >>> map_bind = create_binding('map', 'bipolar', 1024)
    """
    if isinstance(vector_type, str):
        vector_type = VectorType(vector_type.lower())
    
    binding_type = binding_type.lower()
    
    if binding_type == 'xor':
        return XORBinding(vector_type, dimension)
    elif binding_type == 'multiplication':
        return MultiplicationBinding(vector_type, dimension)
    elif binding_type == 'convolution':
        return ConvolutionBinding(vector_type, dimension, **kwargs)
    elif binding_type == 'map':
        return MAPBinding(vector_type, dimension, **kwargs)
    elif binding_type == 'permutation':
        return PermutationBinding(vector_type, dimension)
    else:
        raise ValueError(f"Unknown binding type: {binding_type}")