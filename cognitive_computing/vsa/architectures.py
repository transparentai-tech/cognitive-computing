"""
Specific VSA architecture implementations.

This module provides complete VSA architectures including Binary Spatter Codes (BSC),
Multiply-Add-Permute (MAP), Fourier HRR (FHRR), and others.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy.fft import fft, ifft

from .core import VSA, VSAConfig, VectorType, VSAType
from .vectors import create_vector
from .binding import create_binding
from .operations import permute, bundle, thin, normalize_vector

logger = logging.getLogger(__name__)


class BSC(VSA):
    """
    Binary Spatter Codes (BSC) implementation.
    
    BSC uses binary vectors with XOR binding and is optimized for
    hardware implementation. Key features:
    - Binary vectors {0, 1}
    - XOR for binding (self-inverse)
    - Majority voting for bundling
    - Hamming distance similarity
    """
    
    def __init__(self, dimension: int = 8192, 
                 sparsity: float = 0.0,
                 seed: Optional[int] = None):
        """
        Initialize Binary Spatter Codes.
        
        Parameters
        ----------
        dimension : int
            Vector dimension (typically 8192 or higher)
        sparsity : float
            Sparsity level for vectors
        seed : int, optional
            Random seed
        """
        config = VSAConfig(
            dimension=dimension,
            vector_type=VectorType.BINARY,
            vsa_type=VSAType.BSC,
            binding_method="xor",
            sparsity=sparsity,
            normalize_result=False,  # Binary vectors don't need normalization
            seed=seed
        )
        super().__init__(config)
        
        logger.info(f"Initialized BSC with dimension={dimension}")
    
    def protect(self, vector: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        Protect a vector using XOR with a key (BSC-specific operation).
        
        Parameters
        ----------
        vector : np.ndarray
            Vector to protect
        key : np.ndarray
            Protection key
            
        Returns
        -------
        np.ndarray
            Protected vector
        """
        return self.bind(vector, key)
    
    def make_unitary(self, vector: np.ndarray) -> np.ndarray:
        """
        BSC vectors are already unitary in the binary sense.
        
        Returns the vector unchanged.
        """
        return vector
    
    def consensus(self, vectors: List[np.ndarray],
                  threshold: float = 0.5) -> np.ndarray:
        """
        Find consensus using majority voting.
        
        Parameters
        ----------
        vectors : List[np.ndarray]
            Vectors to find consensus of
        threshold : float
            Voting threshold (default 0.5 for majority)
            
        Returns
        -------
        np.ndarray
            Consensus vector
        """
        if len(vectors) == 0:
            return np.zeros(self.config.dimension, dtype=np.uint8)
        
        # Sum vectors
        summed = np.sum(vectors, axis=0)
        
        # Apply threshold
        consensus = (summed >= len(vectors) * threshold).astype(np.uint8)
        
        return consensus


class MAP(VSA):
    """
    Multiply-Add-Permute (MAP) architecture.
    
    MAP combines multiplication, addition, and permutation for robust binding.
    Key features:
    - Works with multiple vector types
    - Enhanced binding properties
    - Better capacity than simple multiplication
    """
    
    def __init__(self, dimension: int = 10000,
                 vector_type: Union[VectorType, str] = VectorType.BIPOLAR,
                 num_permutations: int = 1,
                 seed: Optional[int] = None):
        """
        Initialize MAP architecture.
        
        Parameters
        ----------
        dimension : int
            Vector dimension
        vector_type : VectorType or str
            Type of vectors to use
        num_permutations : int
            Number of permutations in MAP operation
        seed : int, optional
            Random seed
        """
        config = VSAConfig(
            dimension=dimension,
            vector_type=vector_type,
            vsa_type=VSAType.MAP,
            binding_method="map",
            seed=seed
        )
        super().__init__(config)
        
        self.num_permutations = num_permutations
        
        # Generate permutations
        rng = np.random.RandomState(seed)
        self.permutations = [
            rng.permutation(dimension) for _ in range(num_permutations)
        ]
        
        logger.info(f"Initialized MAP with dimension={dimension}, "
                   f"num_permutations={num_permutations}")
    
    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Enhanced MAP binding with multiple permutations.
        
        bind(x, y) = normalize(x * y + sum(permute_i(x) + permute_i(y)))
        """
        # Element-wise multiplication
        result = x * y
        
        # Add permuted versions if using MAP
        if self.num_permutations > 0:
            for perm in self.permutations:
                result = result + x[perm] + y[perm]
        
        # Normalize based on vector type
        if self.config.normalize_result:
            result = self.vector_factory.normalize(result)
        
        return result
    
    def unbind(self, xy: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Approximate unbinding for MAP.
        
        For MAP: unbind(bind(x,y), x) ≈ y
        This is approximate due to the permutation additions.
        """
        if self.num_permutations == 0:
            # Simple multiplication - exact inverse
            if self.config.vector_type in [VectorType.BIPOLAR, VectorType.TERNARY]:
                # For discrete values, multiply by x (self-inverse property)
                result = xy * x
            else:
                # For continuous values, divide
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = xy / x
                    result[~np.isfinite(result)] = 0
        else:
            # MAP binding: result = normalize(x*y + sum(perm(x) + perm(y)))
            # The normalization makes exact recovery impossible
            # Use iterative approach for better recovery
            
            # Initial estimate: subtract permuted x and multiply by x
            result = xy.copy()
            for perm in self.permutations:
                result = result - x[perm]
            result = result * x
            
            # Iterative refinement (helps recover from normalization loss)
            for _ in range(2):
                # Estimate what binding would produce with current result
                est_bound = x * result
                for perm in self.permutations:
                    est_bound = est_bound + x[perm] + result[perm]
                est_bound = self.vector_factory.normalize(est_bound)
                
                # Compute error and update
                error = xy - est_bound
                result = result + 0.5 * error * x
        
        if self.config.normalize_result:
            result = self.vector_factory.normalize(result)
        
        return result
    
    def permute(self, vector: np.ndarray, shift: Optional[int] = None) -> np.ndarray:
        """
        Apply permutation to vector.
        
        Parameters
        ----------
        vector : np.ndarray
            Vector to permute
        shift : int, optional
            Not used, for compatibility with base class
            
        Returns
        -------
        np.ndarray
            Permuted vector
        """
        if len(self.permutations) > 0:
            return vector[self.permutations[0]]
        return vector
    
    def protect(self, vector: np.ndarray, 
                levels: int = 1) -> np.ndarray:
        """
        Apply multiple levels of permutation protection.
        
        Parameters
        ----------
        vector : np.ndarray
            Vector to protect
        levels : int
            Number of protection levels
            
        Returns
        -------
        np.ndarray
            Protected vector
        """
        result = vector
        for i in range(levels):
            perm_idx = i % len(self.permutations)
            result = result[self.permutations[perm_idx]]
        return result


class FHRR(VSA):
    """
    Fourier Holographic Reduced Representations (FHRR).
    
    FHRR performs binding in the frequency domain using complex vectors.
    Key features:
    - Uses complex vectors
    - Binding via element-wise multiplication in frequency domain
    - Efficient for large dimensions
    - Natural inverse via complex conjugate
    """
    
    def __init__(self, dimension: int = 10000,
                 use_real: bool = True,
                 seed: Optional[int] = None):
        """
        Initialize FHRR architecture.
        
        Parameters
        ----------
        dimension : int
            Vector dimension
        use_real : bool
            Whether to use real-valued vectors (via Hermitian constraint)
        seed : int, optional
            Random seed
        """
        # Use complex vectors internally
        config = VSAConfig(
            dimension=dimension,
            vector_type=VectorType.COMPLEX,
            vsa_type=VSAType.FHRR,
            binding_method="convolution",
            seed=seed
        )
        super().__init__(config)
        
        self.use_real = use_real
        
        logger.info(f"Initialized FHRR with dimension={dimension}, "
                   f"use_real={use_real}")
    
    def generate_vector(self, sparse: Optional[bool] = None) -> np.ndarray:
        """
        Generate FHRR vector with optional Hermitian constraint.
        
        For real-valued vectors, ensures FFT has Hermitian symmetry.
        """
        if self.use_real:
            # Generate random real vector
            real_vec = self._rng.randn(self.config.dimension)
            # Normalize to unit norm
            real_vec = real_vec / np.linalg.norm(real_vec)
            return real_vec.astype(np.float32)
        else:
            # Generate random complex vector with unit magnitude
            phases = self._rng.uniform(0, 2 * np.pi, self.config.dimension)
            vec = np.exp(1j * phases).astype(np.complex128)
            # Normalize
            vec = vec / np.linalg.norm(vec)
            return vec
    
    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        FHRR binding via frequency domain multiplication.
        
        bind(x, y) = IFFT(FFT(x) * FFT(y))
        """
        # Transform to frequency domain
        x_freq = fft(x)
        y_freq = fft(y)
        
        # Element-wise multiplication
        result_freq = x_freq * y_freq
        
        # Transform back
        result = ifft(result_freq)
        
        if self.use_real:
            result = np.real(result).astype(np.float32)
        else:
            result = result.astype(np.complex128)
        
        if self.config.normalize_result:
            result = self.vector_factory.normalize(result)
        
        return result
    
    def unbind(self, xy: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        FHRR unbinding via frequency domain division.
        
        unbind(xy, y) = IFFT(FFT(xy) / FFT(y))
        """
        # Transform to frequency domain
        xy_freq = fft(xy)
        y_freq = fft(y)
        
        # Element-wise division (with protection)
        with np.errstate(divide='ignore', invalid='ignore'):
            result_freq = xy_freq / y_freq
            # Handle division by zero
            result_freq[~np.isfinite(result_freq)] = 0
        
        # Transform back
        result = ifft(result_freq)
        
        if self.use_real:
            result = np.real(result).astype(np.float32)
        else:
            result = result.astype(np.complex128)
        
        if self.config.normalize_result:
            result = self.vector_factory.normalize(result)
        
        return result
    
    def make_unitary(self, vector: np.ndarray) -> np.ndarray:
        """
        Make vector unitary by normalizing in frequency domain.
        
        Sets all frequency magnitudes to 1.
        """
        # Transform to frequency domain
        freq = fft(vector)
        
        # Normalize magnitudes
        magnitudes = np.abs(freq)
        magnitudes[magnitudes == 0] = 1.0
        freq_normalized = freq / magnitudes
        
        # Transform back
        result = ifft(freq_normalized)
        
        if self.use_real:
            result = np.real(result).astype(np.float32)
        
        return self.vector_factory.normalize(result)


class SparseVSA(VSA):
    """
    Sparse VSA implementation with explicit sparsity control.
    
    Optimized for sparse vectors with efficient operations.
    Key features:
    - Ternary vectors {-1, 0, +1}
    - Adaptive sparsity
    - Efficient sparse operations
    - Context-dependent binding
    """
    
    def __init__(self, dimension: int = 10000,
                 sparsity: float = 0.9,
                 adaptive_sparsity: bool = True,
                 seed: Optional[int] = None):
        """
        Initialize Sparse VSA.
        
        Parameters
        ----------
        dimension : int
            Vector dimension
        sparsity : float
            Target sparsity level (0.9 = 90% zeros)
        adaptive_sparsity : bool
            Whether to adapt sparsity based on operations
        seed : int, optional
            Random seed
        """
        config = VSAConfig(
            dimension=dimension,
            vector_type=VectorType.TERNARY,
            vsa_type=VSAType.SPARSE,
            binding_method="multiplication",
            sparsity=sparsity,
            seed=seed
        )
        super().__init__(config)
        
        self.adaptive_sparsity = adaptive_sparsity
        self.target_sparsity = sparsity
        
        logger.info(f"Initialized SparseVSA with dimension={dimension}, "
                   f"sparsity={sparsity}")
    
    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Sparse binding with sparsity preservation.
        """
        # Standard multiplication binding
        result = x * y
        
        # Maintain sparsity if adaptive
        if self.adaptive_sparsity:
            current_sparsity = np.count_nonzero(result == 0) / len(result)
            if current_sparsity < self.target_sparsity:
                # Thin the result
                result = self.thin(result, self.target_sparsity)
        
        return result
    
    def bundle(self, vectors: List[np.ndarray],
               weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Sparse bundling with threshold.
        """
        # Standard bundling
        result = super().bundle(vectors, weights)
        
        # Apply sparsity threshold
        if self.adaptive_sparsity:
            # Keep only significant values
            threshold = np.percentile(np.abs(result), 
                                    self.target_sparsity * 100)
            result[np.abs(result) < threshold] = 0
            result = self.vector_factory.normalize(result)
        
        return result
    
    def compress(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compress sparse vector to indices and values.
        
        Returns
        -------
        indices : np.ndarray
            Indices of non-zero elements
        values : np.ndarray
            Values at those indices
        """
        indices = np.where(vector != 0)[0]
        values = vector[indices]
        return indices, values
    
    def decompress(self, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Decompress from indices and values to full vector.
        """
        vector = np.zeros(self.config.dimension, dtype=np.float32)
        vector[indices] = values
        return vector


class HRRCompatibility(VSA):
    """
    HRR compatibility layer for VSA.
    
    Provides HRR-style interface using VSA infrastructure.
    This allows HRR operations within the VSA framework.
    """
    
    def __init__(self, dimension: int = 1024,
                 seed: Optional[int] = None):
        """
        Initialize HRR compatibility layer.
        
        Parameters
        ----------
        dimension : int
            Vector dimension
        seed : int, optional
            Random seed
        """
        config = VSAConfig(
            dimension=dimension,
            vector_type=VectorType.REAL,  # HRR uses real vectors
            vsa_type=VSAType.HRR,
            binding_method="convolution",
            seed=seed
        )
        super().__init__(config)
        
        # Import HRR operations if available
        try:
            from ..hrr import operations as hrr_ops
            self.hrr_ops = hrr_ops
            self.use_native_hrr = True
        except ImportError:
            logger.warning("HRR module not available, using VSA convolution")
            self.use_native_hrr = False
    
    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """HRR binding via circular convolution."""
        if self.use_native_hrr:
            return self.hrr_ops.CircularConvolution.convolve(x, y)
        else:
            return super().bind(x, y)
    
    def unbind(self, xy: np.ndarray, y: np.ndarray) -> np.ndarray:
        """HRR unbinding via circular correlation."""
        if self.use_native_hrr:
            return self.hrr_ops.CircularConvolution.correlate(xy, y)
        else:
            return super().unbind(xy, y)
    
    def make_unitary(self, vector: np.ndarray) -> np.ndarray:
        """Make vector unitary using HRR method."""
        if self.use_native_hrr and hasattr(self.hrr_ops, 'make_unitary'):
            return self.hrr_ops.make_unitary(vector)
        else:
            # Use FHRR approach
            fhrr = FHRR(self.config.dimension, use_real=True)
            return fhrr.make_unitary(vector)


# Factory functions for easy creation

def create_bsc(dimension: int = 8192, **kwargs) -> BSC:
    """Create a Binary Spatter Codes VSA."""
    return BSC(dimension=dimension, **kwargs)


def create_map(dimension: int = 10000, **kwargs) -> MAP:
    """Create a MAP architecture VSA."""
    return MAP(dimension=dimension, **kwargs)


def create_fhrr(dimension: int = 10000, **kwargs) -> FHRR:
    """Create a Fourier HRR VSA."""
    return FHRR(dimension=dimension, **kwargs)


def create_sparse_vsa(dimension: int = 10000, 
                      sparsity: float = 0.9, **kwargs) -> SparseVSA:
    """Create a Sparse VSA."""
    return SparseVSA(dimension=dimension, sparsity=sparsity, **kwargs)


def create_hrr_compatible(dimension: int = 1024, **kwargs) -> HRRCompatibility:
    """Create an HRR-compatible VSA."""
    return HRRCompatibility(dimension=dimension, **kwargs)