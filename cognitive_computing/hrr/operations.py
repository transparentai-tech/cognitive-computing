"""
Circular convolution and correlation operations for HRR.

This module provides efficient implementations of circular convolution
and related operations used in Holographic Reduced Representations.
"""

import logging
from typing import Optional, Tuple, Union
import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class CircularConvolution:
    """
    Efficient circular convolution implementations.
    
    Provides both direct and FFT-based methods for circular convolution
    and correlation, with automatic selection based on vector dimension.
    """
    
    # Threshold for switching between direct and FFT methods
    FFT_THRESHOLD = 128
    
    @staticmethod
    def convolve(a: np.ndarray, b: np.ndarray, 
                 method: str = "auto") -> np.ndarray:
        """
        Compute circular convolution of two vectors.
        
        Circular convolution is the fundamental binding operation in HRR.
        For vectors a and b, it produces a vector c where each element
        c[k] = sum(a[i] * b[(k-i) mod n]) for i in range(n).
        
        Parameters
        ----------
        a, b : np.ndarray
            Input vectors of the same length
        method : str, optional
            Convolution method: "auto", "direct", or "fft"
            
        Returns
        -------
        np.ndarray
            Circular convolution result
            
        Examples
        --------
        >>> a = np.array([1, 2, 3, 4])
        >>> b = np.array([5, 6, 7, 8])
        >>> c = CircularConvolution.convolve(a, b)
        """
        if a.shape != b.shape:
            raise ValueError(f"Vectors must have same shape, got {a.shape} "
                           f"and {b.shape}")
        
        n = len(a)
        
        # Choose method
        if method == "auto":
            method = "fft" if n > CircularConvolution.FFT_THRESHOLD else "direct"
        
        if method == "direct":
            return CircularConvolution._convolve_direct(a, b)
        elif method == "fft":
            return CircularConvolution._convolve_fft(a, b)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def correlate(a: np.ndarray, b: np.ndarray, 
                  method: str = "auto") -> np.ndarray:
        """
        Compute circular correlation of two vectors.
        
        Circular correlation is used for unbinding in HRR. It's equivalent
        to convolution with a reversed and conjugated version of b.
        
        Parameters
        ----------
        a, b : np.ndarray
            Input vectors of the same length
        method : str, optional
            Correlation method: "auto", "direct", or "fft"
            
        Returns
        -------
        np.ndarray
            Circular correlation result
        """
        if a.shape != b.shape:
            raise ValueError(f"Vectors must have same shape, got {a.shape} "
                           f"and {b.shape}")
        
        n = len(a)
        
        # Choose method
        if method == "auto":
            method = "fft" if n > CircularConvolution.FFT_THRESHOLD else "direct"
        
        if method == "direct":
            return CircularConvolution._correlate_direct(a, b)
        elif method == "fft":
            return CircularConvolution._correlate_fft(a, b)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def _convolve_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Direct O(n²) circular convolution implementation."""
        n = len(a)
        # Create result array with appropriate dtype
        if np.iscomplexobj(a) or np.iscomplexobj(b):
            result = np.zeros(n, dtype=np.complex128)
        else:
            result = np.zeros(n, dtype=a.dtype)
        
        for k in range(n):
            for i in range(n):
                result[k] += a[i] * b[(k - i) % n]
        
        return result
    
    @staticmethod
    def _convolve_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """FFT-based O(n log n) circular convolution implementation."""
        # Convolution theorem: conv(a, b) = IFFT(FFT(a) * FFT(b))
        fft_a = np.fft.fft(a)
        fft_b = np.fft.fft(b)
        result = np.fft.ifft(fft_a * fft_b)
        
        # Return real part for real inputs
        if not (np.iscomplexobj(a) or np.iscomplexobj(b)):
            result = np.real(result)
        
        return result
    
    @staticmethod
    def _correlate_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Direct O(n²) circular correlation implementation."""
        n = len(a)
        # Create result array with appropriate dtype
        if np.iscomplexobj(a) or np.iscomplexobj(b):
            result = np.zeros(n, dtype=np.complex128)
        else:
            result = np.zeros(n, dtype=a.dtype)
        
        # Correlation: result[k] = sum(a[i] * conj(b[(i - k) % n]))
        for k in range(n):
            for i in range(n):
                if np.iscomplexobj(b):
                    result[k] += a[i] * np.conj(b[(i - k) % n])
                else:
                    result[k] += a[i] * b[(i - k) % n]
        
        return result
    
    @staticmethod
    def _correlate_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """FFT-based O(n log n) circular correlation implementation."""
        # Correlation theorem: corr(a, b) = IFFT(FFT(a) * conj(FFT(b)))
        fft_a = np.fft.fft(a)
        fft_b = np.fft.fft(b)
        result = np.fft.ifft(fft_a * np.conj(fft_b))
        
        # Return real part for real inputs
        if not (np.iscomplexobj(a) or np.iscomplexobj(b)):
            result = np.real(result)
        
        return result
    
    @staticmethod
    def convolve_multiple(vectors: list, method: str = "auto") -> np.ndarray:
        """
        Compute circular convolution of multiple vectors.
        
        Convolves vectors in sequence: v1 * v2 * v3 * ...
        Note: This operation is not associative.
        
        Parameters
        ----------
        vectors : list of np.ndarray
            List of vectors to convolve
        method : str, optional
            Convolution method
            
        Returns
        -------
        np.ndarray
            Result of sequential convolution
        """
        if not vectors:
            raise ValueError("Cannot convolve empty list")
        
        result = vectors[0].copy()
        for v in vectors[1:]:
            result = CircularConvolution.convolve(result, v, method)
        
        return result


class VectorOperations:
    """Additional vector operations useful for HRR."""
    
    @staticmethod
    def normalize(vector: np.ndarray, ord: Optional[Union[int, float]] = 2) -> np.ndarray:
        """
        Normalize a vector to unit length.
        
        Parameters
        ----------
        vector : np.ndarray
            Vector to normalize
        ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
            Order of the norm (default: 2 for Euclidean norm)
            
        Returns
        -------
        np.ndarray
            Normalized vector
        """
        if np.iscomplexobj(vector):
            # For complex vectors, use complex inner product
            norm = np.sqrt(np.real(np.vdot(vector, vector)))
        else:
            norm = np.linalg.norm(vector, ord=ord)
        
        if norm > 0:
            return vector / norm
        return vector
    
    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray, 
                   metric: str = "cosine") -> float:
        """
        Calculate similarity between two vectors.
        
        Parameters
        ----------
        a, b : np.ndarray
            Vectors to compare
        metric : str, optional
            Similarity metric: "cosine", "dot", "euclidean"
            
        Returns
        -------
        float
            Similarity value
        """
        if a.shape != b.shape:
            raise ValueError(f"Vectors must have same shape, got {a.shape} "
                           f"and {b.shape}")
        
        if metric == "cosine":
            # Cosine similarity
            if np.iscomplexobj(a) or np.iscomplexobj(b):
                dot_product = np.real(np.vdot(a, b))
                norm_a = np.sqrt(np.real(np.vdot(a, a)))
                norm_b = np.sqrt(np.real(np.vdot(b, b)))
            else:
                dot_product = np.dot(a, b)
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)
            
        elif metric == "dot":
            # Dot product
            if np.iscomplexobj(a) or np.iscomplexobj(b):
                return np.real(np.vdot(a, b))
            return np.dot(a, b)
            
        elif metric == "euclidean":
            # Negative Euclidean distance (so higher is more similar)
            return -np.linalg.norm(a - b)
            
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    @staticmethod
    def make_unitary(vector: np.ndarray) -> np.ndarray:
        """
        Make a vector unitary (self-inverse under correlation).
        
        A unitary vector satisfies: correlate(u, u) = identity
        
        Parameters
        ----------
        vector : np.ndarray
            Input vector
            
        Returns
        -------
        np.ndarray
            Unitary vector
        """
        if np.iscomplexobj(vector):
            # For complex vectors, set all magnitudes to 1
            return np.exp(1j * np.angle(vector))
        else:
            # For real vectors, make FFT have unit magnitude
            # while maintaining conjugate symmetry for real output
            fft = np.fft.fft(vector)
            n = len(vector)
            
            # Set magnitudes to 1 while preserving phases
            phases = np.angle(fft)
            
            # First half (including DC and Nyquist if present)
            for i in range((n + 1) // 2):
                if i == 0 or (n % 2 == 0 and i == n // 2):
                    # DC and Nyquist must be real for real output
                    fft[i] = 1.0 if np.real(fft[i]) >= 0 else -1.0
                else:
                    fft[i] = np.exp(1j * phases[i])
            
            # Second half must be conjugate of first half
            for i in range(1, n // 2):
                fft[n - i] = np.conj(fft[i])
            
            # Transform back
            result = np.real(np.fft.ifft(fft))
            
            # Normalize to ensure unit length
            return VectorOperations.normalize(result)
    
    @staticmethod
    def random_permutation(dimension: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a random permutation vector.
        
        Creates a vector that, when used with circular convolution,
        acts as a permutation operator.
        
        Parameters
        ----------
        dimension : int
            Dimension of the permutation
        seed : int, optional
            Random seed
            
        Returns
        -------
        np.ndarray
            Permutation vector with a single 1.0 at a random position
        """
        rng = np.random.RandomState(seed)
        perm = np.zeros(dimension)
        # Put a 1.0 at a random position
        position = rng.randint(dimension)
        perm[position] = 1.0
        return perm
    
    @staticmethod
    def circular_shift(vector: np.ndarray, shift: int) -> np.ndarray:
        """
        Circularly shift a vector by a given amount.
        
        Parameters
        ----------
        vector : np.ndarray
            Vector to shift
        shift : int
            Number of positions to shift (positive = right)
            
        Returns
        -------
        np.ndarray
            Shifted vector
        """
        return np.roll(vector, shift)
    
    @staticmethod
    def power(vector: np.ndarray, n: int) -> np.ndarray:
        """
        Compute the nth power of a vector under convolution.
        
        power(v, n) = v * v * ... * v (n times)
        
        Parameters
        ----------
        vector : np.ndarray
            Base vector
        n : int
            Power (must be >= 0)
            
        Returns
        -------
        np.ndarray
            Vector raised to the nth power
        """
        if n < 0:
            raise ValueError(f"Power must be non-negative, got {n}")
        
        if n == 0:
            # Identity under convolution is delta function
            result = np.zeros_like(vector)
            result[0] = 1.0
            return result
        
        # Use repeated squaring for efficiency
        result = vector.copy()
        power = 1
        
        while power < n:
            if power * 2 <= n:
                result = CircularConvolution.convolve(result, result)
                power *= 2
            else:
                result = CircularConvolution.convolve(result, vector)
                power += 1
        
        return result
    
    @staticmethod
    def inverse(vector: np.ndarray, method: str = "correlation") -> np.ndarray:
        """
        Compute the inverse of a vector under convolution.
        
        The inverse v_inv satisfies: convolve(v, v_inv) = identity
        
        Parameters
        ----------
        vector : np.ndarray
            Vector to invert
        method : str, optional
            Inversion method: "correlation" or "fft"
            
        Returns
        -------
        np.ndarray
            Inverse vector
        """
        if method == "correlation":
            # For correlation-based inverse, we want v_inv such that
            # convolve(v, v_inv)[0] = 1
            # For circular convolution, if v shifts by k, v_inv should shift by -k
            n = len(vector)
            result = np.zeros_like(vector)
            
            # Find position of max value in vector
            k = np.argmax(np.abs(vector))
            if k == 0:
                # Identity vector
                result[0] = 1.0 / vector[0] if vector[0] != 0 else 1.0
            else:
                # Inverse shifts by -k = n-k
                result[n - k] = 1.0 / vector[k] if vector[k] != 0 else 1.0
            
            return result
        elif method == "fft":
            # Compute inverse in frequency domain
            fft = np.fft.fft(vector)
            # Avoid division by zero
            fft_inv = np.where(np.abs(fft) > 1e-10, 1.0 / fft, 0.0)
            return np.fft.ifft(fft_inv)
        else:
            raise ValueError(f"Unknown method: {method}")