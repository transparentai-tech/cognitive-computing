"""
Encoding strategies for HRR.

This module provides various encoding strategies for creating structured
representations using HRR, including role-filler bindings, sequences,
and hierarchical structures.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from .core import HRR
from .operations import CircularConvolution, VectorOperations

logger = logging.getLogger(__name__)


class RoleFillerEncoder:
    """
    Encode role-filler structures using HRR.
    
    Role-filler encoding is fundamental to representing structured information
    in HRR. Each piece of information (filler) is bound to its role, and
    multiple role-filler pairs are bundled together.
    
    Parameters
    ----------
    hrr : HRR
        HRR system to use for encoding
        
    Examples
    --------
    >>> hrr = HRR(HRRConfig(dimension=1024))
    >>> encoder = RoleFillerEncoder(hrr)
    >>> 
    >>> # Create roles and fillers
    >>> color_role = hrr.generate_vector()
    >>> red_filler = hrr.generate_vector()
    >>> 
    >>> # Encode single pair
    >>> red_color = encoder.encode_pair(color_role, red_filler)
    """
    
    def __init__(self, hrr: HRR):
        """Initialize the encoder."""
        self.hrr = hrr
    
    def encode_pair(self, role: np.ndarray, filler: np.ndarray) -> np.ndarray:
        """
        Encode a single role-filler pair.
        
        Parameters
        ----------
        role : np.ndarray
            Role vector
        filler : np.ndarray
            Filler vector
            
        Returns
        -------
        np.ndarray
            Bound role-filler pair
        """
        return self.hrr.bind(role, filler)
    
    def encode_structure(self, role_filler_pairs: Dict[str, np.ndarray],
                        role_vectors: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Encode a complete structure from role-filler pairs.
        
        Parameters
        ----------
        role_filler_pairs : Dict[str, np.ndarray]
            Dictionary mapping role names to filler vectors
        role_vectors : Dict[str, np.ndarray], optional
            Pre-defined role vectors. If not provided, generates random roles.
            
        Returns
        -------
        np.ndarray
            Bundled representation of all role-filler pairs
            
        Examples
        --------
        >>> structure = encoder.encode_structure({
        ...     "color": red_vector,
        ...     "shape": circle_vector,
        ...     "size": large_vector
        ... })
        """
        if not role_filler_pairs:
            raise ValueError("Cannot encode empty structure")
        
        # Generate or retrieve role vectors
        if role_vectors is None:
            role_vectors = {}
            for role_name in role_filler_pairs:
                if role_name not in self.hrr.memory:
                    # Generate and store new role vector
                    role_vec = self.hrr.generate_vector()
                    self.hrr.add_item(f"role:{role_name}", role_vec)
                else:
                    role_vec = self.hrr.get_item(f"role:{role_name}")
                role_vectors[role_name] = role_vec
        
        # Encode all pairs
        bound_pairs = []
        for role_name, filler in role_filler_pairs.items():
            role = role_vectors.get(role_name)
            if role is None:
                raise ValueError(f"No role vector provided for '{role_name}'")
            
            bound_pair = self.encode_pair(role, filler)
            bound_pairs.append(bound_pair)
        
        # Bundle all pairs
        return self.hrr.bundle(bound_pairs)
    
    def decode_filler(self, structure: np.ndarray, role: np.ndarray) -> np.ndarray:
        """
        Decode a filler from a structure given its role.
        
        Parameters
        ----------
        structure : np.ndarray
            Encoded structure containing role-filler pairs
        role : np.ndarray
            Role vector to query
            
        Returns
        -------
        np.ndarray
            Retrieved filler vector (may be noisy)
        """
        return self.hrr.unbind(structure, role)
    
    def decode_all_fillers(self, structure: np.ndarray,
                          role_vectors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Decode all fillers from a structure.
        
        Parameters
        ----------
        structure : np.ndarray
            Encoded structure
        role_vectors : Dict[str, np.ndarray]
            Dictionary of role vectors
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping role names to decoded fillers
        """
        decoded = {}
        for role_name, role_vec in role_vectors.items():
            decoded[role_name] = self.decode_filler(structure, role_vec)
        return decoded


class SequenceEncoder:
    """
    Encode sequences using HRR.
    
    Provides methods for encoding ordered sequences and retrieving
    elements by position.
    
    Parameters
    ----------
    hrr : HRR
        HRR system to use for encoding
    position_vectors : Optional[List[np.ndarray]]
        Pre-computed position vectors. If None, generates as needed.
    """
    
    def __init__(self, hrr: HRR, position_vectors: Optional[List[np.ndarray]] = None):
        """Initialize the sequence encoder."""
        self.hrr = hrr
        self.position_vectors = position_vectors or []
        self._position_cache: Dict[int, np.ndarray] = {}
    
    def get_position_vector(self, position: int) -> np.ndarray:
        """
        Get or generate a position vector.
        
        Parameters
        ----------
        position : int
            Position index (0-based)
            
        Returns
        -------
        np.ndarray
            Position vector for the given index
        """
        if position < 0:
            raise ValueError(f"Position must be non-negative, got {position}")
        
        # Check cache first
        if position in self._position_cache:
            return self._position_cache[position]
        
        # Check pre-computed vectors
        if position < len(self.position_vectors):
            vec = self.position_vectors[position]
        else:
            # Generate new position vector
            # Use a deterministic method based on position
            seed = 12345 + position  # Deterministic seed
            rng = np.random.RandomState(seed)
            
            if self.hrr.config.storage_method == "complex":
                phases = rng.uniform(0, 2 * np.pi, self.hrr.config.dimension // 2)
                vec = np.cos(phases) + 1j * np.sin(phases)
            else:
                vec = rng.randn(self.hrr.config.dimension)
            
            vec = self.hrr._normalize(vec)
            
            # Store in memory for later retrieval
            self.hrr.add_item(f"pos:{position}", vec)
        
        self._position_cache[position] = vec
        return vec
    
    def encode_sequence(self, items: List[np.ndarray], 
                       method: str = "chaining") -> np.ndarray:
        """
        Encode a sequence of items.
        
        Parameters
        ----------
        items : List[np.ndarray]
            Ordered list of item vectors
        method : str
            Encoding method: "chaining" or "positional"
            
        Returns
        -------
        np.ndarray
            Encoded sequence representation
            
        Examples
        --------
        >>> words = [word1_vec, word2_vec, word3_vec]
        >>> sentence = encoder.encode_sequence(words)
        """
        if not items:
            raise ValueError("Cannot encode empty sequence")
        
        if method == "positional":
            # Bind each item to its position and bundle
            bound_items = []
            for i, item in enumerate(items):
                pos_vec = self.get_position_vector(i)
                bound = self.hrr.bind(pos_vec, item)
                bound_items.append(bound)
            return self.hrr.bundle(bound_items)
            
        elif method == "chaining":
            # Chain items using convolution
            # seq = item[0] + P*item[1] + P²*item[2] + ...
            # where P is a permutation/shift operator
            
            # Get or create permutation vector
            if "sequence_permutation" not in self.hrr.memory:
                perm = self.hrr.generate_vector(unitary=True)
                self.hrr.add_item("sequence_permutation", perm)
            else:
                perm = self.hrr.get_item("sequence_permutation")
            
            # Build sequence
            result = items[0].copy()
            perm_power = perm.copy()
            
            for item in items[1:]:
                # Bind item with current power of permutation
                bound = self.hrr.bind(perm_power, item)
                result = self.hrr.bundle([result, bound])
                # Update permutation power
                perm_power = self.hrr.bind(perm_power, perm)
            
            return result
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def decode_position(self, sequence: np.ndarray, position: int,
                       method: str = "chaining") -> np.ndarray:
        """
        Decode an item at a specific position in the sequence.
        
        Parameters
        ----------
        sequence : np.ndarray
            Encoded sequence
        position : int
            Position to decode (0-based)
        method : str
            Decoding method (must match encoding method)
            
        Returns
        -------
        np.ndarray
            Decoded item vector (may be noisy)
        """
        if method == "positional":
            pos_vec = self.get_position_vector(position)
            return self.hrr.unbind(sequence, pos_vec)
            
        elif method == "chaining":
            # Get permutation vector
            perm = self.hrr.get_item("sequence_permutation")
            if perm is None:
                raise ValueError("No sequence permutation vector found")
            
            # Compute appropriate power of permutation
            if position == 0:
                # No unbinding needed for first position
                return sequence  # Note: will contain noise from other positions
            else:
                # Compute P^position
                perm_power = perm.copy()
                for _ in range(position - 1):
                    perm_power = self.hrr.bind(perm_power, perm)
                
                # Unbind to get item
                return self.hrr.unbind(sequence, perm_power)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def decode_all_positions(self, sequence: np.ndarray, length: int,
                           method: str = "chaining") -> List[np.ndarray]:
        """
        Decode all positions in a sequence.
        
        Parameters
        ----------
        sequence : np.ndarray
            Encoded sequence
        length : int
            Number of positions to decode
        method : str
            Decoding method
            
        Returns
        -------
        List[np.ndarray]
            List of decoded vectors
        """
        decoded = []
        for i in range(length):
            item = self.decode_position(sequence, i, method)
            decoded.append(item)
        return decoded


class HierarchicalEncoder:
    """
    Encode hierarchical structures using HRR.
    
    Supports encoding of tree-like structures with nested components.
    
    Parameters
    ----------
    hrr : HRR
        HRR system to use for encoding
    """
    
    def __init__(self, hrr: HRR):
        """Initialize the hierarchical encoder."""
        self.hrr = hrr
        self.role_encoder = RoleFillerEncoder(hrr)
    
    def encode_tree(self, tree: Dict[str, Any], 
                   role_vectors: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Encode a hierarchical tree structure.
        
        Parameters
        ----------
        tree : Dict[str, Any]
            Tree structure where values can be:
            - np.ndarray: Leaf node vectors
            - Dict: Nested structures
            - str: References to stored items
        role_vectors : Dict[str, np.ndarray], optional
            Pre-defined role vectors
            
        Returns
        -------
        np.ndarray
            Encoded tree representation
            
        Examples
        --------
        >>> tree = {
        ...     "type": type_vector,
        ...     "properties": {
        ...         "color": red_vector,
        ...         "size": large_vector
        ...     },
        ...     "parts": {
        ...         "left": left_part_vector,
        ...         "right": right_part_vector
        ...     }
        ... }
        >>> encoded = encoder.encode_tree(tree)
        """
        encoded_pairs = {}
        
        for key, value in tree.items():
            if isinstance(value, np.ndarray):
                # Leaf node - use directly
                encoded_pairs[key] = value
            elif isinstance(value, dict):
                # Nested structure - recursive encoding
                sub_encoding = self.encode_tree(value, role_vectors)
                encoded_pairs[key] = sub_encoding
            elif isinstance(value, str):
                # Reference to stored item
                item = self.hrr.get_item(value)
                if item is None:
                    raise ValueError(f"Item '{value}' not found in memory")
                encoded_pairs[key] = item
            else:
                raise TypeError(f"Unsupported value type: {type(value)}")
        
        # Encode using role-filler binding
        return self.role_encoder.encode_structure(encoded_pairs, role_vectors)
    
    def decode_path(self, encoding: np.ndarray, 
                   path: List[str],
                   role_vectors: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Decode a value at a specific path in the hierarchy.
        
        Parameters
        ----------
        encoding : np.ndarray
            Encoded hierarchical structure
        path : List[str]
            Path to the desired value (e.g., ["properties", "color"])
        role_vectors : Dict[str, np.ndarray], optional
            Role vectors used in encoding
            
        Returns
        -------
        np.ndarray
            Decoded value at the path (may be noisy)
        """
        if not path:
            return encoding
        
        # Get role vectors from memory if not provided
        if role_vectors is None:
            role_vectors = {}
            for role_name in path:
                role_key = f"role:{role_name}"
                role_vec = self.hrr.get_item(role_key)
                if role_vec is None:
                    raise ValueError(f"Role vector for '{role_name}' not found")
                role_vectors[role_name] = role_vec
        
        # Decode step by step
        current = encoding
        for role_name in path:
            role_vec = role_vectors.get(role_name)
            if role_vec is None:
                raise ValueError(f"No role vector for '{role_name}'")
            current = self.hrr.unbind(current, role_vec)
        
        return current
    
    def flatten_tree(self, tree: Dict[str, Any], 
                    prefix: str = "") -> Dict[str, Any]:
        """
        Flatten a hierarchical tree into path-value pairs.
        
        Parameters
        ----------
        tree : Dict[str, Any]
            Hierarchical tree structure
        prefix : str
            Prefix for paths (used in recursion)
            
        Returns
        -------
        Dict[str, Any]
            Flattened dictionary with paths as keys
            
        Examples
        --------
        >>> tree = {"a": {"b": vec1, "c": vec2}}
        >>> flat = encoder.flatten_tree(tree)
        >>> # Result: {"a.b": vec1, "a.c": vec2}
        """
        flat = {}
        
        for key, value in tree.items():
            path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursive flattening
                flat.update(self.flatten_tree(value, path))
            else:
                flat[path] = value
        
        return flat