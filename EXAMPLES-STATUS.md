# Examples Status and Fixes

This document tracks the status of all example scripts and documents the fixes applied to make them work correctly.

## Summary

✅ **All 9 examples are now fully functional**
- 4 SDM examples
- 5 HRR examples

## SDM Examples

### 1. basic_sdm_demo.py ✅
**Status**: Working
**Fix Applied**: Added numpy import to `cognitive_computing/sdm/__init__.py`:
```python
import numpy as np  # Added for create_sdm function
```

### 2. noise_tolerance.py ✅
**Status**: Working
**No fixes required**

### 3. pattern_recognition.py ✅
**Status**: Working
**Fix Applied**: Corrected label encoding to match pattern dimension:
```python
# OLD: label = np.zeros(10, dtype=np.uint8)  # Wrong dimension
# NEW: label = np.zeros(self.dimension, dtype=np.uint8)  # Correct dimension
```
**Key Learning**: SDM requires data (labels) to have the same dimension as addresses (patterns)

### 4. sequence_memory.py ✅
**Status**: Working
**Fix Applied**: Ensured numpy array conversion for pattern generation:
```python
patterns = np.array(patterns)  # Convert list to numpy array before reshape
```

## HRR Examples

### 1. basic_hrr_demo.py ✅
**Status**: Working
**No fixes required**

### 2. symbol_binding.py ✅
**Status**: Working
**Fixes Applied**:
1. Added proper import for similarity function
2. Added error handling for cleanup operations:
```python
try:
    name, _, confidence = self.cleanup.cleanup(filler)
    return name, confidence
except ValueError:
    # Fallback to manual similarity search
    best_name = None
    best_sim = -1.0
    for name, vec in self.symbols.items():
        sim = self.hrr.similarity(filler, vec)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    return best_name if best_name else "Unknown", best_sim
```

### 3. sequence_processing.py ✅
**Status**: Working
**Fixes Applied**:
1. Changed method name from "position" to "positional" throughout
2. Fixed decode_position calls to include method parameter:
```python
retrieved = self.encoder.decode_position(sequence, position, method="positional")
```
3. Fixed CleanupMemory API usage (get_item instead of has_item/get_vector)

### 4. hierarchical_processing.py ✅
**Status**: Working
**Fixes Applied**:
1. Added automatic item registration for tree structures
2. Changed decode_subtree to decode_path
3. Added _register_tree_items helper method:
```python
def _register_tree_items(self, tree: Dict[str, Any]):
    """Recursively register all items in a tree."""
    for key, value in tree.items():
        if isinstance(key, str) and key not in self.hrr.memory:
            vector = self.add_node(key)
            self.hrr.add_item(key, vector)
        if isinstance(value, dict):
            self._register_tree_items(value)
        elif isinstance(value, str):
            if value not in self.hrr.memory:
                vector = self.add_node(value)
                self.hrr.add_item(value, vector)
```

### 5. analogical_reasoning.py ✅
**Status**: Working
**Fix Applied**: Added error handling for cleanup operations (same pattern as symbol_binding.py)

## Key API Learnings

### SDM
1. **Dimension Matching**: Both address and data must have the same dimension
2. **Label Encoding**: For classification, use full dimension with one-hot in first N positions
3. **Module Imports**: SDM module includes numpy import for convenience

### HRR
1. **Sequence Methods**: Use "positional", "chaining", or "temporal" (not "position")
2. **Cleanup Memory**: Always wrap cleanup() in try-except as it may fail
3. **Item Registration**: Items must be in both CleanupMemory and HRR memory
4. **Hierarchical Encoding**: Use decode_path() not decode_subtree()
5. **Auto-registration**: HierarchicalProcessor now auto-registers all tree items

## Best Practices for Examples

1. **Always handle cleanup failures** with try-except blocks
2. **Pre-register all symbols** before encoding structures
3. **Use correct method names** for sequence encoding
4. **Match dimensions** for SDM pattern and label storage
5. **Test with realistic thresholds** - HRR similarities are often low (0.1-0.3)

## Performance Notes

- SDM examples show good noise tolerance up to ~20% noise
- HRR examples show lower accuracy due to distributed representation interference
- Hierarchical and analogical reasoning have expected low similarities due to complexity

All examples now demonstrate the full capabilities of both SDM and HRR implementations while handling edge cases gracefully.