# VSA API Fixes Summary for data_encoding.py

## Changes Made

### 1. VSA Creation
- **OLD**: `create_vsa(VSAConfig(dimension=..., vector_type=..., binding_method=...))`
- **NEW**: `create_vsa(dimension=..., vector_type=..., vsa_type=..., binding_method=...)`

### 2. Vector Generation
- **OLD**: `vsa.encode('item')` or `vsa.encode(f'item_{i}')`
- **NEW**: `vsa.generate_vector()` followed by `vsa.store('item', vector)`

### 3. Zero Vector
- **OLD**: `vsa.zero()`
- **NEW**: Use empty list pattern `[]` then `vsa.bundle(list)`

### 4. Identity Vector
- **OLD**: `vsa.identity()`
- **NEW**: Use appropriate vector for the context (often just `vsa.generate_vector()`)

### 5. Raw Encoding
- **OLD**: `vsa.encode_raw(data)`
- **NEW**: Direct array operations (e.g., `vector * weight`)

### 6. Sequence Encoding
- **OLD**: `SequenceEncoder` class
- **NEW**: Manual implementation using `vsa.permute()` and `vsa.bind()`

### 7. Random Indexing Encoder
- **OLD**: `RandomIndexingEncoder(vsa, n_gram_size=3)`
- **NEW**: `RandomIndexingEncoder(vsa, num_indices=10, window_size=3)`

### 8. Graph Encoder
- **OLD**: `graph_encoder.encode_edge(node_vector1, node_vector2)`
- **NEW**: `graph_encoder.encode_edge(node_id1, node_id2)`
- **OLD**: `graph_encoder.encode_path(node_vectors)`
- **NEW**: Manual path encoding using bind and permute operations

### 9. Level Encoder
- **OLD**: `level_encoder.level_to_value(level)`
- **NEW**: `level_encoder.decode(vector)` returns the value directly

### 10. Memory Access
- **OLD**: `vsa.encode('field_name')` for retrieval
- **NEW**: `vsa.memory.get('field_name')` to retrieve stored vectors

## Key Patterns

1. **Bundling Pattern**: Instead of starting with `vsa.zero()` and incrementally bundling, collect vectors in a list and bundle at the end.

2. **Role-Filler Pattern**: Create separate role vectors and filler vectors, store both, then bind them together.

3. **Direct Array Operations**: VSA works with numpy arrays directly, no vector object wrappers in the public API.

4. **Explicit Storage**: Always use `vsa.store(key, vector)` to save vectors for later retrieval.

5. **Memory Retrieval**: Use `vsa.memory.get(key)` to retrieve stored vectors, with None check for safety.