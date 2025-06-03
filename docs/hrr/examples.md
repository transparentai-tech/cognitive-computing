# HRR Examples and Use Cases

This guide provides detailed examples of using HRR for various cognitive computing tasks. Each example includes complete code and explanations.

## Table of Contents

1. [Basic Operations](#basic-operations)
2. [Symbol Binding](#symbol-binding)
3. [Sequence Processing](#sequence-processing)
4. [Hierarchical Structures](#hierarchical-structures)
5. [Analogical Reasoning](#analogical-reasoning)
6. [Working with Cleanup Memory](#working-with-cleanup-memory)
7. [Complex Applications](#complex-applications)
8. [Integration Examples](#integration-examples)

## Basic Operations

### Getting Started

```python
from cognitive_computing.hrr import create_hrr
import numpy as np

# Create an HRR system
hrr = create_hrr(dimension=1024, normalize=True)

# Generate vectors
a = hrr.generate_vector()  # Random vector
b = hrr.generate_vector()  # Another random vector
u = hrr.generate_vector(method="unitary")  # Unitary vector
```

### Binding and Unbinding

```python
# Bind two vectors
c = hrr.bind(a, b)

# Unbind to retrieve original
b_retrieved = hrr.unbind(c, a)
a_retrieved = hrr.unbind(c, b)

# Check retrieval quality
print(f"Similarity to b: {hrr.similarity(b, b_retrieved):.3f}")
print(f"Similarity to a: {hrr.similarity(a, a_retrieved):.3f}")

# Perfect unbinding with unitary vectors
v = hrr.generate_vector()
bound = hrr.bind(u, v)
v_perfect = hrr.unbind(bound, u)
print(f"Perfect retrieval: {hrr.similarity(v, v_perfect):.3f}")
```

### Bundling Multiple Items

```python
# Bundle several vectors
items = [hrr.generate_vector() for _ in range(5)]
bundle = hrr.bundle(items)

# Check presence of each item
for i, item in enumerate(items):
    sim = hrr.similarity(bundle, item)
    print(f"Item {i} similarity: {sim:.3f}")

# Weighted bundling
weights = [3.0, 2.0, 1.0, 1.0, 1.0]
weighted_bundle = hrr.bundle(items, weights=weights)
```

## Symbol Binding

### Creating a Symbol System

```python
from cognitive_computing.hrr import RoleFillerEncoder, CleanupMemory, CleanupMemoryConfig

# Create encoder and cleanup memory
encoder = RoleFillerEncoder(hrr)
cleanup = CleanupMemory(CleanupMemoryConfig(threshold=0.3), dimension=1024)

# Define symbols
symbols = {
    "john": hrr.generate_vector(),
    "mary": hrr.generate_vector(),
    "loves": hrr.generate_vector(),
    "knows": hrr.generate_vector()
}

# Add to cleanup memory
for name, vector in symbols.items():
    cleanup.add_item(name, vector)

# Define roles (use unitary vectors)
roles = {
    "agent": hrr.generate_vector(method="unitary"),
    "action": hrr.generate_vector(method="unitary"),
    "patient": hrr.generate_vector(method="unitary")
}
```

### Encoding Propositions

```python
# Encode "John loves Mary"
proposition = encoder.encode_structure({
    "agent": symbols["john"],
    "action": symbols["loves"],
    "patient": symbols["mary"]
})

# Query the structure with error handling
try:
    agent_vec = encoder.decode_filler(proposition, roles["agent"])
    agent_name, _, conf = cleanup.cleanup(agent_vec)
    print(f"Agent: {agent_name} (confidence: {conf:.3f})")
except Exception as e:
    print(f"Could not decode agent: {e}")

try:
    action_vec = encoder.decode_filler(proposition, roles["action"])
    action_name, _, conf = cleanup.cleanup(action_vec)
    print(f"Action: {action_name} (confidence: {conf:.3f})")
except Exception as e:
    print(f"Could not decode action: {e}")
```

### Variable Binding

```python
# Create variables
X = hrr.generate_vector()
Y = hrr.generate_vector()

# Create template: X loves Y
template = encoder.encode_structure({
    "agent": X,
    "action": symbols["loves"],
    "patient": Y
})

# Bind variables to values
substitution = hrr.bind(X, symbols["john"]) + hrr.bind(Y, symbols["mary"])

# Apply substitution (simplified)
instance = template + substitution  # This is a simplification

# In practice, you'd need more sophisticated substitution
```

## Sequence Processing

### Basic Sequence Encoding

```python
from cognitive_computing.hrr import SequenceEncoder

seq_encoder = SequenceEncoder(hrr)

# Create a sequence of words
words = ["the", "quick", "brown", "fox", "jumps"]
word_vectors = {word: hrr.generate_vector() for word in words}

# Add to cleanup
for word, vec in word_vectors.items():
    cleanup.add_item(word, vec)

# Encode sequence - IMPORTANT: Use "positional" not "position"
sequence_vectors = [word_vectors[w] for w in words]
encoded_seq = seq_encoder.encode_sequence(sequence_vectors, method="positional")
# Valid methods are: "positional", "chaining", or "temporal"

# Retrieve items by position
for i in range(len(words)):
    retrieved = seq_encoder.decode_position(encoded_seq, i)
    try:
        word, _, conf = cleanup.cleanup(retrieved)
        print(f"Position {i}: {word} (confidence: {conf:.3f})")
    except Exception as e:
        print(f"Position {i}: Could not retrieve - {e}")
```

### Sequence Patterns

```python
# Create number sequence
numbers = ["one", "two", "three", "four", "five"]
num_vectors = [hrr.generate_vector() for _ in numbers]

# Add to cleanup
for num, vec in zip(numbers, num_vectors):
    cleanup.add_item(num, vec)

# Encode multiple sequences
seq1 = seq_encoder.encode_sequence(num_vectors[:3])  # one, two, three
seq2 = seq_encoder.encode_sequence(num_vectors[1:4])  # two, three, four

# Find pattern similarity
similarity = hrr.similarity(seq1, seq2)
print(f"Sequence similarity: {similarity:.3f}")

# Sequence completion
partial = seq_encoder.encode_sequence(num_vectors[:2])  # one, two
# Logic to find best matching sequence and complete it
```

### Variable-Length Sequences

```python
# Handle sequences of different lengths
sequences = {
    "short": ["cat", "dog"],
    "medium": ["the", "cat", "sat", "on", "mat"],
    "long": ["once", "upon", "a", "time", "in", "a", "land", "far", "away"]
}

encoded_sequences = {}
for name, seq in sequences.items():
    # Get or create vectors
    vectors = []
    for word in seq:
        if not cleanup.has_item(word):
            vec = hrr.generate_vector()
            cleanup.add_item(word, vec)
            vectors.append(vec)
        else:
            vectors.append(cleanup.get_vector(word))
    
    # Encode
    encoded_sequences[name] = seq_encoder.encode_sequence(vectors)
    print(f"{name} sequence encoded ({len(seq)} items)")
```

## Hierarchical Structures

### Tree Encoding

```python
from cognitive_computing.hrr import HierarchicalEncoder

hier_encoder = HierarchicalEncoder(hrr)

# Create a simple tree structure
tree = {
    "value": "root",
    "left": {
        "value": "A",
        "left": {"value": "B"},
        "right": {"value": "C"}
    },
    "right": {
        "value": "D",
        "left": {"value": "E"},
        "right": {"value": "F"}
    }
}

# Encode the tree - items are auto-registered
encoded_tree = hier_encoder.encode_tree(tree)

# Query paths in the tree - IMPORTANT: Use decode_path not decode_subtree
paths = [
    ["left", "value"],      # Should retrieve "A"
    ["right", "left", "value"],  # Should retrieve "E"
    ["left", "right", "value"],  # Should retrieve "C"
]

for path in paths:
    try:
        result = hier_encoder.decode_path(encoded_tree, path)
        # Clean up result if it's a leaf value
        if isinstance(result, np.ndarray):
            # Attempt cleanup if we have a cleanup memory configured
            # Note: HierarchicalEncoder auto-registers items during encoding
            print(f"Path {path}: Retrieved vector successfully")
    except Exception as e:
        print(f"Path {path}: Could not decode - {e}")
```

### Organizational Hierarchy

```python
# Create an org chart
org_chart = {
    "name": "Company",
    "CEO": {
        "name": "Alice",
        "title": "Chief Executive",
        "reports": [
            {
                "name": "Bob",
                "title": "CTO",
                "reports": [
                    {"name": "Charlie", "title": "Engineer"},
                    {"name": "David", "title": "Engineer"}
                ]
            },
            {
                "name": "Eve",
                "title": "CFO",
                "reports": [
                    {"name": "Frank", "title": "Accountant"}
                ]
            }
        ]
    }
}

# Encode organizational structure
encoded_org = hier_encoder.encode_tree(org_chart)

# Query organization
# Find Bob's title
bob_title_path = ["CEO", "reports", 0, "title"]
# Note: Lists need special handling in practice
```

### File System Representation

```python
# Represent a file system
filesystem = {
    "type": "directory",
    "name": "/",
    "contents": {
        "home": {
            "type": "directory",
            "contents": {
                "user": {
                    "type": "directory",
                    "contents": {
                        "document.txt": {
                            "type": "file",
                            "size": 1024,
                            "modified": "2023-01-01"
                        },
                        "image.png": {
                            "type": "file",
                            "size": 2048576,
                            "modified": "2023-01-02"
                        }
                    }
                }
            }
        },
        "etc": {
            "type": "directory",
            "contents": {
                "config": {
                    "type": "file",
                    "size": 512
                }
            }
        }
    }
}

# Encode filesystem
encoded_fs = hier_encoder.encode_tree(filesystem)

# Navigate filesystem
# Find type of /home/user/document.txt
doc_path = ["contents", "home", "contents", "user", "contents", "document.txt", "type"]
```

## Analogical Reasoning

### Simple Analogies

```python
# A:B :: C:D analogies
def solve_analogy(hrr, a, b, c):
    """Solve A:B :: C:? analogy"""
    # Extract transformation from A to B
    transformation = hrr.unbind(b, a)
    
    # Apply to C
    d = hrr.bind(transformation, c)
    return d

# Example: small:large :: cold:?
small = hrr.generate_vector()
large = hrr.generate_vector()
cold = hrr.generate_vector()
hot = hrr.generate_vector()

# Add to cleanup
cleanup.add_item("small", small)
cleanup.add_item("large", large)
cleanup.add_item("cold", cold)
cleanup.add_item("hot", hot)

# Solve analogy
result = solve_analogy(hrr, small, large, cold)
name, _, conf = cleanup.cleanup(result)
print(f"small:large :: cold:{name} (confidence: {conf:.3f})")
```

### Relational Analogies

```python
# Create relational structure
def create_relation(hrr, subject, relation, object):
    """Create a relational structure"""
    subj_vec = hrr.generate_vector()
    rel_vec = hrr.generate_vector(method="unitary")
    obj_vec = hrr.generate_vector()
    
    return hrr.bind(rel_vec, hrr.bind(subj_vec, obj_vec))

# Example relations
relations = [
    ("dog", "chases", "cat"),
    ("cat", "chases", "mouse"),
    ("teacher", "teaches", "student"),
    ("student", "learns_from", "teacher")
]

# Encode relations
encoded_relations = []
for subj, rel, obj in relations:
    # Get or create vectors
    subj_vec = symbols.get(subj, hrr.generate_vector())
    rel_vec = symbols.get(rel, hrr.generate_vector(method="unitary"))
    obj_vec = symbols.get(obj, hrr.generate_vector())
    
    # Store for cleanup
    if subj not in symbols:
        symbols[subj] = subj_vec
        cleanup.add_item(subj, subj_vec)
    if rel not in symbols:
        symbols[rel] = rel_vec
        cleanup.add_item(rel, rel_vec)
    if obj not in symbols:
        symbols[obj] = obj_vec
        cleanup.add_item(obj, obj_vec)
    
    # Encode relation
    encoded = hrr.bind(rel_vec, hrr.bind(subj_vec, obj_vec))
    encoded_relations.append(encoded)
```

### Structure Mapping

```python
# Map between different domains
# Solar system domain
solar_system = {
    "sun": {"orbited_by": ["earth", "mars", "venus"]},
    "earth": {"orbited_by": ["moon"]},
    "relationships": ["orbits", "attracts", "illuminates"]
}

# Atom domain (simplified)
atom = {
    "nucleus": {"orbited_by": ["electron1", "electron2"]},
    "relationships": ["orbits", "attracts"]
}

# Create mapping function
def map_structures(hrr, source_domain, target_domain):
    """Map structures between domains"""
    mappings = {}
    
    # Find structural correspondences
    # This is simplified - real structure mapping is more complex
    
    # Map central objects
    if "sun" in source_domain and "nucleus" in target_domain:
        mappings["sun"] = "nucleus"
    
    # Map orbiting objects
    if "earth" in source_domain.get("sun", {}).get("orbited_by", []):
        mappings["earth"] = "electron1"
    
    return mappings

# Apply mapping
mappings = map_structures(hrr, solar_system, atom)
print("Structure mappings:", mappings)
```

## Working with Cleanup Memory

### Managing Large Vocabularies

```python
# Create a large vocabulary
vocabulary_size = 1000
vocabulary = {}

# Generate vectors for vocabulary
for i in range(vocabulary_size):
    word = f"word_{i}"
    vector = hrr.generate_vector()
    vocabulary[word] = vector
    cleanup.add_item(word, vector)

print(f"Added {vocabulary_size} items to cleanup memory")

# Test retrieval with noise
test_word = "word_42"
test_vector = vocabulary[test_word]

# Add noise
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
for noise_level in noise_levels:
    noise = np.random.normal(0, noise_level, hrr.dimension)
    noisy_vector = test_vector + noise
    
    retrieved, _, conf = cleanup.cleanup(noisy_vector)
    print(f"Noise {noise_level}: Retrieved '{retrieved}' (conf: {conf:.3f})")
```

### K-Nearest Neighbors

```python
# Find multiple similar items
query_vector = vocabulary["word_100"]

# Add some noise
query_vector += np.random.normal(0, 0.15, hrr.dimension)

# Find 5 nearest neighbors
neighbors = cleanup.find_closest(query_vector, k=5)

print("5 nearest neighbors:")
for name, similarity in neighbors:
    print(f"  {name}: {similarity:.3f}")
```

### Dynamic Vocabulary

```python
# Start with empty cleanup memory
dynamic_cleanup = CleanupMemory(
    CleanupMemoryConfig(threshold=0.25),
    dimension=hrr.dimension
)

# Add items dynamically
sentence = "the quick brown fox jumps over the lazy dog"
words = sentence.split()

for word in words:
    if not dynamic_cleanup.has_item(word):
        # Create new vector for unknown word
        vector = hrr.generate_vector()
        dynamic_cleanup.add_item(word, vector)
        print(f"Added new word: {word}")
    else:
        print(f"Word already known: {word}")

# Encode the sentence
word_vectors = [dynamic_cleanup.get_vector(word) for word in words]
sentence_encoding = seq_encoder.encode_sequence(word_vectors)
```

## Complex Applications

### Question Answering System

```python
# Simple QA system using HRR
class SimpleQA:
    def __init__(self, hrr_system):
        self.hrr = hrr_system
        self.encoder = RoleFillerEncoder(hrr_system)
        self.facts = []
        self.cleanup = CleanupMemory(
            CleanupMemoryConfig(threshold=0.3),
            hrr_system.dimension
        )
    
    def add_fact(self, subject, predicate, object):
        """Add a fact to the knowledge base"""
        # Get or create vectors
        subj_vec = self._get_vector(subject)
        pred_vec = self._get_vector(predicate, unitary=True)
        obj_vec = self._get_vector(object)
        
        # Encode fact
        fact = self.encoder.encode_structure({
            "subject": subj_vec,
            "predicate": pred_vec,
            "object": obj_vec
        })
        
        self.facts.append(fact)
    
    def _get_vector(self, item, unitary=False):
        """Get or create vector for item"""
        if not self.cleanup.has_item(item):
            vec = self.hrr.generate_vector(
                method="unitary" if unitary else "random"
            )
            self.cleanup.add_item(item, vec)
        return self.cleanup.get_vector(item)
    
    def query(self, subject=None, predicate=None, object=None):
        """Query the knowledge base"""
        results = []
        
        # Bundle all facts
        if not self.facts:
            return results
        
        knowledge = self.hrr.bundle(self.facts)
        
        # Query based on what's provided
        if subject and predicate and not object:
            # Find object given subject and predicate
            subj_vec = self.cleanup.get_vector(subject)
            pred_vec = self.cleanup.get_vector(predicate)
            
            # Create query
            query_vec = self.hrr.bind(pred_vec, subj_vec)
            
            # Extract from knowledge
            result_vec = self.hrr.unbind(knowledge, query_vec)
            
            # Cleanup
            obj_name, _, conf = self.cleanup.cleanup(result_vec)
            if conf > self.cleanup.config.threshold:
                results.append((subject, predicate, obj_name, conf))
        
        return results

# Use the QA system
qa = SimpleQA(hrr)

# Add facts
qa.add_fact("Paris", "is_capital_of", "France")
qa.add_fact("London", "is_capital_of", "UK")
qa.add_fact("Berlin", "is_capital_of", "Germany")
qa.add_fact("France", "is_in", "Europe")
qa.add_fact("UK", "is_in", "Europe")

# Query
results = qa.query(subject="Paris", predicate="is_capital_of")
for subj, pred, obj, conf in results:
    print(f"{subj} {pred} {obj} (confidence: {conf:.3f})")
```

### Semantic Memory Model

```python
# Semantic memory with categories
class SemanticMemory:
    def __init__(self, hrr_system):
        self.hrr = hrr_system
        self.categories = {}
        self.instances = {}
        self.properties = {}
        self.cleanup = CleanupMemory(
            CleanupMemoryConfig(threshold=0.3),
            hrr_system.dimension
        )
    
    def add_category(self, category, parent=None):
        """Add a category to the hierarchy"""
        cat_vec = self.hrr.generate_vector()
        self.categories[category] = {
            "vector": cat_vec,
            "parent": parent,
            "instances": []
        }
        self.cleanup.add_item(category, cat_vec)
    
    def add_instance(self, instance, category):
        """Add an instance to a category"""
        inst_vec = self.hrr.generate_vector()
        cat_vec = self.categories[category]["vector"]
        
        # Bind instance to category
        binding = self.hrr.bind(inst_vec, cat_vec)
        
        self.instances[instance] = {
            "vector": inst_vec,
            "category": category,
            "binding": binding,
            "properties": {}
        }
        
        self.categories[category]["instances"].append(instance)
        self.cleanup.add_item(instance, inst_vec)
    
    def add_property(self, instance, property_name, value):
        """Add a property to an instance"""
        if instance in self.instances:
            prop_vec = self.hrr.generate_vector(method="unitary")
            val_vec = self.hrr.generate_vector()
            
            # Store property
            self.instances[instance]["properties"][property_name] = {
                "property_vector": prop_vec,
                "value_vector": val_vec,
                "binding": self.hrr.bind(prop_vec, val_vec)
            }
            
            # Add to cleanup
            self.cleanup.add_item(f"{property_name}_{value}", val_vec)
    
    def is_a(self, instance, category):
        """Check if instance belongs to category"""
        if instance not in self.instances:
            return False
        
        inst_data = self.instances[instance]
        
        # Direct category check
        if inst_data["category"] == category:
            return True
        
        # Check parent categories (inheritance)
        current = inst_data["category"]
        while current in self.categories:
            if current == category:
                return True
            current = self.categories[current]["parent"]
        
        return False

# Create semantic memory
sem_mem = SemanticMemory(hrr)

# Build taxonomy
sem_mem.add_category("animal")
sem_mem.add_category("mammal", parent="animal")
sem_mem.add_category("bird", parent="animal")
sem_mem.add_category("dog", parent="mammal")
sem_mem.add_category("cat", parent="mammal")

# Add instances
sem_mem.add_instance("Fido", "dog")
sem_mem.add_instance("Whiskers", "cat")
sem_mem.add_instance("Tweety", "bird")

# Add properties
sem_mem.add_property("Fido", "color", "brown")
sem_mem.add_property("Fido", "size", "large")
sem_mem.add_property("Whiskers", "color", "black")
sem_mem.add_property("Tweety", "color", "yellow")

# Test inheritance
print(f"Fido is a dog: {sem_mem.is_a('Fido', 'dog')}")
print(f"Fido is a mammal: {sem_mem.is_a('Fido', 'mammal')}")
print(f"Fido is an animal: {sem_mem.is_a('Fido', 'animal')}")
print(f"Fido is a bird: {sem_mem.is_a('Fido', 'bird')}")
```

## Integration Examples

### Combining HRR with SDM

```python
from cognitive_computing.sdm import create_sdm

# Create SDM for robust storage
sdm = create_sdm(dimension=hrr.dimension, num_hard_locations=1000)

# Store HRR structures in SDM
structures = []

# Create some structures
for i in range(10):
    structure = encoder.encode_structure({
        "id": hrr.generate_vector(),
        "type": hrr.generate_vector(),
        "value": hrr.generate_vector()
    })
    structures.append(structure)
    
    # Store in SDM
    sdm.store(structure, structure)

# Retrieve with noise
test_structure = structures[5]
noisy_query = test_structure + np.random.normal(0, 0.2, hrr.dimension)

# Retrieve from SDM
retrieved = sdm.recall(noisy_query)

if retrieved is not None:
    similarity = hrr.similarity(test_structure, retrieved)
    print(f"Retrieved with similarity: {similarity:.3f}")
```

### Pipeline Processing

```python
# Create a processing pipeline
class CognitivePipeline:
    def __init__(self, dimension=1024):
        self.hrr = create_hrr(dimension=dimension)
        self.sdm = create_sdm(dimension=dimension)
        self.encoder = RoleFillerEncoder(self.hrr)
        self.seq_encoder = SequenceEncoder(self.hrr)
        self.cleanup = CleanupMemory(
            CleanupMemoryConfig(threshold=0.3),
            dimension
        )
    
    def process_text(self, text):
        """Process text through the pipeline"""
        words = text.lower().split()
        
        # Get word vectors
        word_vectors = []
        for word in words:
            if not self.cleanup.has_item(word):
                vec = self.hrr.generate_vector()
                self.cleanup.add_item(word, vec)
            word_vectors.append(self.cleanup.get_vector(word))
        
        # Encode as sequence
        sequence = self.seq_encoder.encode_sequence(word_vectors)
        
        # Store in SDM
        self.sdm.store(sequence, sequence)
        
        return sequence
    
    def find_similar(self, query_text, threshold=0.7):
        """Find similar stored sequences"""
        query_seq = self.process_text(query_text)
        
        # Retrieve from SDM
        retrieved = self.sdm.recall(query_seq)
        
        if retrieved is not None:
            similarity = self.hrr.similarity(query_seq, retrieved)
            if similarity > threshold:
                return retrieved, similarity
        
        return None, 0.0

# Use the pipeline
pipeline = CognitivePipeline()

# Process some sentences
sentences = [
    "the cat sat on the mat",
    "the dog played in the park",
    "the cat played with yarn",
    "the bird flew over the tree"
]

for sentence in sentences:
    pipeline.process_text(sentence)
    print(f"Processed: {sentence}")

# Find similar
query = "the cat played"
result, sim = pipeline.find_similar(query, threshold=0.5)
if result is not None:
    print(f"Found similar sequence with similarity: {sim:.3f}")
```

## Best Practices

### 1. Vector Generation

```python
# Use appropriate vector types
roles = hrr.generate_vector(method="unitary")  # For roles/relations
content = hrr.generate_vector(method="random")  # For content/fillers

# Normalize when needed
if not hrr.normalize:
    vector = vector / np.linalg.norm(vector)
```

### 2. Capacity Management

```python
# Monitor bundling capacity
def check_bundle_capacity(hrr, n_items):
    """Check if bundling n items is feasible"""
    test_items = [hrr.generate_vector() for _ in range(n_items)]
    bundle = hrr.bundle(test_items)
    
    # Check retrieval quality
    min_sim = 1.0
    for item in test_items:
        sim = hrr.similarity(bundle, item)
        min_sim = min(min_sim, sim)
    
    return min_sim > 0.1  # Threshold for reliable retrieval

# Test before bundling
if check_bundle_capacity(hrr, 10):
    print("Can reliably bundle 10 items")
else:
    print("Too many items for reliable bundling")
```

### 3. Error Handling

```python
# Robust retrieval with error handling
def safe_retrieve(hrr, structure, role, cleanup):
    """Safely retrieve and clean up a filler"""
    try:
        # Decode filler
        filler = encoder.decode_filler(structure, role)
        
        # Check if result is valid
        if np.isnan(filler).any():
            return None, 0.0
        
        # Cleanup - IMPORTANT: May fail if symbol not registered
        name, clean_vec, confidence = cleanup.cleanup(filler)
        
        # Verify confidence
        if confidence < cleanup.config.threshold:
            return None, confidence
        
        return name, confidence
        
    except Exception as e:
        print(f"Retrieval error: {e}")
        return None, 0.0

# Best practice: Always handle cleanup failures
def safe_cleanup(cleanup, vector, default="unknown"):
    """Safely cleanup a vector with fallback."""
    try:
        name, _, conf = cleanup.cleanup(vector)
        return name, conf
    except:
        # If cleanup fails, check if we need to register the symbol
        return default, 0.0
```

### 4. Performance Optimization

```python
# Batch operations for efficiency
def batch_encode_structures(encoder, structures_list):
    """Encode multiple structures efficiently"""
    encoded = []
    
    # Pre-compute common operations
    for structure_dict in structures_list:
        # Single encoding call
        enc = encoder.encode_structure(structure_dict)
        encoded.append(enc)
    
    return encoded

# Reuse vectors when possible
vector_cache = {}

def get_cached_vector(hrr, name, cache):
    """Get vector from cache or generate new"""
    if name not in cache:
        cache[name] = hrr.generate_vector()
    return cache[name]
```

## Conclusion

These examples demonstrate the versatility of HRR for:
- Symbolic reasoning and binding
- Sequential and hierarchical data
- Analogical reasoning
- Integration with other cognitive architectures

Key takeaways:
1. Use unitary vectors for roles and relations
2. Cleanup memory is essential for symbolic outputs
3. Monitor capacity limits for bundling
4. Combine with SDM for robust storage
5. Build modular pipelines for complex tasks

For more examples, see the example scripts in `examples/hrr/`.