# Semantic Pointer Architecture (SPA) Overview

## Introduction

The Semantic Pointer Architecture (SPA) is a cognitive architecture that bridges symbolic and connectionist approaches to artificial intelligence. It provides a framework for building large-scale cognitive models using high-dimensional vector representations called semantic pointers.

## What is SPA?

SPA extends Holographic Reduced Representations (HRR) with:
- **Structured control flow** through action selection mechanisms
- **Cognitive modules** for different types of information processing
- **Production systems** for rule-based reasoning
- **Biologically-inspired** basal ganglia and thalamus models
- **High-level APIs** for declarative model specification

## Key Concepts

### Semantic Pointers
Semantic pointers are high-dimensional vectors that:
- Represent concepts, symbols, or states
- Support vector operations (binding, unbinding, bundling)
- Can be composed to represent complex structures
- Maintain similarity relationships between related concepts

### Vocabulary
A vocabulary is a collection of semantic pointers that:
- Stores named vectors representing concepts
- Provides parsing capabilities for expressions
- Supports cleanup operations to find nearest matches
- Enables symbol-like manipulation with neural implementation

### Cognitive Modules
SPA provides several types of modules:
- **State**: Maintains information with optional feedback
- **Memory**: Stores and retrieves associations
- **Buffer**: Temporary storage with gating control
- **Gate**: Controls information flow between modules
- **Compare**: Computes similarity between inputs

### Action Selection
The action selection system includes:
- **BasalGanglia**: Evaluates action utilities through competition
- **Thalamus**: Routes selected actions to appropriate modules
- **Cortex**: Executes effects of selected actions
- **ActionRules**: Declarative specification of conditions and effects

### Production Systems
SPA supports production-based reasoning:
- IF-THEN rules with vector-based conditions
- Pattern matching on semantic pointers
- Conflict resolution strategies
- Integration with action selection

## Why Use SPA?

### Advantages
1. **Unified Framework**: Combines symbolic and neural approaches
2. **Biological Plausibility**: Inspired by brain structures and processes
3. **Scalability**: Handles large vocabularies and complex models
4. **Flexibility**: Supports various cognitive tasks and domains
5. **Compositionality**: Build complex representations from simple ones

### Use Cases
- Cognitive modeling and simulation
- Question answering systems
- Sequential task execution
- Analogical reasoning
- Language processing
- Decision making systems

## Basic Example

```python
from cognitive_computing.spa import (
    create_spa, Vocabulary, State, Memory,
    BasalGanglia, Thalamus, Action
)

# Create vocabulary
vocab = Vocabulary(256)
vocab.create_pointer("COFFEE")
vocab.create_pointer("TEA")
vocab.create_pointer("HOT")
vocab.create_pointer("COLD")

# Create SPA model
spa = create_spa(dimension=256)

# Add modules
state = State("beverage", 256)
memory = Memory("preferences", 256)

# Store associations
memory.add_pair(vocab["COFFEE"], vocab["HOT"])
memory.add_pair(vocab["TEA"], vocab["COLD"])

# Create action rules
actions = [
    Action(
        name="drink_coffee",
        condition=lambda: state.state @ vocab["COFFEE"] > 0.5,
        effect=lambda: print("Enjoying hot coffee!")
    ),
    Action(
        name="drink_tea",
        condition=lambda: state.state @ vocab["TEA"] > 0.5,
        effect=lambda: print("Sipping cold tea!")
    )
]

# Create action selection system
bg = BasalGanglia(actions)
thal = Thalamus(bg, modules={'state': state})
```

## Comparison with Other Paradigms

### vs. Traditional Symbolic AI
- **Continuous representations** instead of discrete symbols
- **Graceful degradation** with noise
- **Similarity-based** operations
- **Neural implementation** possible

### vs. Deep Learning
- **Compositional structure** built-in
- **Interpretable** representations
- **Explicit reasoning** capabilities
- **Smaller data requirements**

### vs. Other VSA Approaches
- **Integrated control** mechanisms
- **Cognitive architecture** focus
- **Production system** integration
- **Biological inspiration**

## Architecture Components

### Core Components
1. **SemanticPointer**: Vector representation with operations
2. **Vocabulary**: Collection and management of pointers
3. **SPA**: Main system coordinating all components

### Module Types
1. **State**: Working memory with feedback
2. **Memory**: Associative storage
3. **Buffer**: Gated temporary storage
4. **Gate**: Information flow control
5. **Compare**: Similarity computation

### Control Systems
1. **BasalGanglia**: Action selection through competition
2. **Thalamus**: Routing of control signals
3. **Cortex**: Action execution
4. **CognitiveControl**: Executive functions

### Advanced Features
1. **Production System**: Rule-based reasoning
2. **Sequencing**: Sequential behavior control
3. **Model Compiler**: High-level model specification
4. **Neural Networks**: Spiking neural implementation

## Getting Started

### Installation
```bash
pip install cognitive-computing
```

### Quick Start
```python
from cognitive_computing.spa import create_spa, Vocabulary

# Create a simple SPA model
spa = create_spa(dimension=512)

# Create vocabulary
vocab = Vocabulary(512)
vocab.create_pointer("HELLO")
vocab.create_pointer("WORLD")

# Combine concepts
greeting = vocab["HELLO"] * vocab["WORLD"]
```

### Next Steps
- Read the [Theory](theory.md) documentation
- Explore [API Reference](api_reference.md)
- Try the [Examples](examples.md)
- Check [Performance](performance.md) considerations

## Summary

SPA provides a powerful framework for building cognitive models that combine the best of symbolic and connectionist approaches. With its biologically-inspired architecture, flexible module system, and high-level APIs, SPA enables the creation of sophisticated cognitive systems for various applications.

For detailed information on the mathematical foundations, see the [Theory](theory.md) documentation. For practical usage, refer to the [Examples](examples.md) guide.