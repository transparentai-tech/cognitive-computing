# SPA Examples Guide

This guide provides practical examples of using the Semantic Pointer Architecture (SPA) for various cognitive computing tasks.

## Table of Contents
1. [Basic Operations](#basic-operations)
2. [Vocabulary Management](#vocabulary-management)
3. [Module Usage](#module-usage)
4. [Action Selection](#action-selection)
5. [Production Systems](#production-systems)
6. [Cognitive Control](#cognitive-control)
7. [Complete Models](#complete-models)
8. [Advanced Techniques](#advanced-techniques)

## Basic Operations

### Creating and Manipulating Semantic Pointers

```python
from cognitive_computing.spa import Vocabulary, SemanticPointer
import numpy as np

# Create a vocabulary
vocab = Vocabulary(dimension=512)

# Create semantic pointers
vocab.create_pointer("COFFEE")
vocab.create_pointer("TEA")
vocab.create_pointer("HOT")
vocab.create_pointer("COLD")
vocab.create_pointer("MORNING")
vocab.create_pointer("EVENING")

# Binding operations (creating associations)
hot_coffee = vocab["HOT"] * vocab["COFFEE"]
cold_tea = vocab["COLD"] * vocab["TEA"]
morning_beverage = vocab["MORNING"] * vocab["COFFEE"]
evening_beverage = vocab["EVENING"] * vocab["TEA"]

# Unbinding operations (extracting components)
# What beverage is associated with morning?
morning_query = morning_beverage * ~vocab["MORNING"]
matches = vocab.cleanup(morning_query.vector, top_n=3)
print(f"Morning beverage: {matches[0][0]} (similarity: {matches[0][1]:.3f})")

# Bundling operations (superposition)
all_beverages = vocab["COFFEE"] + vocab["TEA"]
all_times = vocab["MORNING"] + vocab["EVENING"]

# Check what's in the bundle
coffee_similarity = all_beverages @ vocab["COFFEE"]
tea_similarity = all_beverages @ vocab["TEA"]
print(f"Bundle contains COFFEE: {coffee_similarity:.3f}")
print(f"Bundle contains TEA: {tea_similarity:.3f}")
```

### Complex Expressions

```python
# Parse complex expressions
vocab.create_pointer("LIKE")
vocab.create_pointer("PERSON")
vocab.create_pointer("ALICE")
vocab.create_pointer("BOB")

# Create structured representation: "Alice likes coffee"
alice_likes_coffee = vocab.parse("PERSON*ALICE + LIKE*COFFEE")

# Query: What does Alice like?
query = alice_likes_coffee * ~vocab["LIKE"]
matches = vocab.cleanup(query.vector, top_n=1)
print(f"Alice likes: {matches[0][0]}")

# Multiple relations
knowledge = vocab.parse("PERSON*ALICE + LIKE*COFFEE") + \
            vocab.parse("PERSON*BOB + LIKE*TEA")

# Query: What does Bob like?
bob_query = knowledge * vocab["PERSON"] * ~vocab["BOB"] * ~vocab["LIKE"]
matches = vocab.cleanup(bob_query.vector, top_n=1)
print(f"Bob likes: {matches[0][0]}")
```

## Vocabulary Management

### Building Large Vocabularies

```python
from cognitive_computing.spa import Vocabulary, analyze_vocabulary

# Create vocabulary with specific configuration
vocab = Vocabulary(dimension=1024)

# Add many related concepts
animals = ["CAT", "DOG", "BIRD", "FISH", "LION", "TIGER"]
colors = ["RED", "BLUE", "GREEN", "YELLOW", "BLACK", "WHITE"]
sizes = ["BIG", "SMALL", "MEDIUM", "TINY", "HUGE"]

for animal in animals:
    vocab.create_pointer(animal)
for color in colors:
    vocab.create_pointer(color)
for size in sizes:
    vocab.create_pointer(size)

# Create structured descriptions
red_cat = vocab["RED"] * vocab["CAT"]
big_dog = vocab["BIG"] * vocab["DOG"]
small_bird = vocab["SMALL"] * vocab["BIRD"]

# Analyze vocabulary quality
stats = analyze_vocabulary(vocab)
print(f"Vocabulary size: {stats['size']}")
print(f"Average similarity: {stats['mean_similarity']:.3f}")
print(f"Max similarity: {stats['max_similarity']:.3f}")
```

### Hierarchical Concepts

```python
# Create hierarchical relationships
vocab.create_pointer("ISA")  # "is a" relation
vocab.create_pointer("ANIMAL")
vocab.create_pointer("MAMMAL")
vocab.create_pointer("REPTILE")

# Define hierarchy
cat_isa_mammal = vocab["CAT"] * vocab["ISA"] * vocab["MAMMAL"]
mammal_isa_animal = vocab["MAMMAL"] * vocab["ISA"] * vocab["ANIMAL"]
dog_isa_mammal = vocab["DOG"] * vocab["ISA"] * vocab["MAMMAL"]

# Store in knowledge base
knowledge = cat_isa_mammal + mammal_isa_animal + dog_isa_mammal

# Query: What is a cat?
cat_isa = knowledge * vocab["CAT"] * vocab["ISA"]
matches = vocab.cleanup(cat_isa.vector, top_n=1)
print(f"CAT is a: {matches[0][0]}")

# Transitive inference: Is a cat an animal?
# First get what cat is (mammal), then what mammal is (animal)
cat_type = knowledge * vocab["CAT"] * vocab["ISA"]
cat_type_clean = vocab.cleanup(cat_type.vector, top_n=1)[0][0]
supertype = knowledge * vocab[cat_type_clean] * vocab["ISA"]
matches = vocab.cleanup(supertype.vector, top_n=1)
print(f"CAT is also a: {matches[0][0]}")
```

## Module Usage

### State Module with Feedback

```python
from cognitive_computing.spa import State, Vocabulary

# Create vocabulary and state module
vocab = Vocabulary(512)
vocab.create_pointer("GOAL")
vocab.create_pointer("TASK1")
vocab.create_pointer("TASK2")

# State module with high feedback (maintains state)
working_memory = State("working_memory", 512, feedback=0.9)

# Set initial state
working_memory.state = vocab["GOAL"].vector

# Update with partial input (blends with existing state)
partial_input = 0.3 * vocab["TASK1"].vector
working_memory.update({"input": partial_input})

# Check current state
matches = vocab.cleanup(working_memory.state, top_n=2)
for name, sim in matches:
    print(f"State contains {name}: {sim:.3f}")
```

### Memory Module for Associations

```python
from cognitive_computing.spa import Memory, Vocabulary

# Create memory module
vocab = Vocabulary(512)
memory = Memory("associative_memory", 512, capacity=20)

# Create concepts
for concept in ["PARIS", "FRANCE", "LONDON", "UK", "BERLIN", "GERMANY"]:
    vocab.create_pointer(concept)
vocab.create_pointer("CAPITAL")

# Store associations
memory.add_pair(vocab["FRANCE"], vocab["PARIS"])
memory.add_pair(vocab["UK"], vocab["LONDON"])
memory.add_pair(vocab["GERMANY"], vocab["BERLIN"])

# Also store reverse associations
memory.add_pair(vocab["PARIS"], vocab["FRANCE"])
memory.add_pair(vocab["LONDON"], vocab["UK"])
memory.add_pair(vocab["BERLIN"], vocab["GERMANY"])

# Query: What is the capital of France?
result = memory.recall(vocab["FRANCE"].vector)
if result is not None:
    matches = vocab.cleanup(result, top_n=1)
    print(f"Capital of France: {matches[0][0]}")

# Query: Which country is Berlin in?
result = memory.recall(vocab["BERLIN"].vector)
if result is not None:
    matches = vocab.cleanup(result, top_n=1)
    print(f"Berlin is in: {matches[0][0]}")
```

### Buffer and Gate Modules

```python
from cognitive_computing.spa import Buffer, Gate, Vocabulary
import numpy as np

vocab = Vocabulary(512)
vocab.create_pointer("INPUT1")
vocab.create_pointer("INPUT2")

# Create gated buffer
buffer = Buffer("gated_buffer", 512, gate_default=0.0)
gate = Gate("control_gate", 512, default_value=0.0)

# Initially gate is closed, buffer doesn't update
buffer.update({"input": vocab["INPUT1"].vector, "gate": gate.gate_signal})
print(f"Buffer norm (gate closed): {np.linalg.norm(buffer.state):.3f}")

# Open gate
gate.gate_signal = 1.0
buffer.update({"input": vocab["INPUT1"].vector, "gate": gate.gate_signal})
print(f"Buffer norm (gate open): {np.linalg.norm(buffer.state):.3f}")

# Check buffer contents
matches = vocab.cleanup(buffer.state, top_n=1)
print(f"Buffer contains: {matches[0][0]}")

# Partial gating
gate.gate_signal = 0.5
buffer.update({"input": vocab["INPUT2"].vector, "gate": gate.gate_signal})
matches = vocab.cleanup(buffer.state, top_n=2)
print("After partial gating:")
for name, sim in matches:
    print(f"  {name}: {sim:.3f}")
```

## Action Selection

### Basic Action Selection

```python
from cognitive_computing.spa import Action, BasalGanglia, Vocabulary
import numpy as np

# Create vocabulary
vocab = Vocabulary(512)
for state in ["HUNGRY", "THIRSTY", "TIRED"]:
    vocab.create_pointer(state)

# Current state
current_state = vocab["HUNGRY"].vector + 0.3 * vocab["THIRSTY"].vector

# Define actions with conditions
actions = [
    Action(
        name="eat",
        condition=lambda: float(np.dot(current_state, vocab["HUNGRY"].vector)),
        effect=lambda: print("Eating food...")
    ),
    Action(
        name="drink",
        condition=lambda: float(np.dot(current_state, vocab["THIRSTY"].vector)),
        effect=lambda: print("Drinking water...")
    ),
    Action(
        name="sleep",
        condition=lambda: float(np.dot(current_state, vocab["TIRED"].vector)),
        effect=lambda: print("Going to sleep...")
    )
]

# Create basal ganglia
bg = BasalGanglia(actions, mutual_inhibition=1.0, threshold=0.5)

# Evaluate actions
utilities = bg.update({})
selected = bg.get_selected_action(utilities)

if selected is not None:
    print(f"Selected action: {actions[selected].name}")
    actions[selected].effect()
```

### Dynamic Action Selection

```python
from cognitive_computing.spa import Action, BasalGanglia, Thalamus, State
import numpy as np

# Create modules
state = State("context", 512, feedback=0.8)
vocab = Vocabulary(512)

# Create context pointers
contexts = ["WORK", "HOME", "GYM"]
activities = ["TYPE", "RELAX", "EXERCISE"]
for ctx in contexts:
    vocab.create_pointer(ctx)
for act in activities:
    vocab.create_pointer(act)

# Context-dependent actions
def create_context_action(context, activity):
    return Action(
        name=f"{activity.lower()}_in_{context.lower()}",
        condition=lambda: float(state.state @ vocab[context].vector),
        effect=lambda: print(f"{activity} in {context}")
    )

# Create all combinations
actions = []
for ctx, act in zip(contexts, activities):
    actions.append(create_context_action(ctx, act))

# Create action selection system
bg = BasalGanglia(actions, threshold=0.3)

# Simulate changing contexts
for context in contexts:
    print(f"\nContext: {context}")
    state.state = vocab[context].vector
    
    utilities = bg.update({})
    selected = bg.get_selected_action(utilities)
    
    if selected is not None:
        actions[selected].effect()
```

## Production Systems

### Rule-Based Reasoning

```python
from cognitive_computing.spa import (
    Production, ProductionSystem, MatchCondition,
    AssignEffect, Vocabulary
)

# Create vocabulary
vocab = Vocabulary(512)
for concept in ["ANIMAL", "BIRD", "CAN_FLY", "PENGUIN", "EAGLE"]:
    vocab.create_pointer(concept)

# Create production system
prod_system = ProductionSystem(conflict_resolution="priority")

# Add production rules
# Rule 1: Birds can fly (default)
prod_system.add_production(Production(
    name="birds_fly",
    condition=MatchCondition("type", vocab["BIRD"].vector, threshold=0.7),
    effect=AssignEffect("ability", vocab["CAN_FLY"].vector),
    priority=1.0
))

# Rule 2: Penguins cannot fly (exception)
prod_system.add_production(Production(
    name="penguins_dont_fly",
    condition=MatchCondition("type", vocab["PENGUIN"].vector, threshold=0.7),
    effect=AssignEffect("ability", -vocab["CAN_FLY"].vector),  # Negation
    priority=2.0  # Higher priority overrides general rule
))

# Test the system
test_cases = [
    ("Eagle", vocab["EAGLE"].vector + vocab["BIRD"].vector),
    ("Penguin", vocab["PENGUIN"].vector + vocab["BIRD"].vector)
]

for name, type_vector in test_cases:
    state = {"type": type_vector, "ability": np.zeros(512)}
    
    # Find matching productions
    matches = prod_system.match(state)
    if matches:
        selected = prod_system.select(matches)
        print(f"\n{name}:")
        print(f"  Rule fired: {selected.name}")
        
        # Execute effect
        selected.effect.apply(state)
        
        # Check ability
        can_fly = state["ability"] @ vocab["CAN_FLY"].vector
        print(f"  Can fly: {'Yes' if can_fly > 0 else 'No'}")
```

### Multi-Step Reasoning

```python
from cognitive_computing.spa import Production, ProductionSystem, CompareCondition
import numpy as np

# Problem: Tower of Hanoi with 3 disks
vocab = Vocabulary(512)

# Create concepts
for disk in ["DISK1", "DISK2", "DISK3"]:  # 1 is smallest
    vocab.create_pointer(disk)
for peg in ["A", "B", "C"]:
    vocab.create_pointer(peg)
vocab.create_pointer("ON")
vocab.create_pointer("EMPTY")

# Production system for Tower of Hanoi
prod_system = ProductionSystem()

# Helper function to check if move is legal
def can_move(disk, from_peg, to_peg, state):
    # Simplified for example
    return True

# Rule: Move disk from A to B
prod_system.add_production(Production(
    name="move_disk1_A_to_B",
    condition=lambda state: state["disk1_on"] @ vocab["A"].vector > 0.7,
    effect=lambda state: state.update({"disk1_on": vocab["B"].vector}),
    priority=1.0
))

# Initial state: All disks on peg A
state = {
    "disk1_on": vocab["A"].vector,
    "disk2_on": vocab["A"].vector,
    "disk3_on": vocab["A"].vector,
    "goal": vocab["C"].vector  # Move all to C
}

# Run production system for multiple steps
for step in range(7):  # Minimum steps for 3 disks
    matches = prod_system.match(state)
    if matches:
        selected = prod_system.select(matches)
        print(f"Step {step + 1}: {selected.name}")
        selected.execute(state)
```

## Cognitive Control

### Working Memory Management

```python
from cognitive_computing.spa import CognitiveControl, SPAConfig, Vocabulary

# Create cognitive control system
config = SPAConfig(dimension=512)
vocab = Vocabulary(512)
control = CognitiveControl(512, config, vocab)

# Create task representations
tasks = ["READ_EMAIL", "WRITE_REPORT", "ATTEND_MEETING"]
for task in tasks:
    vocab.create_pointer(task)

# Simulate task management
print("Task Management Simulation:")

# Add tasks to working memory
control.push_task("READ_EMAIL")
control.push_task("WRITE_REPORT")
print(f"Current task: {control.current_task}")
print(f"Task stack: {control.task_stack}")

# Set attention to urgent item
control.set_attention("ATTEND_MEETING")
print(f"Attention on: ATTEND_MEETING")

# Process interruption
control.push_task("ATTEND_MEETING")  # Urgent interruption
print(f"After interruption: {control.current_task}")

# Complete current task
control.pop_task()
print(f"After completing meeting: {control.current_task}")
```

### Sequential Behavior

```python
from cognitive_computing.spa import Sequencing, SPAConfig, Vocabulary

# Create sequencing controller
config = SPAConfig(dimension=512)
vocab = Vocabulary(512)
sequencer = Sequencing(512, config, vocab)

# Define a morning routine sequence
morning_routine = [
    "WAKE_UP",
    "SHOWER", 
    "BREAKFAST",
    "BRUSH_TEETH",
    "GET_DRESSED",
    "LEAVE_HOME"
]

# Create pointers for each step
for step in morning_routine:
    vocab.create_pointer(step)

# Define the sequence
sequencer.define_sequence("morning_routine", morning_routine)

# Execute sequence
print("Morning Routine:")
sequencer.start_sequence("morning_routine")

while not sequencer.is_finished():
    current = sequencer.get_current_step()
    print(f"  Step {sequencer.sequence_index + 1}: {current}")
    
    # Simulate step completion
    sequencer.next_step()
    
    # Check state representation
    state_vec = sequencer.encode_state()
    matches = vocab.cleanup(state_vec, top_n=1)
    print(f"    State encodes: {matches[0][0]}")

print("Routine completed!")
```

## Complete Models

### Question Answering System

```python
from cognitive_computing.spa import (
    create_spa, Vocabulary, Memory, State,
    Action, BasalGanglia, Thalamus
)

# Create Q&A system
spa = create_spa(dimension=512)
vocab = spa.vocabulary

# Knowledge base concepts
facts = [
    ("PARIS", "CAPITAL", "FRANCE"),
    ("LONDON", "CAPITAL", "UK"),
    ("BERLIN", "CAPITAL", "GERMANY"),
    ("FRANCE", "CONTINENT", "EUROPE"),
    ("UK", "CONTINENT", "EUROPE"),
    ("GERMANY", "CONTINENT", "EUROPE")
]

# Create pointers
concepts = set()
for fact in facts:
    concepts.update(fact)
for concept in concepts:
    vocab.create_pointer(concept)
vocab.create_pointer("WHAT")
vocab.create_pointer("WHERE")

# Create modules
question = State("question", 512)
knowledge = Memory("knowledge", 512, capacity=50)
answer = State("answer", 512)

# Store facts in memory
for subject, relation, obj in facts:
    key = vocab[subject] * vocab[relation]
    knowledge.add_pair(key, vocab[obj])

# Define Q&A actions
def process_what_question():
    # Extract subject and relation from question
    q = question.state
    # Simple extraction (in practice would be more sophisticated)
    for subj, rel, _ in facts:
        if vocab[subj].vector @ q > 0.5 and vocab[rel].vector @ q > 0.5:
            # Lookup answer
            key = vocab[subj] * vocab[rel]
            result = knowledge.recall(key.vector)
            if result is not None:
                answer.state = result
                return

actions = [
    Action(
        "answer_what",
        lambda: question.state @ vocab["WHAT"].vector,
        process_what_question
    )
]

# Test the system
test_questions = [
    "WHAT*CAPITAL*FRANCE",
    "WHAT*CONTINENT*UK"
]

for q_str in test_questions:
    print(f"\nQuestion: {q_str}")
    question.state = vocab.parse(q_str).vector
    
    # Process question
    bg = BasalGanglia(actions, threshold=0.3)
    utilities = bg.update({})
    selected = bg.get_selected_action(utilities)
    
    if selected is not None:
        actions[selected].effect()
        
        # Get answer
        matches = vocab.cleanup(answer.state, top_n=1)
        if matches:
            print(f"Answer: {matches[0][0]}")
```

### Cognitive Agent

```python
from cognitive_computing.spa import SPAModel, ModelBuilder

# Define a cognitive agent model
model = SPAModel("cognitive_agent", dimension=512)

# Add modules
model.add_module("perception", "buffer")
model.add_module("working_memory", "state", feedback=0.9)
model.add_module("long_term_memory", "memory", capacity=1000)
model.add_module("motor", "buffer")
model.add_module("goal", "state", feedback=0.95)

# Add connections
model.connect("perception", "working_memory")
model.connect("working_memory", "long_term_memory")
model.connect("working_memory", "motor")
model.connect("goal", "working_memory", transform=0.5)  # Goal biases processing

# Add vocabulary
model.add_vocabulary("objects", ["BALL", "BOX", "CUP"])
model.add_vocabulary("actions", ["GRASP", "MOVE", "RELEASE"])
model.add_vocabulary("locations", ["LEFT", "RIGHT", "CENTER"])

# Define behavior rules
model.add_action(
    "grasp_ball",
    condition="perception.BALL > 0.5 and motor.EMPTY > 0.5",
    effect="motor.GRASP*BALL"
)

model.add_action(
    "move_to_goal",
    condition="motor.GRASP*BALL > 0.5 and goal.RIGHT > 0.5",
    effect="motor.MOVE*RIGHT"
)

# Build the model
builder = ModelBuilder()
agent = builder.build(model)

# Run simulation
print("Running cognitive agent simulation...")
agent.modules["perception"].state = agent.vocabulary["BALL"].vector
agent.modules["goal"].state = agent.vocabulary["RIGHT"].vector

for step in range(10):
    agent.step()
    
    # Check motor output
    motor_state = agent.modules["motor"].state
    matches = agent.vocabulary.cleanup(motor_state, top_n=1)
    if matches and matches[0][1] > 0.5:
        print(f"Step {step}: Motor command = {matches[0][0]}")
```

## Advanced Techniques

### Analogical Reasoning

```python
from cognitive_computing.spa import Vocabulary
import numpy as np

# Create vocabulary for analogical reasoning
vocab = Vocabulary(1024)  # Larger dimension for complex relationships

# Source domain: Solar system
for concept in ["SUN", "EARTH", "MOON", "ORBITS", "PLANET", "SATELLITE"]:
    vocab.create_pointer(concept)

# Target domain: Atom
for concept in ["NUCLEUS", "ELECTRON", "ATOM"]:
    vocab.create_pointer(concept)

# Encode source relationships
solar_system = vocab["EARTH"] * vocab["ORBITS"] * vocab["SUN"] + \
               vocab["MOON"] * vocab["ORBITS"] * vocab["EARTH"] + \
               vocab["EARTH"] * vocab["ISA"] * vocab["PLANET"] + \
               vocab["MOON"] * vocab["ISA"] * vocab["SATELLITE"]

# Create mapping between domains
# SUN -> NUCLEUS, EARTH -> ELECTRON, PLANET -> ?
mapping = vocab["SUN"] * vocab["NUCLEUS"] + \
          vocab["EARTH"] * vocab["ELECTRON"]

# Apply analogical reasoning
# What orbits the nucleus? (SUN->NUCLEUS, EARTH->?)
# First extract what orbits the sun
orbits_sun = solar_system * vocab["ORBITS"] * vocab["SUN"]
what_orbits_sun = vocab.cleanup(orbits_sun.vector, top_n=1)[0][0]  # EARTH

# Apply mapping
target = mapping * vocab[what_orbits_sun]
result = vocab.cleanup(target.vector, top_n=1)
print(f"In the atom model, {result[0][0]} orbits the NUCLEUS")

# What is an electron? (EARTH is a PLANET, ELECTRON is a ?)
earth_isa = solar_system * vocab["EARTH"] * vocab["ISA"]
earth_type = vocab.cleanup(earth_isa.vector, top_n=1)[0][0]  # PLANET

# Need to find target concept analogous to PLANET
# This would require learning or additional knowledge
```

### Compositional Semantics

```python
from cognitive_computing.spa import Vocabulary

# Create vocabulary for language processing
vocab = Vocabulary(1024)

# Basic concepts
for word in ["RED", "BLUE", "BIG", "SMALL", "BALL", "BOX", "AND", "THE"]:
    vocab.create_pointer(word)

# Grammatical roles
for role in ["AGENT", "PATIENT", "ACTION", "MODIFIER", "OBJECT"]:
    vocab.create_pointer(role)

# Compose "the big red ball"
big_red_ball = vocab["OBJECT"] * vocab["BALL"] + \
               vocab["MODIFIER"] * (vocab["BIG"] + vocab["RED"])

# Parse and analyze
print("Analyzing 'the big red ball':")

# What is the object?
obj_query = big_red_ball * vocab["OBJECT"]
obj_match = vocab.cleanup(obj_query.vector, top_n=1)
print(f"  Object: {obj_match[0][0]}")

# What are the modifiers?
mod_query = big_red_ball * vocab["MODIFIER"]
mod_matches = vocab.cleanup(mod_query.vector, top_n=2)
print("  Modifiers:")
for mod, sim in mod_matches:
    if sim > 0.3:
        print(f"    - {mod}")

# Compose "the small blue box"
small_blue_box = vocab["OBJECT"] * vocab["BOX"] + \
                 vocab["MODIFIER"] * (vocab["SMALL"] + vocab["BLUE"])

# Compare compositions
similarity = big_red_ball @ small_blue_box
print(f"\nSimilarity between phrases: {similarity:.3f}")

# Extract common structure
common_structure = (big_red_ball + small_blue_box) * vocab["MODIFIER"]
common_mods = vocab.cleanup(common_structure.vector, top_n=4)
print("\nAll modifiers across both phrases:")
for mod, sim in common_mods:
    if sim > 0.2:
        print(f"  - {mod}")
```

### Learning Associations

```python
from cognitive_computing.spa import Memory, Vocabulary
import numpy as np

# Create vocabulary and memory
vocab = Vocabulary(512)
memory = Memory("learnable_memory", 512, capacity=100)

# Training data: word associations
associations = [
    ("DOCTOR", "HOSPITAL"),
    ("TEACHER", "SCHOOL"),
    ("CHEF", "RESTAURANT"),
    ("PILOT", "AIRPLANE"),
    ("NURSE", "HOSPITAL"),  # Another hospital association
    ("STUDENT", "SCHOOL")   # Another school association
]

# Create all pointers
all_words = set()
for pair in associations:
    all_words.update(pair)
for word in all_words:
    vocab.create_pointer(word)

# Train the memory
print("Training associations:")
for word1, word2 in associations:
    memory.add_pair(vocab[word1], vocab[word2])
    print(f"  {word1} -> {word2}")

# Test recall
print("\nTesting recall:")
test_words = ["DOCTOR", "TEACHER", "NURSE"]
for word in test_words:
    result = memory.recall(vocab[word].vector)
    if result is not None:
        matches = vocab.cleanup(result, top_n=1)
        print(f"  {word} associates with: {matches[0][0]}")

# Test generalization
print("\nTesting generalization:")
# Create a new concept that's similar to existing ones
vocab.create_pointer("PROFESSOR")  # Similar to TEACHER
professor_vec = 0.8 * vocab["TEACHER"].vector + \
                0.2 * np.random.randn(512)
professor_vec = professor_vec / np.linalg.norm(professor_vec)
vocab.pointers["PROFESSOR"].vector = professor_vec

# See what PROFESSOR associates with
result = memory.recall(vocab["PROFESSOR"].vector)
if result is not None:
    matches = vocab.cleanup(result, top_n=1)
    print(f"  PROFESSOR associates with: {matches[0][0]} (generalized from TEACHER)")
```

## Best Practices

### 1. Dimension Selection
- Use 512-1024 dimensions for most applications
- Larger dimensions for more complex vocabularies
- Test capacity with your specific use case

### 2. Vocabulary Design
- Keep pointer names meaningful and consistent
- Group related concepts
- Use hierarchical organization for large vocabularies

### 3. Module Configuration
- Adjust feedback based on required memory duration
- Set appropriate capacity for memory modules
- Use gates for conditional processing

### 4. Action Selection
- Set threshold based on noise level
- Adjust mutual inhibition for competition strength
- Priority values should reflect importance

### 5. Performance Optimization
- Pre-compute frequently used bindings
- Cache cleanup results when possible
- Use appropriate data structures for your scale

### 6. Debugging Tips
- Visualize similarity matrices
- Track module states over time
- Log action selection utilities
- Verify vocabulary orthogonality

## Common Patterns

### Pattern 1: Fact Storage and Retrieval
```python
# Store: subject-relation-object triples
fact = vocab["SUBJECT"] * vocab["RELATION"] * vocab["OBJECT"]

# Query: What is the object for this subject-relation?
query = fact * vocab["SUBJECT"] * vocab["RELATION"]
answer = vocab.cleanup(query.vector)
```

### Pattern 2: Role-Filler Binding
```python
# Bind roles to fillers
event = vocab["AGENT"] * vocab["JOHN"] + \
        vocab["ACTION"] * vocab["GIVE"] + \
        vocab["PATIENT"] * vocab["MARY"] + \
        vocab["OBJECT"] * vocab["BOOK"]

# Extract who performed the action
agent = event * vocab["AGENT"]
```

### Pattern 3: Sequential Processing
```python
# Encode sequence position
sequence = vocab["POS1"] * vocab["A"] + \
           vocab["POS2"] * vocab["B"] + \
           vocab["POS3"] * vocab["C"]

# Retrieve item at position 2
item = sequence * vocab["POS2"]
```

### Pattern 4: Set Membership
```python
# Create a set representation
fruit_set = vocab["APPLE"] + vocab["BANANA"] + vocab["ORANGE"]

# Check membership
is_member = fruit_set @ vocab["APPLE"] > threshold
```

## Troubleshooting

### Issue: Poor cleanup accuracy
**Solution**: Increase vocabulary dimension or reduce vocabulary size

### Issue: Action selection not working
**Solution**: Check threshold values and condition functions

### Issue: Memory recall fails
**Solution**: Verify key-value pairs are properly normalized

### Issue: Binding depth limitations
**Solution**: Use intermediate cleanup steps or increase dimensions

### Issue: Module connections not working
**Solution**: Ensure dimensions match and transforms are correct

## Next Steps

- Explore the [API Reference](api_reference.md) for detailed method documentation
- Read the [Theory](theory.md) section for mathematical foundations
- Check [Performance](performance.md) for optimization strategies
- Review example scripts in `examples/spa/` directory