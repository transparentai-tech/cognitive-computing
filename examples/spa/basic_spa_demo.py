#!/usr/bin/env python3
"""
Basic SPA (Semantic Pointer Architecture) Demonstration.

This example introduces the fundamental concepts of SPA:
- Creating and manipulating semantic pointers
- Building vocabularies
- Binding and unbinding operations
- Module interactions
- Simple action selection

SPA provides a cognitive architecture that bridges symbolic and neural
computation through semantic pointers and biologically-inspired control.
"""

import numpy as np
from cognitive_computing.spa import (
    create_spa, create_vocabulary,
    SemanticPointer, Vocabulary, SPAConfig,
    State, Memory, Buffer, Gate, Compare,
    ActionSet, Action, BasalGanglia, Thalamus
)
from cognitive_computing.spa.visualizations import (
    plot_similarity_matrix, plot_action_selection
)
import matplotlib.pyplot as plt


def demonstrate_semantic_pointers():
    """Demonstrate basic semantic pointer operations."""
    print("\n=== Semantic Pointer Basics ===")
    
    # Create a vocabulary
    vocab = create_vocabulary(dimensions=256)
    
    # Create some semantic pointers
    vocab.create_pointer("COFFEE")
    vocab.create_pointer("TEA")
    vocab.create_pointer("HOT")
    vocab.create_pointer("COLD")
    vocab.create_pointer("MORNING")
    vocab.create_pointer("EVENING")
    
    # Demonstrate similarity
    print("\n1. Semantic Pointer Similarity:")
    coffee = vocab["COFFEE"]
    tea = vocab["TEA"]
    similarity = coffee.similarity(tea)
    print(f"   COFFEE ~ TEA: {similarity:.3f} (should be near 0 for random vectors)")
    
    # Demonstrate binding (circular convolution)
    print("\n2. Binding Operations:")
    hot_coffee = vocab["HOT"] * vocab["COFFEE"]
    cold_tea = vocab["COLD"] * vocab["TEA"]
    
    # Check similarity after binding
    print(f"   HOT*COFFEE ~ COLD*TEA: {hot_coffee.similarity(cold_tea):.3f}")
    print(f"   HOT*COFFEE ~ COFFEE: {hot_coffee.similarity(coffee):.3f}")
    
    # Demonstrate unbinding
    print("\n3. Unbinding Operations:")
    # To get COFFEE from HOT*COFFEE, multiply by ~HOT (inverse of HOT)
    unbound_coffee = hot_coffee * ~vocab["HOT"]
    print(f"   (HOT*COFFEE)*~HOT ~ COFFEE: {unbound_coffee.similarity(coffee):.3f}")
    print(f"   Should be close to 1.0")
    
    # Demonstrate bundling (superposition)
    print("\n4. Bundling Operations:")
    morning_drinks = vocab["COFFEE"] + vocab["TEA"]
    morning_drinks = morning_drinks.normalize()
    
    print(f"   (COFFEE+TEA) ~ COFFEE: {morning_drinks.similarity(coffee):.3f}")
    print(f"   (COFFEE+TEA) ~ TEA: {morning_drinks.similarity(tea):.3f}")
    print(f"   Both should be positive (around 0.7)")
    
    # Cleanup demonstration
    print("\n5. Cleanup Memory:")
    # Add noise to coffee vector
    noisy_coffee = coffee.vector + 0.3 * np.random.randn(256)
    noisy_coffee_sp = SemanticPointer(noisy_coffee, vocabulary=vocab)
    
    # Find closest match
    matches = vocab.cleanup(noisy_coffee, top_n=3)
    print(f"   Cleanup results for noisy COFFEE:")
    for name, sim in matches:
        print(f"   - {name}: {sim:.3f}")
    
    return vocab


def demonstrate_spa_modules():
    """Demonstrate SPA module interactions."""
    print("\n\n=== SPA Module Interactions ===")
    
    # Create configuration
    config = SPAConfig(dimensions=128)
    
    # Create modules
    state = State("working_memory", 128, feedback=0.9)
    memory = Memory("semantic_memory", 128, capacity=10)
    buffer = Buffer("input_buffer", 128)
    gate = Gate("gate", 128)
    compare = Compare("similarity", 128)
    
    print("\n1. Module Types:")
    print(f"   - State: Maintains information with feedback")
    print(f"   - Memory: Stores and retrieves associations")
    print(f"   - Buffer: Temporary storage with gating")
    print(f"   - Gate: Controls information flow")
    print(f"   - Compare: Computes similarity between inputs")
    
    # Create vocabulary for the modules
    vocab = Vocabulary(128)
    vocab.create_pointer("CAT")
    vocab.create_pointer("DOG")
    vocab.create_pointer("ANIMAL")
    vocab.create_pointer("PET")
    
    # Store associations in memory
    print("\n2. Memory Storage:")
    memory.add_pair(vocab["CAT"], vocab["ANIMAL"])
    memory.add_pair(vocab["DOG"], vocab["ANIMAL"])
    memory.add_pair(vocab["CAT"], vocab["PET"])
    memory.add_pair(vocab["DOG"], vocab["PET"])
    print("   Stored: CAT->ANIMAL, DOG->ANIMAL, CAT->PET, DOG->PET")
    
    # Test retrieval
    print("\n3. Memory Retrieval:")
    retrieved = memory.recall(vocab["CAT"].vector)
    if retrieved is not None:
        # Find what was retrieved
        matches = vocab.cleanup(retrieved, top_n=2)
        print(f"   Retrieved from CAT:")
        for name, sim in matches:
            print(f"   - {name}: {sim:.3f}")
    
    # Demonstrate state module with feedback
    print("\n4. State Module with Feedback:")
    state.set_state(vocab["DOG"].vector)
    print("   Initial state: DOG")
    
    # Update with partial input
    partial_input = 0.3 * vocab["CAT"].vector
    state._state = state._state * 0.9 + partial_input * 0.1
    
    # Check what's in state
    matches = vocab.cleanup(state.state, top_n=2)
    print("   After partial CAT input:")
    for name, sim in matches:
        print(f"   - {name}: {sim:.3f}")
    
    return vocab, state, memory


def demonstrate_action_selection():
    """Demonstrate action selection with basal ganglia and thalamus."""
    print("\n\n=== Action Selection System ===")
    
    # Create vocabulary
    vocab = Vocabulary(128)
    vocab.create_pointer("HUNGRY")
    vocab.create_pointer("THIRSTY")
    vocab.create_pointer("TIRED")
    vocab.create_pointer("EAT")
    vocab.create_pointer("DRINK")
    vocab.create_pointer("SLEEP")
    
    # Create state module for context
    context = State("context", 128)
    
    # Create action set
    action_set = ActionSet()
    
    # Define actions with conditions
    # Action 1: If HUNGRY then EAT
    eat_action = Action(
        condition=lambda: context.get_semantic_pointer(vocab).similarity(vocab["HUNGRY"]),
        effect=lambda: print("   -> Executing: EAT"),
        name="eat_when_hungry"
    )
    
    # Action 2: If THIRSTY then DRINK
    drink_action = Action(
        condition=lambda: context.get_semantic_pointer(vocab).similarity(vocab["THIRSTY"]),
        effect=lambda: print("   -> Executing: DRINK"),
        name="drink_when_thirsty"
    )
    
    # Action 3: If TIRED then SLEEP
    sleep_action = Action(
        condition=lambda: context.get_semantic_pointer(vocab).similarity(vocab["TIRED"]),
        effect=lambda: print("   -> Executing: SLEEP"),
        name="sleep_when_tired"
    )
    
    action_set.add_action(eat_action)
    action_set.add_action(drink_action)
    action_set.add_action(sleep_action)
    
    # Create basal ganglia for action selection
    config = SPAConfig(dimensions=128, threshold=0.5)
    bg = BasalGanglia(action_set, config)
    
    # Test different contexts
    print("\n1. Context: HUNGRY")
    context.set_state(vocab["HUNGRY"].vector)
    selected = action_set.select_action("max")
    if selected:
        print(f"   Selected action: {selected.name}")
        selected.execute()
    
    print("\n2. Context: THIRSTY")
    context.set_state(vocab["THIRSTY"].vector)
    selected = action_set.select_action("max")
    if selected:
        print(f"   Selected action: {selected.name}")
        selected.execute()
    
    print("\n3. Context: HUNGRY + THIRSTY (conflict)")
    mixed_state = (vocab["HUNGRY"] + vocab["THIRSTY"]).normalize()
    context.set_state(mixed_state.vector)
    
    # Evaluate all actions
    utilities = action_set.evaluate_all()
    print("   Action utilities:")
    for i, action in enumerate(action_set.actions):
        print(f"   - {action.name}: {utilities[i]:.3f}")
    
    selected = action_set.select_action("max")
    if selected:
        print(f"   Winner: {selected.name}")
        selected.execute()
    
    # Demonstrate action dynamics over time
    print("\n4. Action Selection Dynamics:")
    history = []
    states = [
        vocab["HUNGRY"].vector,
        vocab["THIRSTY"].vector,
        vocab["TIRED"].vector,
        (vocab["HUNGRY"] + vocab["THIRSTY"]).normalize().vector,
        (vocab["THIRSTY"] + vocab["TIRED"]).normalize().vector
    ]
    
    for state in states:
        context.set_state(state)
        utilities = action_set.evaluate_all()
        history.append(utilities)
    
    history = np.array(history)
    
    # Plot action selection dynamics
    fig, ax = plot_action_selection(
        bg, history,
        action_labels=["Eat", "Drink", "Sleep"],
        threshold=0.5
    )
    plt.title("Action Selection Over Different Contexts")
    plt.show()
    
    return vocab, action_set


def demonstrate_complex_behavior():
    """Demonstrate more complex SPA behavior with multiple modules."""
    print("\n\n=== Complex SPA Behavior ===")
    
    # Create a simple cognitive model
    config = SPAConfig(dimensions=256)
    spa = create_spa(config)
    
    # Create vocabulary
    vocab = spa.vocabulary
    
    # Add concepts
    concepts = ["QUESTION", "STATEMENT", "ANSWER", 
                "WHAT", "WHERE", "WHO",
                "PARIS", "FRANCE", "CAPITAL"]
    
    for concept in concepts:
        vocab.create_pointer(concept)
    
    # Create bindings for facts
    facts = {
        "capital_of_france": vocab["CAPITAL"] * vocab["FRANCE"] * vocab["PARIS"],
        "what_capital": vocab["WHAT"] * vocab["CAPITAL"],
        "where_paris": vocab["WHERE"] * vocab["PARIS"]
    }
    
    print("\n1. Stored Facts:")
    print("   - CAPITAL * FRANCE * PARIS (Paris is capital of France)")
    print("   - WHAT * CAPITAL (What is the capital?)")
    print("   - WHERE * PARIS (Where is Paris?)")
    
    # Query processing
    print("\n2. Query Processing:")
    
    # Query: What is the capital of France?
    query = vocab["WHAT"] * vocab["CAPITAL"] * vocab["FRANCE"]
    print("   Query: WHAT * CAPITAL * FRANCE")
    
    # Process: Unbind WHAT and CAPITAL to get answer
    # (WHAT * CAPITAL * FRANCE) * ~WHAT * ~CAPITAL = FRANCE
    # Then use fact: CAPITAL * FRANCE * PARIS
    step1 = query * ~vocab["WHAT"] * ~vocab["CAPITAL"]  # Gets FRANCE
    
    # Now bind with CAPITAL to prepare for lookup
    step2 = vocab["CAPITAL"] * step1  # CAPITAL * FRANCE
    
    # Compare with stored fact
    similarity = step2.similarity(facts["capital_of_france"])
    print(f"   Query matches stored fact with similarity: {similarity:.3f}")
    
    # Extract answer
    answer = facts["capital_of_france"] * ~vocab["CAPITAL"] * ~vocab["FRANCE"]
    
    # Cleanup to get clear answer
    matches = vocab.cleanup(answer.vector, top_n=3)
    print("   Answer:")
    for name, sim in matches:
        if sim > 0.3:
            print(f"   - {name}: {sim:.3f}")
    
    # Visualize vocabulary structure
    print("\n3. Vocabulary Structure:")
    fig, ax = plot_similarity_matrix(
        vocab,
        subset=["WHAT", "WHERE", "CAPITAL", "FRANCE", "PARIS"],
        annotate=True
    )
    plt.title("Semantic Pointer Similarity Matrix")
    plt.tight_layout()
    plt.show()
    
    return spa, vocab


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("SPA (Semantic Pointer Architecture) Basic Demo")
    print("=" * 60)
    
    # Run demonstrations
    vocab1 = demonstrate_semantic_pointers()
    vocab2, state, memory = demonstrate_spa_modules()
    vocab3, action_set = demonstrate_action_selection()
    spa, vocab4 = demonstrate_complex_behavior()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nKey Concepts Demonstrated:")
    print("- Semantic pointers: high-dimensional vectors representing concepts")
    print("- Binding: combining concepts with circular convolution")
    print("- Unbinding: extracting components with inverse operation")
    print("- Bundling: superposition of multiple concepts")
    print("- Cleanup: finding nearest semantic pointer to noisy input")
    print("- Modules: State, Memory, Buffer, Gate, Compare")
    print("- Action selection: basal ganglia-inspired competition")
    print("- Complex reasoning: multi-step binding/unbinding operations")
    print("=" * 60)


if __name__ == "__main__":
    main()