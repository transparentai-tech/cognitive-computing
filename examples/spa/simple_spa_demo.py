#!/usr/bin/env python3
"""
Simple SPA demonstration that works with the actual implementation.
"""

import numpy as np
from cognitive_computing.spa import (
    create_spa, create_vocabulary,
    SemanticPointer, Vocabulary, SPAConfig,
    State, Memory, Buffer, Gate,
    ActionSet, Action, BasalGanglia
)


def main():
    """Run simple SPA demonstrations."""
    print("=" * 60)
    print("Simple SPA (Semantic Pointer Architecture) Demo")
    print("=" * 60)
    
    # Create vocabulary
    print("\n=== Creating Vocabulary ===")
    vocab = create_vocabulary(dimension=256)
    
    # Add some concepts
    concepts = ["COFFEE", "TEA", "HOT", "COLD", "MORNING", "EVENING"]
    for concept in concepts:
        vocab.create_pointer(concept)
    print(f"Created {len(concepts)} semantic pointers")
    
    # Test similarity
    print("\n=== Testing Similarity ===")
    coffee = vocab["COFFEE"]
    tea = vocab["TEA"]
    similarity = coffee.similarity(tea)
    print(f"COFFEE ~ TEA: {similarity:.3f} (should be near 0)")
    
    # Test binding
    print("\n=== Testing Binding ===")
    hot_coffee = vocab["HOT"] * vocab["COFFEE"]
    print("Created HOT*COFFEE")
    
    # Test unbinding
    unbound = hot_coffee * ~vocab["HOT"]
    sim_to_coffee = unbound.similarity(coffee)
    print(f"(HOT*COFFEE)*~HOT ~ COFFEE: {sim_to_coffee:.3f} (should be near 1)")
    
    # Test bundling
    print("\n=== Testing Bundling ===")
    morning_drinks = vocab["COFFEE"] + vocab["TEA"]
    morning_drinks = morning_drinks.normalize()
    print(f"(COFFEE+TEA) ~ COFFEE: {morning_drinks.similarity(coffee):.3f}")
    print(f"(COFFEE+TEA) ~ TEA: {morning_drinks.similarity(tea):.3f}")
    
    # Create SPA model
    print("\n=== Creating SPA Model ===")
    spa = create_spa(dimension=256)
    print(f"Created SPA model with dimension {spa.config.dimension}")
    
    # Create modules
    print("\n=== Creating Modules ===")
    state = State("working_memory", 256)
    memory = Memory("semantic_memory", 256, capacity=10)
    buffer = Buffer("input", 256)
    gate = Gate("control", 256)
    
    print("Created State, Memory, Buffer, and Gate modules")
    
    # Store and retrieve from memory
    print("\n=== Memory Operations ===")
    memory.add_pair(vocab["MORNING"], vocab["COFFEE"])
    memory.add_pair(vocab["EVENING"], vocab["TEA"])
    print("Stored: MORNING->COFFEE, EVENING->TEA")
    
    # Retrieve
    retrieved = memory.recall(vocab["MORNING"].vector)
    if retrieved is not None:
        matches = vocab.cleanup(retrieved, top_n=1)
        if matches:
            print(f"Retrieved for MORNING: {matches[0][0]}")
    
    # Simple action selection
    print("\n=== Action Selection ===")
    actions = ActionSet()
    
    # Create a simple state for testing
    context = State("context", 256)
    
    # Define a simple action
    def print_coffee():
        print("   -> Selected: COFFEE")
        
    coffee_action = Action(
        condition=lambda: context.get_semantic_pointer(vocab).similarity(vocab["MORNING"]),
        effect=print_coffee,
        name="morning_coffee"
    )
    
    actions.add_action(coffee_action)
    
    # Test action
    context.state = vocab["MORNING"].vector
    utilities = actions.evaluate_all()
    print(f"Action utility for MORNING context: {utilities[0]:.3f}")
    
    selected = actions.select_action()
    if selected:
        print(f"Selected action: {selected.name}")
        selected.execute()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()