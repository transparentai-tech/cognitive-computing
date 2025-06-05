#!/usr/bin/env python3
"""
Production System Demonstration using SPA.

This example demonstrates rule-based reasoning with production systems:
- Defining production rules (IF-THEN statements)
- Pattern matching with semantic pointers
- Conflict resolution strategies
- Forward chaining inference
- Learning and adaptation in production systems
- Integration with SPA modules

Production systems provide a bridge between symbolic AI and neural
implementation through SPA's semantic pointer architecture.
"""

import numpy as np
import matplotlib.pyplot as plt
from cognitive_computing.spa import (
    SPAConfig, Vocabulary,
    State, Memory, Buffer,
    Production, ProductionSystem, 
    Condition, Effect, ConditionalModule,
    parse_production_rules
)
from cognitive_computing.spa.visualizations import visualize_production_flow
from cognitive_computing.spa.utils import analyze_production_system


def create_animal_classification_system():
    """Create a production system for animal classification."""
    print("\n=== Animal Classification Production System ===")
    
    # Create vocabulary
    vocab = Vocabulary(256)
    
    # Features
    features = [
        "HAS_FUR", "HAS_FEATHERS", "HAS_SCALES", "HAS_WINGS",
        "LAYS_EGGS", "GIVES_MILK", "SWIMS", "FLIES", "WALKS",
        "WARM_BLOODED", "COLD_BLOODED", "HAS_BACKBONE"
    ]
    
    # Classes
    classes = [
        "MAMMAL", "BIRD", "REPTILE", "FISH", "AMPHIBIAN",
        "DOG", "CAT", "EAGLE", "PENGUIN", "SNAKE", "SHARK"
    ]
    
    # Add all to vocabulary
    for item in features + classes:
        vocab.create_pointer(item)
    
    # Create modules
    observation = State("observation", 256)
    classification = State("classification", 256)
    memory = Memory("knowledge", 256, capacity=20)
    
    # Create production system
    ps = ProductionSystem()
    
    print("\n1. Defining Classification Rules:")
    
    # Rule 1: If has fur and gives milk, then mammal
    rule1 = Production(
        name="classify_mammal",
        condition=Condition(
            lambda: observation.get_semantic_pointer(vocab).similarity(vocab["HAS_FUR"]) > 0.5 and
                   observation.get_semantic_pointer(vocab).similarity(vocab["GIVES_MILK"]) > 0.5,
            "has fur AND gives milk"
        ),
        effect=Effect(
            lambda: setattr(classification, 'state', vocab["MAMMAL"].vector),
            "classify as MAMMAL"
        ),
        priority=1.0
    )
    
    # Rule 2: If has feathers and has wings, then bird
    rule2 = Production(
        name="classify_bird",
        condition=Condition(
            lambda: observation.get_semantic_pointer(vocab).similarity(vocab["HAS_FEATHERS"]) > 0.5 and
                   observation.get_semantic_pointer(vocab).similarity(vocab["HAS_WINGS"]) > 0.5,
            "has feathers AND has wings"
        ),
        effect=Effect(
            lambda: setattr(classification, 'state', vocab["BIRD"].vector),
            "classify as BIRD"
        ),
        priority=1.0
    )
    
    # Rule 3: If has scales and cold-blooded, then reptile
    rule3 = Production(
        name="classify_reptile",
        condition=Condition(
            lambda: observation.get_semantic_pointer(vocab).similarity(vocab["HAS_SCALES"]) > 0.5 and
                   observation.get_semantic_pointer(vocab).similarity(vocab["COLD_BLOODED"]) > 0.5,
            "has scales AND cold-blooded"
        ),
        effect=Effect(
            lambda: setattr(classification, 'state', vocab["REPTILE"].vector),
            "classify as REPTILE"
        ),
        priority=1.0
    )
    
    # More specific rules (higher priority)
    
    # Rule 4: If mammal and barks, then dog
    rule4 = Production(
        name="identify_dog",
        condition=Condition(
            lambda: classification.get_semantic_pointer(vocab).similarity(vocab["MAMMAL"]) > 0.5 and
                   observation.get_semantic_pointer(vocab).similarity(vocab["WALKS"]) > 0.5,
            "is mammal AND walks"
        ),
        effect=Effect(
            lambda: setattr(classification, 'state', (vocab["DOG"] + vocab["MAMMAL"]).normalize().vector),
            "identify as DOG"
        ),
        priority=2.0  # Higher priority for specific identification
    )
    
    # Add rules to system
    ps.add_production(rule1)
    ps.add_production(rule2)
    ps.add_production(rule3)
    ps.add_production(rule4)
    
    print(f"   Added {len(ps.productions)} classification rules")
    
    # Set context
    ps.set_context({
        "observation": observation,
        "classification": classification,
        "memory": memory,
        "vocab": vocab
    })
    
    return ps, observation, classification, vocab


def demonstrate_forward_chaining():
    """Demonstrate forward chaining inference."""
    print("\n\n=== Forward Chaining Inference ===")
    
    # Create the classification system
    ps, observation, classification, vocab = create_animal_classification_system()
    
    print("\n1. Test Case: Furry animal that gives milk")
    
    # Set observation
    obs = vocab["HAS_FUR"] + vocab["GIVES_MILK"] + vocab["WARM_BLOODED"]
    observation.state = obs.normalize().vector
    
    # Run inference
    print("   Observations: HAS_FUR, GIVES_MILK, WARM_BLOODED")
    print("   Running inference...")
    
    # Step through production system
    fired = ps.step()
    if fired:
        print(f"   Fired: {fired.name}")
        
        # Check classification
        result = classification.get_semantic_pointer(vocab)
        matches = vocab.cleanup(result.vector, top_n=3)
        print("   Classification:")
        for name, sim in matches:
            if sim > 0.3:
                print(f"   - {name}: {sim:.3f}")
    
    print("\n2. Test Case: Feathered animal with wings")
    
    # Reset and set new observation
    classification.state = np.zeros(256)
    obs = vocab["HAS_FEATHERS"] + vocab["HAS_WINGS"] + vocab["LAYS_EGGS"]
    observation.state = obs.normalize().vector
    
    print("   Observations: HAS_FEATHERS, HAS_WINGS, LAYS_EGGS")
    fired = ps.step()
    if fired:
        print(f"   Fired: {fired.name}")
        
        result = classification.get_semantic_pointer(vocab)
        matches = vocab.cleanup(result.vector, top_n=3)
        print("   Classification:")
        for name, sim in matches:
            if sim > 0.3:
                print(f"   - {name}: {sim:.3f}")
    
    print("\n3. Chaining Multiple Rules:")
    
    # First classify as mammal, then as dog
    classification.state = np.zeros(256)
    obs = vocab["HAS_FUR"] + vocab["GIVES_MILK"] + vocab["WALKS"]
    observation.state = obs.normalize().vector
    
    print("   Observations: HAS_FUR, GIVES_MILK, WALKS")
    
    # Run multiple cycles
    for cycle in range(3):
        fired = ps.step()
        if fired:
            print(f"   Cycle {cycle + 1}: Fired {fired.name}")
            
            result = classification.get_semantic_pointer(vocab)
            matches = vocab.cleanup(result.vector, top_n=2)
            if matches:
                print(f"   Current classification: {matches[0][0]}")
    
    return ps, vocab


def demonstrate_conflict_resolution():
    """Demonstrate conflict resolution in production systems."""
    print("\n\n=== Conflict Resolution Strategies ===")
    
    # Create vocabulary
    vocab = Vocabulary(128)
    
    # Stimuli and responses
    items = ["STIMULUS_A", "STIMULUS_B", "RESPONSE_1", "RESPONSE_2", "URGENT", "NORMAL"]
    for item in items:
        vocab.create_pointer(item)
    
    # Create modules
    input_state = State("input", 128)
    output_state = State("output", 128)
    
    # Create production system
    ps = ProductionSystem()
    
    print("\n1. Conflicting Rules:")
    
    # Rule 1: A -> Response 1
    rule1 = Production(
        name="a_to_1",
        condition=Condition(
            lambda: input_state.get_semantic_pointer(vocab).similarity(vocab["STIMULUS_A"]) > 0.5,
            "stimulus is A"
        ),
        effect=Effect(
            lambda: setattr(output_state, 'state', vocab["RESPONSE_1"].vector),
            "respond with 1"
        ),
        priority=1.0,
        specificity=1  # General rule
    )
    
    # Rule 2: A + Urgent -> Response 2
    rule2 = Production(
        name="a_urgent_to_2",
        condition=Condition(
            lambda: input_state.get_semantic_pointer(vocab).similarity(vocab["STIMULUS_A"]) > 0.5 and
                   input_state.get_semantic_pointer(vocab).similarity(vocab["URGENT"]) > 0.5,
            "stimulus is A AND urgent"
        ),
        effect=Effect(
            lambda: setattr(output_state, 'state', vocab["RESPONSE_2"].vector),
            "respond with 2"
        ),
        priority=1.0,
        specificity=2  # More specific rule
    )
    
    ps.add_production(rule1)
    ps.add_production(rule2)
    
    ps.set_context({
        "input": input_state,
        "output": output_state,
        "vocab": vocab
    })
    
    print("   Rule 1: IF A THEN Response_1 (specificity=1)")
    print("   Rule 2: IF A AND Urgent THEN Response_2 (specificity=2)")
    
    # Test with just A
    print("\n2. Test with A only:")
    input_state.state = vocab["STIMULUS_A"].vector
    
    # Check which rules match
    utilities = ps.evaluate_all()
    print("   Matching rules:")
    for prod, util in utilities:
        print(f"   - {prod.name}: utility={util:.3f}")
    
    # Resolve conflict
    fired = ps.step()
    if fired:
        print(f"   Winner: {fired.name} (by priority/specificity)")
    
    # Test with A + Urgent
    print("\n3. Test with A + Urgent:")
    input_state.state = (vocab["STIMULUS_A"] + vocab["URGENT"]).normalize().vector
    
    utilities = ps.evaluate_all()
    print("   Matching rules:")
    for prod, util in utilities:
        print(f"   - {prod.name}: utility={util:.3f}")
    
    fired = ps.step()
    if fired:
        print(f"   Winner: {fired.name} (more specific)")
    
    # Demonstrate priority-based resolution
    print("\n4. Priority-Based Resolution:")
    
    # Add high-priority override rule
    rule3 = Production(
        name="emergency_override",
        condition=Condition(
            lambda: input_state.get_semantic_pointer(vocab).similarity(vocab["URGENT"]) > 0.7,
            "very urgent"
        ),
        effect=Effect(
            lambda: print("   EMERGENCY OVERRIDE ACTIVATED"),
            "emergency response"
        ),
        priority=10.0  # High priority
    )
    
    ps.add_production(rule3)
    
    # Test with urgent input
    print("   Added emergency rule with priority=10.0")
    input_state.state = vocab["URGENT"].vector
    
    fired = ps.step()
    if fired:
        print(f"   Winner: {fired.name} (highest priority)")
        fired.execute()
    
    return ps, vocab


def demonstrate_learning_productions():
    """Demonstrate learning in production systems."""
    print("\n\n=== Learning in Production Systems ===")
    
    # Create vocabulary
    vocab = Vocabulary(256)
    
    # States and actions for a simple game
    states = ["SEE_FOOD", "SEE_DANGER", "HUNGRY", "SCARED", "SAFE", "FED"]
    actions = ["APPROACH", "FLEE", "HIDE", "EAT", "WAIT"]
    outcomes = ["REWARD", "PUNISH", "NEUTRAL"]
    
    for item in states + actions + outcomes:
        vocab.create_pointer(item)
    
    # Create modules
    perception = State("perception", 256)
    action = State("action", 256)
    outcome = State("outcome", 256)
    
    # Create production system with learning
    ps = ProductionSystem()
    
    print("\n1. Initial Rules with Strengths:")
    
    # Create rules with initial strengths
    rules_data = [
        ("see_food_approach", "SEE_FOOD", "APPROACH", 0.5),
        ("see_food_wait", "SEE_FOOD", "WAIT", 0.5),
        ("see_danger_flee", "SEE_DANGER", "FLEE", 0.8),
        ("see_danger_hide", "SEE_DANGER", "HIDE", 0.2)
    ]
    
    # Track rule strengths for learning
    rule_strengths = {}
    
    for name, cond, act, strength in rules_data:
        rule = Production(
            name=name,
            condition=Condition(
                lambda c=cond, s=strength: perception.get_semantic_pointer(vocab).similarity(vocab[c]) * s,
                f"perceive {cond}"
            ),
            effect=Effect(
                lambda a=act: setattr(action, 'state', vocab[a].vector),
                f"action {act}"
            )
        )
        ps.add_production(rule)
        rule_strengths[name] = strength
        print(f"   {name}: strength={strength:.2f}")
    
    ps.set_context({
        "perception": perception,
        "action": action,
        "outcome": outcome,
        "vocab": vocab
    })
    
    # Simulate learning trials
    print("\n2. Learning from Experience:")
    
    # Trial 1: See food, approach, get reward
    print("\n   Trial 1:")
    perception.state = vocab["SEE_FOOD"].vector
    
    # Check which rule wins
    utilities = ps.evaluate_all()
    winner = max(utilities, key=lambda x: x[1])[0]
    print(f"   Perception: SEE_FOOD")
    print(f"   Selected: {winner.name}")
    
    # Execute and get outcome
    winner.execute()
    outcome.state = vocab["REWARD"].vector
    
    # Update strength based on outcome
    if "approach" in winner.name:
        rule_strengths["see_food_approach"] += 0.1  # Strengthen
        rule_strengths["see_food_wait"] -= 0.05    # Weaken alternative
        print("   Outcome: REWARD - strengthening approach rule")
    
    # Trial 2: See food again
    print("\n   Trial 2:")
    perception.state = vocab["SEE_FOOD"].vector
    
    # Update conditions with new strengths
    for prod in ps.productions:
        if prod.name in rule_strengths:
            # Create new condition with updated strength
            if "see_food" in prod.name:
                s = rule_strengths[prod.name]
                prod.condition = Condition(
                    lambda s=s: perception.get_semantic_pointer(vocab).similarity(vocab["SEE_FOOD"]) * s,
                    f"perceive SEE_FOOD (strength={s:.2f})"
                )
    
    utilities = ps.evaluate_all()
    print("   Updated utilities:")
    for prod, util in utilities:
        if "see_food" in prod.name:
            print(f"   - {prod.name}: {util:.3f}")
    
    # Show learning effect
    winner = max(utilities, key=lambda x: x[1])[0]
    print(f"   Selected: {winner.name} (learned preference)")
    
    # Demonstrate adaptation over multiple trials
    print("\n3. Adaptation Over Time:")
    
    learning_history = {"approach": [], "wait": []}
    
    for trial in range(10):
        perception.state = vocab["SEE_FOOD"].vector
        
        # Occasionally punish approach to show adaptation
        if trial == 5:
            outcome.state = vocab["PUNISH"].vector
            rule_strengths["see_food_approach"] -= 0.3
            rule_strengths["see_food_wait"] += 0.2
            print(f"   Trial {trial + 1}: PUNISHED for approach")
        
        # Update rule conditions
        for prod in ps.productions:
            if "see_food_approach" in prod.name:
                s = rule_strengths["see_food_approach"]
                prod.condition = Condition(
                    lambda s=s: perception.get_semantic_pointer(vocab).similarity(vocab["SEE_FOOD"]) * s,
                    f"strength={s:.2f}"
                )
            elif "see_food_wait" in prod.name:
                s = rule_strengths["see_food_wait"]
                prod.condition = Condition(
                    lambda s=s: perception.get_semantic_pointer(vocab).similarity(vocab["SEE_FOOD"]) * s,
                    f"strength={s:.2f}"
                )
        
        # Record strengths
        learning_history["approach"].append(rule_strengths["see_food_approach"])
        learning_history["wait"].append(rule_strengths["see_food_wait"])
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    trials = range(1, 11)
    plt.plot(trials, learning_history["approach"], 'b-', label="Approach", marker='o')
    plt.plot(trials, learning_history["wait"], 'r-', label="Wait", marker='s')
    plt.axvline(x=6, color='k', linestyle='--', alpha=0.5, label="Punishment")
    plt.xlabel("Trial")
    plt.ylabel("Rule Strength")
    plt.title("Production Rule Learning")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return ps, vocab, rule_strengths


def demonstrate_production_parsing():
    """Demonstrate parsing production rules from text."""
    print("\n\n=== Production Rule Parsing ===")
    
    # Create vocabulary
    vocab = Vocabulary(128)
    
    # Add vocabulary items
    items = ["STUDENT", "STUDYING", "EXAM_SOON", "TIRED", "ALERT",
             "STUDY", "REST", "COFFEE", "REVIEW", "PRACTICE"]
    for item in items:
        vocab.create_pointer(item)
    
    # Create modules
    state = State("state", 128)
    action = State("action", 128)
    
    print("\n1. Rule Specification Format:")
    
    # Define rules as text
    rule_text = """
    # Study planning production rules
    
    # Rule 1: If student and exam soon, then study
    IF state.STUDENT > 0.5 AND state.EXAM_SOON > 0.5 THEN action = STUDY
    
    # Rule 2: If studying and tired, then coffee
    IF state.STUDYING > 0.5 AND state.TIRED > 0.5 THEN action = COFFEE
    
    # Rule 3: If alert and exam soon, then practice
    IF state.ALERT > 0.5 AND state.EXAM_SOON > 0.5 THEN action = PRACTICE [priority=2]
    
    # Rule 4: If very tired, then rest
    IF state.TIRED > 0.8 THEN action = REST [priority=3]
    """
    
    print("   Rule specifications:")
    for line in rule_text.strip().split('\n'):
        if line.strip() and not line.strip().startswith('#'):
            print(f"   {line.strip()}")
    
    # Parse rules
    print("\n2. Parsing Rules:")
    
    productions = parse_production_rules(rule_text, {"state": state, "action": action}, vocab)
    
    print(f"   Parsed {len(productions)} production rules")
    
    # Create production system
    ps = ProductionSystem()
    for prod in productions:
        ps.add_production(prod)
        print(f"   - {prod.name}: {prod.condition.description}")
    
    ps.set_context({
        "state": state,
        "action": action,
        "vocab": vocab
    })
    
    # Test the parsed rules
    print("\n3. Testing Parsed Rules:")
    
    # Test scenario 1: Student with exam soon
    print("\n   Scenario 1: Student with upcoming exam")
    state.state = (vocab["STUDENT"] + vocab["EXAM_SOON"]).normalize().vector
    
    fired = ps.step()
    if fired:
        print(f"   Fired: {fired.name}")
        result = action.get_semantic_pointer(vocab)
        matches = vocab.cleanup(result.vector, top_n=1)
        if matches:
            print(f"   Action: {matches[0][0]}")
    
    # Test scenario 2: Very tired student
    print("\n   Scenario 2: Very tired")
    state.state = vocab["TIRED"].vector
    
    fired = ps.step()
    if fired:
        print(f"   Fired: {fired.name}")
        result = action.get_semantic_pointer(vocab)
        matches = vocab.cleanup(result.vector, top_n=1)
        if matches:
            print(f"   Action: {matches[0][0]} (high priority)")
    
    return ps, vocab


def demonstrate_production_analysis():
    """Analyze production system behavior."""
    print("\n\n=== Production System Analysis ===")
    
    # Create a more complex production system
    ps, observation, classification, vocab = create_animal_classification_system()
    
    # Add more rules for complexity
    extra_rules = [
        Production(
            name="identify_bird_that_swims",
            condition=Condition(
                lambda: classification.get_semantic_pointer(vocab).similarity(vocab["BIRD"]) > 0.5 and
                       observation.get_semantic_pointer(vocab).similarity(vocab["SWIMS"]) > 0.5,
                "is bird AND swims"
            ),
            effect=Effect(
                lambda: setattr(classification, 'state', vocab["PENGUIN"].vector),
                "identify as PENGUIN"
            ),
            priority=2.0
        ),
        Production(
            name="identify_flying_bird",
            condition=Condition(
                lambda: classification.get_semantic_pointer(vocab).similarity(vocab["BIRD"]) > 0.5 and
                       observation.get_semantic_pointer(vocab).similarity(vocab["FLIES"]) > 0.5,
                "is bird AND flies"
            ),
            effect=Effect(
                lambda: setattr(classification, 'state', vocab["EAGLE"].vector),
                "identify as EAGLE"
            ),
            priority=2.0
        )
    ]
    
    for rule in extra_rules:
        ps.add_production(rule)
    
    # Analyze the system
    print("\n1. Production System Structure:")
    analysis = analyze_production_system(ps)
    
    print(f"   Total productions: {analysis['total_productions']}")
    print(f"   Average priority: {analysis['avg_priority']:.2f}")
    print(f"   Max chain length: {analysis['max_chain_length']}")
    
    print("\n   Production rules by priority:")
    for prod in sorted(ps.productions, key=lambda p: p.priority, reverse=True):
        print(f"   - {prod.name}: priority={prod.priority}")
    
    # Trace execution path
    print("\n2. Execution Trace:")
    
    # Set up for bird that swims
    observation.state = (vocab["HAS_FEATHERS"] + vocab["HAS_WINGS"] + vocab["SWIMS"]).normalize().vector
    classification.state = np.zeros(256)
    
    print("   Input: HAS_FEATHERS, HAS_WINGS, SWIMS")
    
    executed = []
    for cycle in range(5):
        # Evaluate all rules
        utilities = ps.evaluate_all()
        
        if any(util > 0 for _, util in utilities):
            print(f"\n   Cycle {cycle + 1}:")
            print("   Matching rules:")
            for prod, util in utilities:
                if util > 0:
                    print(f"   - {prod.name}: {util:.3f}")
            
            # Execute highest utility
            fired = ps.step()
            if fired:
                executed.append(fired)
                print(f"   Executed: {fired.name}")
                
                # Check result
                result = classification.get_semantic_pointer(vocab)
                matches = vocab.cleanup(result.vector, top_n=1)
                if matches:
                    print(f"   Classification: {matches[0][0]}")
        else:
            break
    
    # Visualize execution flow
    print("\n3. Visualizing Production Flow:")
    
    if executed:
        fig, ax = visualize_production_flow(ps, executed_productions=executed)
        plt.title("Production System Execution Flow")
        plt.show()
    
    # Analyze potential cycles
    print("\n4. Checking for Cycles:")
    
    if "has_cycle" in analysis and analysis["has_cycle"]:
        print("   WARNING: Potential cycle detected in production rules")
        if "cycle_rules" in analysis:
            print("   Rules involved:")
            for rule in analysis["cycle_rules"]:
                print(f"   - {rule}")
    else:
        print("   No cycles detected")
    
    return ps, vocab, analysis


def main():
    """Run all production system demonstrations."""
    print("=" * 60)
    print("Production System Demonstration")
    print("=" * 60)
    
    # Run demonstrations
    ps1, vocab1 = demonstrate_forward_chaining()
    ps2, vocab2 = demonstrate_conflict_resolution()
    ps3, vocab3, strengths = demonstrate_learning_productions()
    ps4, vocab4 = demonstrate_production_parsing()
    ps5, vocab5, analysis = demonstrate_production_analysis()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nKey Concepts Demonstrated:")
    print("- Production rules: IF-THEN statements with pattern matching")
    print("- Forward chaining: Data-driven inference")
    print("- Conflict resolution: Priority and specificity-based selection")
    print("- Learning: Strength adaptation based on outcomes")
    print("- Rule parsing: Text-based rule specification")
    print("- System analysis: Structure and behavior analysis")
    print("- Integration: Seamless work with SPA modules")
    print("=" * 60)


if __name__ == "__main__":
    main()