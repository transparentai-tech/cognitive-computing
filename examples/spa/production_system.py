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
    SPAConfig, Vocabulary, SemanticPointer,
    State, Memory, Buffer,
    Production, ProductionSystem, 
    Condition, MatchCondition, CompareCondition, CompoundCondition,
    Effect, SetEffect, BindEffect, CompoundEffect,
    ConditionalModule,
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
    fur_condition = MatchCondition("observation", "HAS_FUR", threshold=0.5)
    milk_condition = MatchCondition("observation", "GIVES_MILK", threshold=0.5)
    mammal_condition = CompoundCondition([fur_condition, milk_condition], "and")
    
    rule1 = Production(
        name="classify_mammal",
        condition=mammal_condition,
        effect=SetEffect("classification", "MAMMAL"),
        priority=1.0
    )
    
    # Rule 2: If has feathers and has wings, then bird
    feathers_condition = MatchCondition("observation", "HAS_FEATHERS", threshold=0.5)
    wings_condition = MatchCondition("observation", "HAS_WINGS", threshold=0.5)
    bird_condition = CompoundCondition([feathers_condition, wings_condition], "and")
    
    rule2 = Production(
        name="classify_bird",
        condition=bird_condition,
        effect=SetEffect("classification", "BIRD"),
        priority=1.0
    )
    
    # Rule 3: If has scales and cold-blooded, then reptile
    scales_condition = MatchCondition("observation", "HAS_SCALES", threshold=0.5)
    cold_condition = MatchCondition("observation", "COLD_BLOODED", threshold=0.5)
    reptile_condition = CompoundCondition([scales_condition, cold_condition], "and")
    
    rule3 = Production(
        name="classify_reptile",
        condition=reptile_condition,
        effect=SetEffect("classification", "REPTILE"),
        priority=1.0
    )
    
    # More specific rules (higher priority)
    
    # Rule 4: If mammal and walks, then dog
    is_mammal = MatchCondition("classification", "MAMMAL", threshold=0.5)
    walks = MatchCondition("observation", "WALKS", threshold=0.5)
    dog_condition = CompoundCondition([is_mammal, walks], "and")
    
    rule4 = Production(
        name="identify_dog",
        condition=dog_condition,
        effect=SetEffect("classification", "DOG"),
        priority=2.0  # Higher priority for specific identification
    )
    
    # Add rules to system
    ps.add_production(rule1)
    ps.add_production(rule2)
    ps.add_production(rule3)
    ps.add_production(rule4)
    
    print(f"   Added {len(ps.productions)} classification rules")
    
    # Set context
    ps.set_context(
        modules={
            "observation": observation,
            "classification": classification,
            "memory": memory
        },
        vocab=vocab
    )
    
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
    selected = ps.select_production()
    if selected:
        print(f"   Fired: {selected.name}")
        selected.fire(ps._context)
        
        # Check classification
        result = SemanticPointer(classification.state, vocabulary=vocab)
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
    selected = ps.select_production()
    if selected:
        print(f"   Fired: {selected.name}")
        selected.fire(ps._context)
        
        result = SemanticPointer(classification.state, vocabulary=vocab)
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
        selected = ps.select_production()
        if selected:
            print(f"   Cycle {cycle + 1}: Fired {selected.name}")
            selected.fire(ps._context)
            
            result = SemanticPointer(classification.state, vocabulary=vocab)
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
        condition=MatchCondition("input", "STIMULUS_A", threshold=0.5),
        effect=SetEffect("output", "RESPONSE_1"),
        priority=1.0,
        # specificity=1  # General rule
    )
    
    # Rule 2: A + Urgent -> Response 2
    a_cond = MatchCondition("input", "STIMULUS_A", threshold=0.5)
    urgent_cond = MatchCondition("input", "URGENT", threshold=0.5)
    a_urgent_cond = CompoundCondition([a_cond, urgent_cond], "and")
    
    rule2 = Production(
        name="a_urgent_to_2",
        condition=a_urgent_cond,
        effect=SetEffect("output", "RESPONSE_2"),
        priority=1.0,
        # specificity=2  # More specific rule
    )
    
    ps.add_production(rule1)
    ps.add_production(rule2)
    
    ps.set_context(
        modules={
            "input": input_state,
            "output": output_state
        },
        vocab=vocab
    )
    
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
    selected = ps.select_production()
    if selected:
        print(f"   Winner: {selected.name} (by priority/specificity)")
    
    # Test with A + Urgent
    print("\n3. Test with A + Urgent:")
    input_state.state = (vocab["STIMULUS_A"] + vocab["URGENT"]).normalize().vector
    
    utilities = ps.evaluate_all()
    print("   Matching rules:")
    for prod, util in utilities:
        print(f"   - {prod.name}: utility={util:.3f}")
    
    selected = ps.select_production()
    if selected:
        print(f"   Winner: {selected.name} (more specific)")
    
    # Demonstrate priority-based resolution
    print("\n4. Priority-Based Resolution:")
    
    # Add high-priority override rule
    # For the effect, we'll create a custom class
    class PrintEffect(Effect):
        def __init__(self, message):
            self.message = message
        def execute(self, context):
            print(self.message)
        def __repr__(self):
            return f"Print('{self.message}')"
    
    rule3 = Production(
        name="emergency_override",
        condition=MatchCondition("input", "URGENT", threshold=0.7),
        effect=PrintEffect("   EMERGENCY OVERRIDE ACTIVATED"),
        priority=10.0  # High priority
    )
    
    ps.add_production(rule3)
    
    # Test with urgent input
    print("   Added emergency rule with priority=10.0")
    input_state.state = vocab["URGENT"].vector
    
    selected = ps.select_production()
    if selected:
        print(f"   Winner: {selected.name} (highest priority)")
        selected.fire(ps._context)
    
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
    
    # Create a custom learning condition class
    class LearningCondition(Condition):
        def __init__(self, module_name, pattern, strength=1.0):
            self.module_name = module_name
            self.pattern = pattern
            self.strength = strength
            
        def evaluate(self, context):
            modules = context.get('modules', {})
            vocab = context.get('vocab')
            if self.module_name not in modules or vocab is None:
                return 0.0
            module = modules[self.module_name]
            pattern_vec = vocab[self.pattern].vector
            sp = SemanticPointer(module.state, vocabulary=vocab)
            similarity = sp.similarity(SemanticPointer(pattern_vec, vocabulary=vocab))
            return similarity * self.strength
            
        def __repr__(self):
            return f"LearningCondition({self.module_name}, {self.pattern}, strength={self.strength:.2f})"
    
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
            condition=LearningCondition("perception", cond, strength),
            effect=SetEffect("action", act)
        )
        ps.add_production(rule)
        rule_strengths[name] = strength
        print(f"   {name}: strength={strength:.2f}")
    
    ps.set_context(
        modules={
            "perception": perception,
            "action": action,
            "outcome": outcome
        },
        vocab=vocab
    )
    
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
    winner.fire(ps._context)
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
            # Update condition strength
            if "see_food" in prod.name:
                s = rule_strengths[prod.name]
                if isinstance(prod.condition, LearningCondition):
                    prod.condition.strength = s
    
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
            if prod.name in rule_strengths and isinstance(prod.condition, LearningCondition):
                prod.condition.strength = rule_strengths[prod.name]
        
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
    
    productions = parse_production_rules(rule_text, vocab)
    
    print(f"   Parsed {len(productions)} production rules")
    
    # Create production system
    ps = ProductionSystem()
    for prod in productions:
        ps.add_production(prod)
        print(f"   - {prod.name}: {prod.condition}")
    
    ps.set_context(
        modules={
            "state": state,
            "action": action
        },
        vocab=vocab
    )
    
    # Test the parsed rules
    print("\n3. Testing Parsed Rules:")
    
    # Test scenario 1: Student with exam soon
    print("\n   Scenario 1: Student with upcoming exam")
    state.state = (vocab["STUDENT"] + vocab["EXAM_SOON"]).normalize().vector
    
    selected = ps.select_production()
    if selected:
        print(f"   Fired: {selected.name}")
        selected.fire(ps._context)
        result = SemanticPointer(action.state, vocabulary=vocab)
        matches = vocab.cleanup(result.vector, top_n=1)
        if matches:
            print(f"   Action: {matches[0][0]}")
    
    # Test scenario 2: Very tired student
    print("\n   Scenario 2: Very tired")
    state.state = vocab["TIRED"].vector
    
    selected = ps.select_production()
    if selected:
        print(f"   Fired: {selected.name}")
        selected.fire(ps._context)
        result = SemanticPointer(action.state, vocabulary=vocab)
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
    # Create compound conditions for complex rules
    bird_cond1 = MatchCondition("classification", "BIRD", threshold=0.5)
    swims_cond = MatchCondition("observation", "SWIMS", threshold=0.5)
    bird_swims_cond = CompoundCondition([bird_cond1, swims_cond], "and")
    
    bird_cond2 = MatchCondition("classification", "BIRD", threshold=0.5)
    flies_cond = MatchCondition("observation", "FLIES", threshold=0.5)
    bird_flies_cond = CompoundCondition([bird_cond2, flies_cond], "and")
    
    extra_rules = [
        Production(
            name="identify_bird_that_swims",
            condition=bird_swims_cond,
            effect=SetEffect("classification", "PENGUIN"),
            priority=2.0
        ),
        Production(
            name="identify_flying_bird",
            condition=bird_flies_cond,
            effect=SetEffect("classification", "EAGLE"),
            priority=2.0
        )
    ]
    
    for rule in extra_rules:
        ps.add_production(rule)
    
    # Analyze the system
    print("\n1. Production System Structure:")
    test_context = {
        'modules': {
            'observation': observation,
            'classification': classification
        },
        'vocab': vocab
    }
    analysis = analyze_production_system(ps, test_context)
    
    print(f"   Total productions: {analysis['total_productions']}")
    if analysis['production_stats']:
        priorities = [stats['priority'] for stats in analysis['production_stats'].values()]
        avg_priority = sum(priorities) / len(priorities) if priorities else 0
        print(f"   Average priority: {avg_priority:.2f}")
    print(f"   Productions fired: {analysis['productions_fired']}")
    
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
            selected = ps.select_production()
            if selected:
                executed.append(selected)
                print(f"   Executed: {selected.name}")
                selected.fire(ps._context)
                
                # Check result
                result = SemanticPointer(classification.state, vocabulary=vocab)
                matches = vocab.cleanup(result.vector, top_n=1)
                if matches:
                    print(f"   Classification: {matches[0][0]}")
        else:
            break
    
    # Visualize execution flow
    print("\n3. Visualizing Production Flow:")
    
    if executed:
        executed_names = [prod.name for prod in executed]
        fig, ax = visualize_production_flow(ps, executed_productions=executed_names)
        plt.title("Production System Execution Flow")
        plt.show()
    
    # Analyze potential cycles
    print("\n4. Checking for Cycles:")
    
    if analysis.get("cycle_detected", False):
        print(f"   WARNING: Cycle detected with length {analysis.get('cycle_length', 0)}")
        print("   Firing sequence tail:")
        seq = analysis.get('firing_sequence', [])
        if seq:
            print(f"   - Last 10: {seq[-10:]}")
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