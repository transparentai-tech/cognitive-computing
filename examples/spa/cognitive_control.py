#!/usr/bin/env python3
"""
Cognitive Control Demonstration using SPA.

This example demonstrates executive control functions:
- Working memory manipulation
- Attention control and focus
- Task switching and cognitive flexibility
- Inhibition and selection
- Goal-directed behavior
- Conflict monitoring and resolution

These mechanisms are inspired by frontal cortex functions and
demonstrate how SPA can model high-level cognitive control.
"""

import numpy as np
import matplotlib.pyplot as plt
from cognitive_computing.spa import (
    create_spa, SPAConfig, Vocabulary, SemanticPointer,
    State, Buffer, Gate,
    CognitiveControl, Routing, Gating, Sequencing,
    ActionSet, Action, BasalGanglia,
    SPAModel, ModelBuilder
)
from cognitive_computing.spa.visualizations import (
    plot_module_activity, plot_action_selection
)


def demonstrate_working_memory():
    """Demonstrate working memory operations with cognitive control."""
    print("\n=== Working Memory Control ===")
    
    # Create cognitive control system
    config = SPAConfig(dimension=256)
    control = CognitiveControl(256, config)
    
    # Create vocabulary
    vocab = Vocabulary(256)
    items = ["APPLE", "BANANA", "ORANGE", "GRAPE", "PEACH"]
    for item in items:
        vocab.create_pointer(item)
    
    # Add attention markers
    vocab.create_pointer("FOCUS")
    vocab.create_pointer("IGNORE")
    vocab.create_pointer("MAINTAIN")
    vocab.create_pointer("UPDATE")
    
    # Set control's vocabulary
    control.vocab = vocab
    
    print("\n1. Working Memory Capacity:")
    print(f"   Max items: {config.capacity if hasattr(config, 'capacity') else 'unlimited'}")
    
    # Add working memory buffers
    print("\n2. Loading Working Memory:")
    for i, item in enumerate(items[:3]):
        buffer_name = f"slot_{i}"
        control.add_working_memory(buffer_name)
        control.working_memory[buffer_name].state = vocab[item].vector
        print(f"   Slot {i}: {item}")
    
    # Demonstrate maintenance with attention
    print("\n3. Selective Attention:")
    # Focus on BANANA (slot 1)
    control.set_attention(vocab["BANANA"].vector)
    print("   Focusing on BANANA")
    
    # Check attention similarity to items
    for i, item in enumerate(items[:3]):
        sim = vocab[item].similarity(SemanticPointer(control.attention, vocabulary=vocab))
        if sim > 0.8:
            print(f"   Attended item in slot {i}: {item} (similarity: {sim:.3f})")
    
    # Demonstrate updating with cognitive control
    print("\n4. Controlled Updating:")
    print("   Current slot 2: ORANGE")
    
    # Create an update gate
    control.add_working_memory("update_gate")
    update_gate = control.working_memory["update_gate"]
    
    # Gate starts closed
    update_gate.gate_value = 0.0
    print(f"   Update gate: {update_gate.gate_value:.2f}")
    
    # Open gate to update
    update_gate.gate_value = 1.0
    print("   Gate opened - updating slot 2")
    control.working_memory["slot_2"].state = vocab["GRAPE"].vector
    
    # Check update
    wm_content = control.working_memory["slot_2"].state
    matches = vocab.cleanup(wm_content, top_n=1)
    if matches:
        print(f"   Updated slot 2: {matches[0][0]}")
    
    # Close gate to maintain
    update_gate.gate_value = 0.0
    print("   Gate closed - maintaining current contents")
    
    return control, vocab


def demonstrate_task_switching():
    """Demonstrate task switching and cognitive flexibility."""
    print("\n\n=== Task Switching ===")
    
    # Create control system
    config = SPAConfig(dimension=128)
    control = CognitiveControl(128, config)
    
    # Create vocabulary for tasks and stimuli
    vocab = Vocabulary(128)
    
    # Task representations
    tasks = ["COLOR_TASK", "SHAPE_TASK", "SIZE_TASK"]
    for task in tasks:
        vocab.create_pointer(task)
    
    # Stimulus features
    colors = ["RED", "BLUE", "GREEN"]
    shapes = ["CIRCLE", "SQUARE", "TRIANGLE"]
    sizes = ["SMALL", "MEDIUM", "LARGE"]
    
    for item in colors + shapes + sizes:
        vocab.create_pointer(item)
    
    # Response mappings
    responses = ["LEFT", "RIGHT", "CENTER"]
    for resp in responses:
        vocab.create_pointer(resp)
    
    print("\n1. Task Stack Management:")
    
    # Set control's vocabulary first
    control.vocab = vocab
    
    # Push tasks onto stack
    control.push_task("COLOR_TASK")
    control.push_task("SHAPE_TASK")
    
    print("   Task stack:")
    for i, task in enumerate(control.task_stack):
        print(f"   {i}: {task}")
    
    # Switch tasks
    print("\n2. Task Switching:")
    if control.current_task is not None:
        print(f"   Current task: {control.current_task}")
    
    # Pop to switch
    control.pop_task()
    if control.current_task is not None:
        print(f"   After switch: {control.current_task}")
    
    # Demonstrate task-based routing
    print("\n3. Task-Based Response Routing:")
    
    # Create routing rules
    routing = Routing(128, config)
    
    # Color task routes colors to responses
    color_gate = routing.add_route("color_input", "response")
    
    # Shape task routes shapes to responses  
    shape_gate = routing.add_route("shape_input", "response")
    
    # Test routing with different tasks
    control.set_goal(vocab["COLOR_TASK"].vector)
    
    # Simulate color input
    color_signal = vocab["RED"].vector
    # Open color gate when color task is active
    if control.task_state @ vocab["COLOR_TASK"].vector > 0.8:
        color_gate.set_gate(1.0)
        print("   COLOR_TASK active: Routing RED -> response")
    
    # Switch task
    control.set_goal(vocab["SHAPE_TASK"].vector)
    
    # Now shape input should route
    shape_signal = vocab["CIRCLE"].vector
    # Open shape gate when shape task is active
    if control.task_state @ vocab["SHAPE_TASK"].vector > 0.8:
        shape_gate.set_gate(1.0)
        color_gate.set_gate(0.0)  # Close color gate
        print("   SHAPE_TASK active: Routing CIRCLE -> response")
    
    return control, vocab, routing


def demonstrate_inhibition_control():
    """Demonstrate inhibition and conflict resolution."""
    print("\n\n=== Inhibition and Conflict Control ===")
    
    # Create modules
    config = SPAConfig(dimension=128)
    
    # Stimulus input
    stimulus = State("stimulus", 128)
    
    # Response options
    response = State("response", 128)
    
    # Inhibitory control gate
    inhibit_gate = Gate("inhibit", 128)
    
    # Create vocabulary
    vocab = Vocabulary(128)
    
    # Stroop-like task items
    words = ["RED", "BLUE", "GREEN"]
    colors = ["COLOR_RED", "COLOR_BLUE", "COLOR_GREEN"]
    
    for item in words + colors:
        vocab.create_pointer(item)
    
    print("\n1. Stroop-like Conflict:")
    
    # Congruent case: word RED in red color
    congruent = vocab["RED"] + vocab["COLOR_RED"]
    congruent = congruent.normalize()
    
    # Incongruent case: word BLUE in red color
    incongruent = vocab["BLUE"] + vocab["COLOR_RED"]
    incongruent = incongruent.normalize()
    
    print("   Congruent: RED + COLOR_RED")
    print("   Incongruent: BLUE + COLOR_RED")
    
    # Create action set for responses
    actions = ActionSet()
    
    # Word reading (automatic, strong)
    word_actions = {
        "RED": Action(
            condition=lambda: stimulus.get_semantic_pointer(vocab).dot(vocab["RED"]),
            effect=lambda: setattr(response, 'state', vocab["RED"].vector),
            name="read_red"
        ),
        "BLUE": Action(
            condition=lambda: stimulus.get_semantic_pointer(vocab).dot(vocab["BLUE"]),
            effect=lambda: setattr(response, 'state', vocab["BLUE"].vector),
            name="read_blue"
        )
    }
    
    # Color naming (controlled, weaker)
    color_actions = {
        "COLOR_RED": Action(
            condition=lambda: stimulus.get_semantic_pointer(vocab).dot(vocab["COLOR_RED"]) * inhibit_gate.gate_value,
            effect=lambda: setattr(response, 'state', vocab["RED"].vector),
            name="name_red"
        ),
        "COLOR_BLUE": Action(
            condition=lambda: stimulus.get_semantic_pointer(vocab).dot(vocab["COLOR_BLUE"]) * inhibit_gate.gate_value,
            effect=lambda: setattr(response, 'state', vocab["BLUE"].vector),
            name="name_blue"
        )
    }
    
    for action in word_actions.values():
        actions.add_action(action)
    for action in color_actions.values():
        actions.add_action(action)
    
    # Test without inhibition
    print("\n2. Without Inhibitory Control:")
    inhibit_gate.set_gate(0.3)  # Weak control
    stimulus.state = incongruent.vector
    
    utilities = actions.evaluate_all()
    print("   Action utilities (BLUE + COLOR_RED):")
    for i, action in enumerate(actions.actions):
        print(f"   - {action.name}: {utilities[i]:.3f}")
    
    # Test with inhibition
    print("\n3. With Strong Inhibitory Control:")
    inhibit_gate.set_gate(2.0)  # Strong control
    
    utilities = actions.evaluate_all()
    print("   Action utilities (BLUE + COLOR_RED):")
    for i, action in enumerate(actions.actions):
        print(f"   - {action.name}: {utilities[i]:.3f}")
    
    selected = actions.select_action()
    if selected:
        print(f"   Selected: {selected.name}")
    
    return stimulus, response, actions, vocab


def demonstrate_goal_directed_behavior():
    """Demonstrate goal-directed behavior with cognitive control."""
    print("\n\n=== Goal-Directed Behavior ===")
    
    # Create SPA model with cognitive control
    config = SPAConfig(dimension=256)
    model = SPAModel("goal_directed_model", config)
    
    # Add vocabulary
    vocab = Vocabulary(256)
    
    # Goals
    goals = ["FIND_FOOD", "FIND_WATER", "FIND_SHELTER", "REST"]
    for goal in goals:
        vocab.create_pointer(goal)
    
    # States and actions
    states = ["HUNGRY", "THIRSTY", "TIRED", "COLD"]
    actions = ["SEARCH", "APPROACH", "CONSUME", "ENTER", "SLEEP"]
    
    for item in states + actions:
        vocab.create_pointer(item)
    
    model.add_vocabulary("main", 256)
    
    # Add modules
    model.add_module("goal", "state", dimensions=256)
    model.add_module("state", "state", dimensions=256)
    model.add_module("action", "state", dimensions=256)
    model.add_module("working_memory", "memory", dimensions=256)
    
    # Create cognitive control
    config = SPAConfig(dimension=256)
    control = CognitiveControl(256, config)
    
    print("\n1. Goal Hierarchy:")
    
    # Set top-level goal
    control.set_goal(vocab["FIND_FOOD"].vector)
    if np.any(control.goal_state):
        matches = vocab.cleanup(control.goal_state, top_n=1)
        print(f"   Top goal: {matches[0][0] if matches else 'Unknown'}")
    
    # Break down into subgoals
    subgoals = {
        "FIND_FOOD": ["SEARCH", "APPROACH", "CONSUME"],
        "FIND_WATER": ["SEARCH", "APPROACH", "CONSUME"],
        "FIND_SHELTER": ["SEARCH", "APPROACH", "ENTER"],
        "REST": ["FIND_SHELTER", "SLEEP"]
    }
    
    # Demonstrate goal-driven action selection
    print("\n2. Goal-Driven Action Selection:")
    
    # Create goal-conditional actions
    goal_actions = ActionSet()
    
    # If goal is FIND_FOOD and HUNGRY, then SEARCH
    def search_food_condition():
        goal_sim = control.goal_state @ vocab["FIND_FOOD"].vector / np.linalg.norm(control.goal_state)
        return goal_sim * 0.8 + np.random.random() * 0.2
        
    search_food = Action(
        condition=search_food_condition,
        effect=lambda: print("   -> Action: SEARCH for food"),
        name="search_food"
    )
    
    # If goal is FIND_FOOD and near food, then APPROACH
    def approach_food_condition():
        goal_sim = control.goal_state @ vocab["FIND_FOOD"].vector / np.linalg.norm(control.goal_state)
        return goal_sim * 0.6 + np.random.random() * 0.2
        
    approach_food = Action(
        condition=approach_food_condition,
        effect=lambda: print("   -> Action: APPROACH food"),
        name="approach_food"
    )
    
    goal_actions.add_action(search_food)
    goal_actions.add_action(approach_food)
    
    # Evaluate with goal active
    print("   With FIND_FOOD goal:")
    utilities = goal_actions.evaluate_all()
    for i, action in enumerate(goal_actions.actions):
        print(f"   - {action.name}: {utilities[i]:.3f}")
    
    selected = goal_actions.select_action()
    if selected:
        selected.execute()
    
    # Change goal
    print("\n3. Goal Switching:")
    control.set_goal(vocab["FIND_WATER"].vector)
    if np.any(control.goal_state):
        matches = vocab.cleanup(control.goal_state, top_n=1)
        print(f"   New goal: {matches[0][0] if matches else 'Unknown'}")
    
    # Goals affect action utilities
    utilities = goal_actions.evaluate_all()
    print("   Action utilities with FIND_WATER goal:")
    for i, action in enumerate(goal_actions.actions):
        print(f"   - {action.name}: {utilities[i]:.3f} (reduced)")
    
    return model, control, vocab


def demonstrate_sequencing_control():
    """Demonstrate sequential behavior control."""
    print("\n\n=== Sequential Behavior Control ===")
    
    # Create vocabulary first
    vocab = Vocabulary(128)
    config = SPAConfig(dimension=128)
    
    # Create sequencing controller
    sequencing = Sequencing(128, config, vocab)
    
    # Define steps for making coffee
    coffee_steps = [
        "GET_CUP",
        "ADD_COFFEE",
        "ADD_WATER", 
        "STIR",
        "ADD_SUGAR",
        "DRINK"
    ]
    
    for step in coffee_steps:
        vocab.create_pointer(step)
    
    # Define sequence
    sequencing.define_sequence("make_coffee", coffee_steps)
    
    print("\n1. Sequence Definition:")
    print("   Make coffee sequence:")
    for i, step in enumerate(coffee_steps):
        print(f"   {i+1}. {step}")
    
    # Execute sequence
    print("\n2. Sequence Execution:")
    sequencing.start_sequence("make_coffee")
    
    step_count = 0
    while step_count < len(coffee_steps):
        current = sequencing.next_step()
        if current is not None:
            print(f"   Step {step_count + 1}: {current}")
        else:
            break
        step_count += 1
    
    print("   Sequence complete!")
    
    # Demonstrate interruption and resumption
    print("\n3. Interruption Handling:")
    
    # Start sequence again
    sequencing.start_sequence("make_coffee")
    
    # Execute first few steps
    for i in range(3):
        current = sequencing.next_step()
        if current is not None:
            print(f"   Executed: {current}")
    
    # Interrupt
    print("   [INTERRUPTED]")
    sequencing.pause_sequence()
    position = sequencing.sequence_index
    print(f"   Saved position: {position}")
    
    # Resume
    print("   [RESUMING]")
    sequencing.resume_sequence()
    current = sequencing.next_step()
    if current is not None:
        print(f"   Resuming at: {current}")
    
    # Loop demonstration
    print("\n4. Sequence Looping:")
    
    # Simple counting sequence
    count_steps = ["ONE", "TWO", "THREE"]
    for step in count_steps:
        vocab.create_pointer(step)
    
    sequencing.define_sequence("count", count_steps)
    sequencing.max_loops = 3  # Enable looping
    
    sequencing.start_sequence("count")
    
    print("   Looping count sequence:")
    for i in range(9):  # Show 3 loops
        current = sequencing.next_step()
        if current is not None:
            print(f"   {i+1}: {current}", end="")
            if (i + 1) % 3 == 0:
                print(" [loop]")
            else:
                print()
        else:
            break
    
    return sequencing, vocab


def demonstrate_conflict_monitoring():
    """Demonstrate conflict monitoring and resolution."""
    print("\n\n=== Conflict Monitoring ===")
    
    # Create cognitive control with conflict monitoring
    config = SPAConfig(dimension=128)
    control = CognitiveControl(128, config)
    
    # Create vocabulary
    vocab = Vocabulary(128)
    
    # Response options that can conflict
    responses = ["GO", "STOP", "WAIT"]
    for resp in responses:
        vocab.create_pointer(resp)
    
    # Signals that trigger responses
    signals = ["GREEN_LIGHT", "RED_LIGHT", "YELLOW_LIGHT", "PEDESTRIAN"]
    for sig in signals:
        vocab.create_pointer(sig)
    
    print("\n1. Conflict Detection:")
    
    # Create conflicting action utilities
    # Simulate two strong competing responses
    response1_utility = 0.8  # GO
    response2_utility = 0.75  # STOP
    response3_utility = 0.2  # WAIT
    
    utilities = np.array([response1_utility, response2_utility, response3_utility])
    
    # Monitor conflict
    control.update_conflict(utilities)
    conflict = control.conflict_level
    print(f"   Response utilities: GO={response1_utility}, STOP={response2_utility}, WAIT={response3_utility}")
    print(f"   Conflict level: {conflict:.3f}")
    
    # High conflict case
    utilities_high = np.array([0.8, 0.79, 0.78])
    control.update_conflict(utilities_high)
    conflict_high = control.conflict_level
    print(f"\n   High conflict utilities: {utilities_high}")
    print(f"   Conflict level: {conflict_high:.3f} (high)")
    
    # Low conflict case  
    utilities_low = np.array([0.9, 0.2, 0.1])
    control.update_conflict(utilities_low)
    conflict_low = control.conflict_level
    print(f"\n   Low conflict utilities: {utilities_low}")
    print(f"   Conflict level: {conflict_low:.3f} (low)")
    
    print("\n2. Conflict Resolution Strategies:")
    
    # Strategy 1: Increase control based on conflict
    control_strength = 0.5 + conflict_high * 0.5
    print(f"   Adaptive control strength: {control_strength:.3f}")
    
    # Strategy 2: Gather more evidence when conflict is high
    if conflict_high > 0.7:
        print("   High conflict detected - gathering more evidence...")
        # Simulate evidence accumulation
        utilities_updated = utilities_high + np.random.randn(3) * 0.1
        utilities_updated = np.clip(utilities_updated, 0, 1)
        print(f"   Updated utilities: {utilities_updated}")
        
        control.update_conflict(utilities_updated)
        conflict_updated = control.conflict_level
        print(f"   Updated conflict: {conflict_updated:.3f}")
    
    # Strategy 3: Invoke supervisory control
    if conflict_high > 0.8:
        print("   Very high conflict - invoking supervisory control")
        # Could switch to different decision strategy
        # or request additional input
    
    # Demonstrate error monitoring
    print("\n3. Error Monitoring:")
    
    # Simulate action and outcome
    selected_action = "GO"
    expected_outcome = vocab["GO"].vector
    actual_outcome = vocab["STOP"].vector  # Error!
    
    control.update_error(expected_outcome, actual_outcome)
    error_signal = control.error_signal
    print(f"   Selected: {selected_action}")
    print(f"   Expected: GO, Actual: STOP")
    print(f"   Error signal: {error_signal:.3f}")
    
    # Error leads to control adjustment
    if error_signal > 0.5:
        print("   Error detected - increasing cognitive control")
        # Could implement control adjustment logic here
        print(f"   Error signal strength: {error_signal:.3f}")
    
    return control, vocab


def visualize_cognitive_control():
    """Visualize cognitive control dynamics."""
    print("\n\n=== Visualizing Cognitive Control ===")
    
    # Create a simple model with control
    config = SPAConfig(dimension=128)
    control = CognitiveControl(128, config)
    
    # Track control dynamics over time
    time_steps = 100
    control_history = []
    conflict_history = []
    
    # Simulate varying conflict levels
    for t in range(time_steps):
        # Generate utilities with varying conflict
        phase = t / time_steps * 2 * np.pi
        
        # Create conflicting utilities that vary over time
        util1 = 0.6 + 0.3 * np.sin(phase)
        util2 = 0.6 + 0.3 * np.sin(phase + np.pi/3)
        util3 = 0.3
        
        utilities = np.array([util1, util2, util3])
        
        # Monitor conflict
        control.update_conflict(utilities)
        conflict = control.conflict_level
        conflict_history.append(conflict)
        
        # Adjust control based on conflict
        control_strength = 0.5 + conflict * 0.5
        control_history.append(control_strength)
    
    # Plot dynamics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    time = np.arange(time_steps)
    
    # Conflict level
    ax1.plot(time, conflict_history, 'r-', linewidth=2)
    ax1.set_ylabel('Conflict Level')
    ax1.set_title('Cognitive Control Dynamics')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Control strength
    ax2.plot(time, control_history, 'b-', linewidth=2)
    ax2.set_ylabel('Control Strength')
    ax2.set_xlabel('Time Steps')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print("   Visualization shows:")
    print("   - Red: Conflict level between response options")
    print("   - Blue: Adaptive control strength")
    print("   - Control increases when conflict is high")


def main():
    """Run all cognitive control demonstrations."""
    print("=" * 60)
    print("Cognitive Control Demonstration")
    print("=" * 60)
    
    # Run demonstrations
    control1, vocab1 = demonstrate_working_memory()
    control2, vocab2, routing = demonstrate_task_switching()
    stimulus, response, actions, vocab3 = demonstrate_inhibition_control()
    model, control3, vocab4 = demonstrate_goal_directed_behavior()
    sequencing, vocab5 = demonstrate_sequencing_control()
    control4, vocab6 = demonstrate_conflict_monitoring()
    visualize_cognitive_control()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nKey Concepts Demonstrated:")
    print("- Working memory: Controlled maintenance and updating")
    print("- Task switching: Flexible cognitive control")
    print("- Inhibition: Suppressing prepotent responses")
    print("- Goal-directed behavior: Top-down control")
    print("- Sequencing: Structured behavioral programs")
    print("- Conflict monitoring: Detecting and resolving competition")
    print("- Adaptive control: Adjusting based on task demands")
    print("=" * 60)


if __name__ == "__main__":
    main()