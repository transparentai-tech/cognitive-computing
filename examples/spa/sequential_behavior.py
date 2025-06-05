#!/usr/bin/env python3
"""
Sequential Behavior Demonstration using SPA.

This example demonstrates sequential task execution and planning:
- Defining and executing action sequences
- Implementing state machines with semantic pointers
- Conditional branching in sequences
- Interruption handling and recovery
- Hierarchical sequence organization
- Temporal pattern learning

Sequential behavior is fundamental to many cognitive tasks,
from motor control to language production to problem solving.
"""

import numpy as np
import matplotlib.pyplot as plt
from cognitive_computing.spa import (
    SPAConfig, Vocabulary, SemanticPointer,
    State, Buffer, Gate, Memory,
    Sequencing, CognitiveControl,
    ProductionSystem, Production, Condition, Effect,
    ActionSet, Action
)
from cognitive_computing.spa.visualizations import (
    plot_module_activity, animate_state_evolution
)


def create_motor_sequence():
    """Create a motor sequence for typing a word."""
    print("\n=== Motor Sequence: Typing ===")
    
    # Create vocabulary for motor actions
    vocab = Vocabulary(128)
    
    # Letter keys
    letters = list("HELLO")
    for letter in letters:
        vocab.create_pointer(f"KEY_{letter}")
    
    # Motor primitives
    primitives = ["PRESS", "RELEASE", "MOVE_TO", "SPACE", "ENTER"]
    for prim in primitives:
        vocab.create_pointer(prim)
    
    # States
    states = ["READY", "PRESSING", "MOVING", "COMPLETE"]
    for state in states:
        vocab.create_pointer(state)
    
    # Create sequencing controller
    sequencer = Sequencing()
    
    # Create motor state
    motor_state = State("motor", 128)
    key_buffer = Buffer("key_buffer", 128)
    
    print("\n1. Defining Typing Sequence:")
    
    # Create sequence for typing "HELLO"
    typing_sequence = []
    for letter in "HELLO":
        # Move to key
        typing_sequence.append(vocab["MOVE_TO"] * vocab[f"KEY_{letter}"])
        # Press key
        typing_sequence.append(vocab["PRESS"] * vocab[f"KEY_{letter}"])
        # Release key
        typing_sequence.append(vocab["RELEASE"] * vocab[f"KEY_{letter}"])
    
    # Add the sequence
    sequencer.define_sequence("type_hello", [s.vector for s in typing_sequence])
    
    print("   Sequence 'type_hello' defined with steps:")
    for i in range(0, len(typing_sequence), 3):
        letter = "HELLO"[i // 3]
        print(f"   - Move to {letter}, Press {letter}, Release {letter}")
    
    # Execute sequence
    print("\n2. Executing Typing Sequence:")
    
    sequencer.start_sequence("type_hello")
    
    typed_letters = []
    step_count = 0
    
    while not sequencer.is_sequence_complete("type_hello") and step_count < 20:
        current = sequencer.get_current_step("type_hello")
        
        if current is not None:
            motor_state.set_state(current)
            
            # Decode action
            pointer = motor_state.get_semantic_pointer(vocab)
            
            # Check for PRESS actions
            for letter in "HELLO":
                press_vec = vocab["PRESS"] * vocab[f"KEY_{letter}"]
                if pointer.similarity(press_vec) > 0.7:
                    typed_letters.append(letter)
                    print(f"   Step {step_count + 1}: Pressed {letter}")
                    break
        
        sequencer.advance_sequence("type_hello")
        step_count += 1
    
    print(f"   Typed: {''.join(typed_letters)}")
    
    return vocab, sequencer, motor_state


def create_cooking_sequence():
    """Create a hierarchical cooking sequence."""
    print("\n\n=== Hierarchical Sequence: Cooking ===")
    
    # Create vocabulary
    vocab = Vocabulary(256)
    
    # Ingredients
    ingredients = ["EGGS", "MILK", "FLOUR", "SUGAR", "BUTTER", "SALT"]
    for ing in ingredients:
        vocab.create_pointer(ing)
    
    # Actions
    actions = ["GET", "MIX", "POUR", "HEAT", "FLIP", "SERVE", "STIR", "WAIT"]
    for act in actions:
        vocab.create_pointer(act)
    
    # Tools
    tools = ["BOWL", "PAN", "SPATULA", "WHISK"]
    for tool in tools:
        vocab.create_pointer(tool)
    
    # States
    states = ["RAW", "MIXED", "COOKING", "READY", "BURNED"]
    for state in states:
        vocab.create_pointer(state)
    
    # Create modules
    action_state = State("action", 256)
    ingredient_state = State("ingredient", 256)
    tool_state = State("tool", 256)
    status_state = State("status", 256)
    
    # Create hierarchical sequencer
    main_seq = Sequencing()
    sub_seq = Sequencing()
    
    print("\n1. Defining Hierarchical Recipe:")
    
    # Sub-sequence: Mix batter
    mix_steps = [
        vocab["GET"] * vocab["BOWL"],
        vocab["GET"] * vocab["EGGS"],
        vocab["GET"] * vocab["MILK"],
        vocab["GET"] * vocab["FLOUR"],
        vocab["MIX"] * vocab["WHISK"]
    ]
    
    sub_seq.define_sequence("mix_batter", [s.vector for s in mix_steps])
    
    # Main sequence: Make pancakes
    main_steps = [
        vocab["GET"] * vocab["PAN"],
        vocab["HEAT"] * vocab["PAN"],
        vocab["MIX"],  # Trigger sub-sequence
        vocab["POUR"] * vocab["BOWL"],
        vocab["WAIT"],
        vocab["FLIP"] * vocab["SPATULA"],
        vocab["WAIT"],
        vocab["SERVE"]
    ]
    
    main_seq.define_sequence("make_pancakes", [s.vector for s in main_steps])
    
    print("   Main sequence: Make Pancakes")
    print("   - Prepare pan")
    print("   - [Mix batter sub-sequence]")
    print("   - Cook pancakes")
    
    # Execute with hierarchical control
    print("\n2. Executing Hierarchical Sequence:")
    
    main_seq.start_sequence("make_pancakes")
    
    step_count = 0
    in_subsequence = False
    
    while not main_seq.is_sequence_complete("make_pancakes") and step_count < 30:
        current = main_seq.get_current_step("make_pancakes")
        
        if current is not None:
            action_state.set_state(current)
            pointer = action_state.get_semantic_pointer(vocab)
            
            # Check if we need to execute sub-sequence
            if pointer.similarity(vocab["MIX"]) > 0.7 and not in_subsequence:
                print(f"\n   [Entering sub-sequence: Mix Batter]")
                sub_seq.start_sequence("mix_batter")
                in_subsequence = True
            
            # Execute sub-sequence if active
            if in_subsequence and not sub_seq.is_sequence_complete("mix_batter"):
                sub_current = sub_seq.get_current_step("mix_batter")
                if sub_current is not None:
                    action_state.set_state(sub_current)
                    sub_pointer = action_state.get_semantic_pointer(vocab)
                    
                    # Decode sub-action
                    for act in actions:
                        for obj in ingredients + tools:
                            test_vec = vocab[act] * vocab[obj]
                            if sub_pointer.similarity(test_vec) > 0.5:
                                print(f"     - {act} {obj}")
                                break
                    
                    sub_seq.advance_sequence("mix_batter")
                
                if sub_seq.is_sequence_complete("mix_batter"):
                    print("   [Sub-sequence complete]")
                    in_subsequence = False
                    status_state.set_state(vocab["MIXED"].vector)
            else:
                # Regular main sequence step
                for act in actions:
                    if pointer.similarity(vocab[act]) > 0.5:
                        print(f"   Step {step_count + 1}: {act}")
                        
                        # Update status based on action
                        if act == "HEAT":
                            status_state.set_state(vocab["COOKING"].vector)
                        elif act == "SERVE":
                            status_state.set_state(vocab["READY"].vector)
                        break
                
                main_seq.advance_sequence("make_pancakes")
        
        step_count += 1
    
    print("\n   Recipe complete!")
    
    return vocab, main_seq, sub_seq, status_state


def demonstrate_conditional_sequences():
    """Demonstrate conditional branching in sequences."""
    print("\n\n=== Conditional Sequences ===")
    
    # Create vocabulary
    vocab = Vocabulary(256)
    
    # Traffic light states
    lights = ["GREEN", "YELLOW", "RED", "FLASHING"]
    for light in lights:
        vocab.create_pointer(light)
    
    # Actions
    actions = ["GO", "SLOW", "STOP", "PROCEED_CAUTION", "WAIT", "CHECK"]
    for act in actions:
        vocab.create_pointer(act)
    
    # Conditions
    conditions = ["CLEAR", "PEDESTRIAN", "EMERGENCY", "TRAFFIC"]
    for cond in conditions:
        vocab.create_pointer(cond)
    
    # Create modules
    light_state = State("traffic_light", 256)
    condition_state = State("condition", 256)
    action_state = State("action", 256)
    
    # Create production system for conditional logic
    ps = ProductionSystem()
    
    print("\n1. Defining Conditional Rules:")
    
    # Rule: Green light + Clear -> GO
    rule1 = Production(
        name="green_clear_go",
        condition=Condition(
            lambda: light_state.get_semantic_pointer(vocab).similarity(vocab["GREEN"]) > 0.5 and
                   condition_state.get_semantic_pointer(vocab).similarity(vocab["CLEAR"]) > 0.5,
            "green light AND clear"
        ),
        effect=Effect(
            lambda: action_state.set_state(vocab["GO"].vector),
            "action GO"
        )
    )
    
    # Rule: Green light + Pedestrian -> WAIT
    rule2 = Production(
        name="green_pedestrian_wait",
        condition=Condition(
            lambda: light_state.get_semantic_pointer(vocab).similarity(vocab["GREEN"]) > 0.5 and
                   condition_state.get_semantic_pointer(vocab).similarity(vocab["PEDESTRIAN"]) > 0.5,
            "green light AND pedestrian"
        ),
        effect=Effect(
            lambda: action_state.set_state(vocab["WAIT"].vector),
            "action WAIT"
        ),
        priority=2.0  # Higher priority for safety
    )
    
    # Rule: Red light -> STOP
    rule3 = Production(
        name="red_stop",
        condition=Condition(
            lambda: light_state.get_semantic_pointer(vocab).similarity(vocab["RED"]) > 0.5,
            "red light"
        ),
        effect=Effect(
            lambda: action_state.set_state(vocab["STOP"].vector),
            "action STOP"
        )
    )
    
    # Rule: Yellow light + Clear -> SLOW
    rule4 = Production(
        name="yellow_slow",
        condition=Condition(
            lambda: light_state.get_semantic_pointer(vocab).similarity(vocab["YELLOW"]) > 0.5,
            "yellow light"
        ),
        effect=Effect(
            lambda: action_state.set_state(vocab["SLOW"].vector),
            "action SLOW"
        )
    )
    
    ps.add_production(rule1)
    ps.add_production(rule2)
    ps.add_production(rule3)
    ps.add_production(rule4)
    
    ps.set_context({
        "traffic_light": light_state,
        "condition": condition_state,
        "action": action_state,
        "vocab": vocab
    })
    
    print("   Rules defined for traffic scenarios")
    
    # Test scenarios
    print("\n2. Testing Conditional Sequences:")
    
    scenarios = [
        ("GREEN", "CLEAR", "Normal green"),
        ("GREEN", "PEDESTRIAN", "Green with pedestrian"),
        ("RED", "CLEAR", "Red light"),
        ("YELLOW", "CLEAR", "Yellow light")
    ]
    
    action_sequence = []
    
    for light, condition, desc in scenarios:
        print(f"\n   Scenario: {desc}")
        print(f"   Light: {light}, Condition: {condition}")
        
        light_state.set_state(vocab[light].vector)
        condition_state.set_state(vocab[condition].vector)
        
        # Execute rules
        fired = ps.step()
        if fired:
            print(f"   Rule fired: {fired.name}")
            
            # Get action
            action_ptr = action_state.get_semantic_pointer(vocab)
            matches = vocab.cleanup(action_ptr.vector, top_n=1)
            if matches:
                action = matches[0][0]
                action_sequence.append(action)
                print(f"   Action: {action}")
    
    print(f"\n   Action sequence: {' -> '.join(action_sequence)}")
    
    return vocab, ps, action_sequence


def demonstrate_interruption_handling():
    """Demonstrate interruption and recovery in sequences."""
    print("\n\n=== Interruption Handling ===")
    
    # Create vocabulary
    vocab = Vocabulary(256)
    
    # Phone call sequence
    phone_actions = ["HEAR_RING", "PICK_UP", "SAY_HELLO", "LISTEN", "RESPOND", "HANG_UP"]
    for act in phone_actions:
        vocab.create_pointer(act)
    
    # Original task actions
    task_actions = ["READ", "WRITE", "THINK", "TYPE"]
    for act in task_actions:
        vocab.create_pointer(act)
    
    # States
    states = ["WORKING", "INTERRUPTED", "RESUMING"]
    for state in states:
        vocab.create_pointer(state)
    
    # Create modules
    action = State("action", 256)
    status = State("status", 256)
    saved_state = Buffer("saved_state", 256)
    
    # Create sequencers
    main_seq = Sequencing()
    interrupt_seq = Sequencing()
    
    print("\n1. Main Task Sequence:")
    
    # Main work sequence
    work_steps = [
        vocab["READ"],
        vocab["THINK"], 
        vocab["WRITE"],
        vocab["TYPE"],
        vocab["THINK"],
        vocab["WRITE"]
    ]
    
    main_seq.define_sequence("work", [s.vector for s in work_steps])
    
    # Interruption sequence
    phone_steps = [
        vocab["HEAR_RING"],
        vocab["PICK_UP"],
        vocab["SAY_HELLO"],
        vocab["LISTEN"],
        vocab["RESPOND"],
        vocab["HANG_UP"]
    ]
    
    interrupt_seq.define_sequence("phone", [s.vector for s in phone_steps])
    
    print("   Work sequence: READ -> THINK -> WRITE -> TYPE -> THINK -> WRITE")
    print("   Phone sequence: HEAR_RING -> PICK_UP -> SAY_HELLO -> LISTEN -> RESPOND -> HANG_UP")
    
    # Execute with interruption
    print("\n2. Execution with Interruption:")
    
    main_seq.start_sequence("work")
    status.set_state(vocab["WORKING"].vector)
    
    step = 0
    interrupted_at = -1
    saved_position = -1
    
    while step < 15:
        # Check for interruption at step 3
        if step == 3 and interrupted_at == -1:
            print("\n   [INTERRUPTION: Phone rings!]")
            
            # Save current state
            current = main_seq.get_current_step("work")
            saved_state.set_state(current)
            saved_position = main_seq.sequences["work"]["position"]
            
            # Mark interruption
            status.set_state(vocab["INTERRUPTED"].vector)
            interrupted_at = step
            
            # Start interrupt sequence
            interrupt_seq.start_sequence("phone")
        
        # Execute appropriate sequence
        if status.get_semantic_pointer(vocab).similarity(vocab["INTERRUPTED"]) > 0.5:
            # Handle interruption
            if not interrupt_seq.is_sequence_complete("phone"):
                current = interrupt_seq.get_current_step("phone")
                if current is not None:
                    action.set_state(current)
                    act_ptr = action.get_semantic_pointer(vocab)
                    
                    # Find action
                    for act in phone_actions:
                        if act_ptr.similarity(vocab[act]) > 0.5:
                            print(f"   Step {step + 1}: {act} (interrupt)")
                            break
                    
                    interrupt_seq.advance_sequence("phone")
            else:
                # Interruption complete, resume main task
                print("\n   [RESUMING MAIN TASK]")
                status.set_state(vocab["RESUMING"].vector)
                
                # Restore position
                main_seq.sequences["work"]["position"] = saved_position
        
        elif status.get_semantic_pointer(vocab).similarity(vocab["RESUMING"]) > 0.5:
            # Transition back to working
            status.set_state(vocab["WORKING"].vector)
            print(f"   Resumed at position {saved_position}")
        
        else:
            # Normal work sequence
            if not main_seq.is_sequence_complete("work"):
                current = main_seq.get_current_step("work")
                if current is not None:
                    action.set_state(current)
                    act_ptr = action.get_semantic_pointer(vocab)
                    
                    # Find action
                    for act in task_actions:
                        if act_ptr.similarity(vocab[act]) > 0.5:
                            print(f"   Step {step + 1}: {act}")
                            break
                    
                    main_seq.advance_sequence("work")
            else:
                print("\n   Main task complete!")
                break
        
        step += 1
    
    return vocab, main_seq, interrupt_seq


def demonstrate_temporal_patterns():
    """Demonstrate learning temporal patterns in sequences."""
    print("\n\n=== Temporal Pattern Learning ===")
    
    # Create vocabulary
    vocab = Vocabulary(256)
    
    # Musical notes
    notes = ["C", "D", "E", "F", "G", "A", "B", "REST"]
    for note in notes:
        vocab.create_pointer(note)
    
    # Durations
    durations = ["QUARTER", "HALF", "WHOLE", "EIGHTH"]
    for dur in durations:
        vocab.create_pointer(dur)
    
    # Create modules
    note_state = State("note", 256)
    duration_state = State("duration", 256)
    pattern_memory = Memory("patterns", 256, capacity=20)
    
    print("\n1. Learning Musical Patterns:")
    
    # Pattern 1: Simple scale
    scale_pattern = ["C", "D", "E", "F", "G", "F", "E", "D", "C"]
    
    # Encode as temporal sequence
    print("   Scale pattern: C-D-E-F-G-F-E-D-C")
    
    # Store transitions
    for i in range(len(scale_pattern) - 1):
        current = vocab[scale_pattern[i]]
        next_note = vocab[scale_pattern[i + 1]]
        
        # Encode transition: current -> next
        transition = current * vocab["QUARTER"] + next_note
        pattern_memory.add_pair(current.vector, transition.vector)
    
    # Pattern 2: Chord progression
    chord_pattern = ["C", "E", "G", "C", "F", "A", "C", "F"]
    
    print("   Chord pattern: C-E-G-C-F-A-C-F")
    
    for i in range(len(chord_pattern) - 1):
        current = vocab[chord_pattern[i]]
        next_note = vocab[chord_pattern[i + 1]]
        
        transition = current * vocab["EIGHTH"] + next_note
        pattern_memory.add_pair(current.vector, transition.vector)
    
    # Test pattern prediction
    print("\n2. Pattern Prediction:")
    
    test_notes = ["C", "D", "E"]
    
    for note in test_notes:
        print(f"\n   After {note}, predict next:")
        
        # Recall from pattern memory
        recalled = pattern_memory.recall(vocab[note].vector)
        
        if recalled is not None:
            # Decode predictions
            matches = vocab.cleanup(recalled, top_n=3)
            
            predictions = []
            for match_str, sim in matches:
                # Try to extract next note
                if match_str in notes and match_str != note:
                    predictions.append((match_str, sim))
            
            if predictions:
                for pred_note, conf in predictions[:2]:
                    print(f"   - {pred_note} (confidence: {conf:.2f})")
    
    # Demonstrate pattern completion
    print("\n3. Pattern Completion:")
    
    partial = ["C", "D", "E", "F"]
    print(f"   Partial sequence: {'-'.join(partial)}")
    
    # Complete the pattern
    current_note = partial[-1]
    completed = partial.copy()
    
    for _ in range(4):  # Try to extend by 4 notes
        recalled = pattern_memory.recall(vocab[current_note].vector)
        
        if recalled is not None:
            matches = vocab.cleanup(recalled, top_n=5)
            
            # Find best next note
            best_next = None
            best_sim = 0
            
            for match_str, sim in matches:
                if match_str in notes and match_str != current_note:
                    if sim > best_sim:
                        best_next = match_str
                        best_sim = sim
            
            if best_next and best_sim > 0.2:
                completed.append(best_next)
                current_note = best_next
            else:
                break
    
    print(f"   Completed: {'-'.join(completed)}")
    
    return vocab, pattern_memory


def visualize_sequence_execution():
    """Visualize sequence execution over time."""
    print("\n\n=== Visualizing Sequence Execution ===")
    
    # Create simple sequence
    vocab = Vocabulary(128)
    
    actions = ["START", "STEP1", "STEP2", "STEP3", "STEP4", "END"]
    for act in actions:
        vocab.create_pointer(act)
    
    # Create sequence
    sequencer = Sequencing()
    sequence = [vocab[act].vector for act in actions]
    sequencer.define_sequence("demo", sequence)
    
    # Track states over time
    states = []
    sequencer.start_sequence("demo")
    
    while not sequencer.is_sequence_complete("demo"):
        current = sequencer.get_current_step("demo")
        if current is not None:
            states.append(current)
        sequencer.advance_sequence("demo")
    
    print("\n1. Sequence State Evolution:")
    
    # Create animation
    if len(states) > 0:
        anim = animate_state_evolution(
            states,
            vocab=vocab,
            top_k=3,
            interval=500
        )
        
        # Note: Animation will display if running in appropriate environment
        print("   Animation created showing state transitions")
        print(f"   Total steps: {len(states)}")
    
    # Plot activity over time
    print("\n2. Activity Pattern:")
    
    # Convert states to activity pattern
    activity = np.zeros((len(states), len(actions)))
    
    for t, state in enumerate(states):
        sp = SemanticPointer(state, vocabulary=vocab)
        for i, act in enumerate(actions):
            activity[t, i] = sp.similarity(vocab[act])
    
    # Plot activity
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(activity.T, aspect='auto', cmap='hot',
                   interpolation='nearest')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Action')
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions)
    ax.set_title('Sequence Execution Activity')
    
    plt.colorbar(im, ax=ax, label='Activation')
    plt.tight_layout()
    plt.show()
    
    print("   Visualization shows:")
    print("   - Each action activates in sequence")
    print("   - Clean transitions between steps")
    print("   - No overlap between actions")


def main():
    """Run all sequential behavior demonstrations."""
    print("=" * 60)
    print("Sequential Behavior Demonstration")
    print("=" * 60)
    
    # Run demonstrations
    vocab1, seq1, motor = create_motor_sequence()
    vocab2, main_seq, sub_seq, status = create_cooking_sequence()
    vocab3, ps, actions = demonstrate_conditional_sequences()
    vocab4, main2, interrupt = demonstrate_interruption_handling()
    vocab5, patterns = demonstrate_temporal_patterns()
    visualize_sequence_execution()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nKey Concepts Demonstrated:")
    print("- Sequential action execution")
    print("- Hierarchical sequence organization")
    print("- Conditional branching based on context")
    print("- Interruption handling and recovery")
    print("- Temporal pattern learning")
    print("- State machine implementation with SPA")
    print("- Sequence visualization and analysis")
    print("=" * 60)


if __name__ == "__main__":
    main()