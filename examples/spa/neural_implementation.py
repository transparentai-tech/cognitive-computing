#!/usr/bin/env python3
"""
Neural Implementation of SPA Modules.

This example demonstrates the neural implementation aspects of SPA:
- Building neural networks with ensembles
- Ensemble array representations of semantic pointers
- Neural dynamics and learning
- Connection weight optimization
- Spiking neural network principles
- Biologically plausible implementations

SPA bridges the gap between symbolic and neural computation,
showing how high-level cognitive functions can emerge from
neural mechanisms.
"""

import numpy as np
import matplotlib.pyplot as plt
from cognitive_computing.spa import (
    SPAConfig, Vocabulary, SemanticPointer,
    Network, Ensemble, EnsembleArray, Connection, Probe,
    CircularConvolution, NeuronParams,
    State, Memory, Gate
)
from cognitive_computing.spa.visualizations import (
    plot_network_graph, plot_module_activity
)


def demonstrate_ensemble_basics():
    """Demonstrate basic neural ensemble properties."""
    print("\n=== Neural Ensemble Basics ===")
    
    # Create ensemble parameters
    n_neurons = 100
    dimensions = 1  # Start with scalar
    
    # Neuron parameters
    neuron_params = NeuronParams(
        tau_rc=0.02,  # Membrane time constant
        tau_ref=0.002,  # Refractory period
        gain_range=(0.5, 2.0),  # Gain distribution
        bias_range=(-1.0, 1.0)  # Bias distribution
    )
    
    # Create ensemble
    ensemble = Ensemble(
        name="scalar_ensemble",
        n_neurons=n_neurons,
        dimensions=dimensions,
        params=neuron_params
    )
    
    print(f"\n1. Ensemble Properties:")
    print(f"   Neurons: {n_neurons}")
    print(f"   Dimensions: {dimensions}")
    print(f"   Representational range: [-1, 1]")
    
    # Test encoding and decoding
    print("\n2. Encoding and Decoding:")
    
    test_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for value in test_values:
        # Encode value into neural activities
        activities = ensemble.encode(np.array([value]))
        
        # Decode back to value
        decoded = ensemble.decode(activities)
        
        # Count active neurons
        active = np.sum(activities > 0)
        
        print(f"   Value: {value:5.1f} -> {active:3d} active neurons -> Decoded: {decoded[0]:5.2f}")
    
    # Visualize tuning curves
    print("\n3. Tuning Curves:")
    
    x_values = np.linspace(-1, 1, 100)
    activities = np.zeros((len(x_values), n_neurons))
    
    for i, x in enumerate(x_values):
        activities[i] = ensemble.encode(np.array([x]))
    
    # Plot tuning curves for a few neurons
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Individual tuning curves
    neurons_to_plot = np.random.choice(n_neurons, 5, replace=False)
    for n in neurons_to_plot:
        ax1.plot(x_values, activities[:, n], label=f'Neuron {n}')
    
    ax1.set_xlabel('Represented Value')
    ax1.set_ylabel('Firing Rate')
    ax1.set_title('Individual Neuron Tuning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Population activity
    ax2.imshow(activities.T, aspect='auto', cmap='hot',
               extent=[-1, 1, 0, n_neurons])
    ax2.set_xlabel('Represented Value')
    ax2.set_ylabel('Neuron Index')
    ax2.set_title('Population Activity')
    
    plt.tight_layout()
    plt.show()
    
    return ensemble


def demonstrate_ensemble_arrays():
    """Demonstrate ensemble arrays for vector representation."""
    print("\n\n=== Ensemble Arrays for Semantic Pointers ===")
    
    # Create vocabulary
    vocab = Vocabulary(64)
    vocab.create_pointer("APPLE")
    vocab.create_pointer("ORANGE")
    vocab.create_pointer("FRUIT")
    
    # Create ensemble array
    config = SPAConfig(
        dimensions=64,
        neurons_per_dimension=50,
        subdimensions=16  # Group neurons into sub-ensembles
    )
    
    ensemble_array = EnsembleArray(
        name="semantic_array",
        dimensions=64,
        neurons_per_dimension=50,
        subdimensions=16
    )
    
    print(f"\n1. Ensemble Array Configuration:")
    print(f"   Total dimensions: {ensemble_array.dimensions}")
    print(f"   Neurons per dimension: {ensemble_array.neurons_per_dimension}")
    print(f"   Subdimensions: {ensemble_array.subdimensions}")
    print(f"   Number of sub-ensembles: {ensemble_array.dimensions // ensemble_array.subdimensions}")
    print(f"   Total neurons: {ensemble_array.n_neurons}")
    
    # Encode semantic pointers
    print("\n2. Encoding Semantic Pointers:")
    
    # Encode APPLE
    apple_activities = ensemble_array.encode(vocab["APPLE"].vector)
    apple_decoded = ensemble_array.decode(apple_activities)
    
    similarity = vocab["APPLE"].similarity(SemanticPointer(apple_decoded))
    print(f"   APPLE encoding fidelity: {similarity:.3f}")
    
    # Test superposition
    print("\n3. Superposition (Bundling):")
    
    # Create APPLE + ORANGE
    superposed = (vocab["APPLE"] + vocab["ORANGE"]).normalize()
    
    # Encode and decode
    super_activities = ensemble_array.encode(superposed.vector)
    super_decoded = ensemble_array.decode(super_activities)
    super_sp = SemanticPointer(super_decoded, vocabulary=vocab)
    
    # Check similarities
    apple_sim = super_sp.similarity(vocab["APPLE"])
    orange_sim = super_sp.similarity(vocab["ORANGE"])
    
    print(f"   Superposition contains:")
    print(f"   - APPLE: {apple_sim:.3f}")
    print(f"   - ORANGE: {orange_sim:.3f}")
    
    # Visualize sparsity
    print("\n4. Sparsity Analysis:")
    
    # Count active neurons for different pointers
    pointers = {"APPLE": vocab["APPLE"], 
                "ORANGE": vocab["ORANGE"],
                "APPLE+ORANGE": superposed}
    
    sparsity_data = {}
    
    for name, pointer in pointers.items():
        activities = ensemble_array.encode(pointer.vector)
        active_fraction = np.mean(activities > 0)
        sparsity_data[name] = active_fraction
        print(f"   {name}: {active_fraction:.1%} neurons active")
    
    return ensemble_array, vocab


def demonstrate_neural_binding():
    """Demonstrate neural implementation of binding."""
    print("\n\n=== Neural Circular Convolution ===")
    
    # Create vocabulary
    vocab = Vocabulary(128)
    vocab.create_pointer("ROLE")
    vocab.create_pointer("FILLER")
    vocab.create_pointer("AGENT")
    vocab.create_pointer("JOHN")
    
    # Create neural circular convolution
    conv_net = CircularConvolution(dimensions=128)
    
    print("\n1. Neural Binding Network:")
    print(f"   Input dimensions: 2 x {conv_net.dimensions}")
    print(f"   Output dimensions: {conv_net.dimensions}")
    print("   Operation: Circular convolution in neural substrate")
    
    # Test binding
    print("\n2. Binding Operations:")
    
    # Bind AGENT * JOHN
    a = vocab["AGENT"].vector
    b = vocab["JOHN"].vector
    
    # Neural computation
    bound_neural = conv_net.convolve(a, b)
    
    # Compare with mathematical convolution
    bound_math = CircularConvolution.convolve(a, b)
    
    # Check similarity
    similarity = np.dot(bound_neural, bound_math) / (
        np.linalg.norm(bound_neural) * np.linalg.norm(bound_math)
    )
    
    print(f"   AGENT * JOHN")
    print(f"   Neural vs mathematical similarity: {similarity:.3f}")
    
    # Test unbinding
    print("\n3. Unbinding with Inverse:")
    
    # To get JOHN from AGENT*JOHN, convolve with ~AGENT
    agent_inv = ~vocab["AGENT"]
    unbound = conv_net.convolve(bound_neural, agent_inv.vector)
    
    # Check recovery
    recovery = SemanticPointer(unbound, vocabulary=vocab)
    john_sim = recovery.similarity(vocab["JOHN"])
    
    print(f"   (AGENT*JOHN) * ~AGENT")
    print(f"   Similarity to JOHN: {john_sim:.3f}")
    
    # Demonstrate noise robustness
    print("\n4. Noise Robustness:")
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    similarities = []
    
    for noise in noise_levels:
        # Add noise to bound vector
        noisy = bound_neural + noise * np.random.randn(128)
        
        # Try to unbind
        unbound_noisy = conv_net.convolve(noisy, agent_inv.vector)
        recovery_noisy = SemanticPointer(unbound_noisy, vocabulary=vocab)
        
        sim = recovery_noisy.similarity(vocab["JOHN"])
        similarities.append(sim)
        print(f"   Noise level {noise}: similarity = {sim:.3f}")
    
    # Plot noise robustness
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, similarities, 'b-o', linewidth=2, markersize=8)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Recognition threshold')
    plt.xlabel('Noise Level')
    plt.ylabel('Recovery Similarity')
    plt.title('Neural Binding Noise Robustness')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return conv_net, vocab


def demonstrate_learning():
    """Demonstrate learning in neural SPA networks."""
    print("\n\n=== Neural Learning ===")
    
    # Create simple network
    vocab = Vocabulary(64)
    vocab.create_pointer("INPUT")
    vocab.create_pointer("OUTPUT")
    vocab.create_pointer("TARGET")
    
    # Create network components
    input_array = EnsembleArray("input", 64, neurons_per_dimension=30)
    output_array = EnsembleArray("output", 64, neurons_per_dimension=30)
    
    # Create learnable connection
    connection = Connection(
        source=input_array,
        target=output_array,
        transform=np.eye(64) * 0.1,  # Weak initial connection
        learning_rate=0.1
    )
    
    print("\n1. Initial Network:")
    print(f"   Input: {input_array.n_neurons} neurons")
    print(f"   Output: {output_array.n_neurons} neurons")
    print(f"   Connection: {connection.transform.shape}")
    print(f"   Learning rate: {connection.learning_rate}")
    
    # Training data: Learn INPUT -> TARGET mapping
    print("\n2. Training Phase:")
    
    input_vec = vocab["INPUT"].vector
    target_vec = vocab["TARGET"].vector
    
    errors = []
    n_epochs = 50
    
    for epoch in range(n_epochs):
        # Forward pass
        input_activities = input_array.encode(input_vec)
        
        # Simple connection (no full simulation)
        output_pre_acts = np.dot(connection.transform, input_vec)
        output_activities = output_array.encode(output_pre_acts)
        output_decoded = output_array.decode(output_activities)
        
        # Compute error
        error = target_vec - output_decoded
        error_magnitude = np.linalg.norm(error)
        errors.append(error_magnitude)
        
        # Update weights (simplified PES rule)
        # ΔW = α * error * input^T
        weight_update = connection.learning_rate * np.outer(error, input_vec)
        connection.transform += weight_update
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: error = {error_magnitude:.3f}")
    
    # Test learned mapping
    print("\n3. Testing Learned Mapping:")
    
    # Final forward pass
    output_final = np.dot(connection.transform, input_vec)
    output_sp = SemanticPointer(output_final, vocabulary=vocab)
    
    # Check similarity to target
    target_sim = output_sp.similarity(vocab["TARGET"])
    print(f"   INPUT -> OUTPUT similarity to TARGET: {target_sim:.3f}")
    
    # Plot learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(errors, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Error Magnitude')
    plt.title('Neural Learning Curve')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return connection, errors


def demonstrate_dynamics():
    """Demonstrate neural dynamics and temporal processing."""
    print("\n\n=== Neural Dynamics ===")
    
    # Create vocabulary
    vocab = Vocabulary(64)
    states = ["REST", "ACTIVE", "INHIBITED"]
    for state in states:
        vocab.create_pointer(state)
    
    # Create network
    network = Network()
    
    # Add ensemble with recurrent connections
    state_ens = network.add_ensemble(
        name="state",
        n_neurons=1000,
        dimensions=64
    )
    
    # Add recurrent connection (self-loop with decay)
    recurrent = network.connect(
        source=state_ens,
        target=state_ens,
        transform=np.eye(64) * 0.9,  # 90% feedback
        synapse=0.1  # 100ms synaptic filter
    )
    
    # Add probes
    probe = network.probe(state_ens, sample_rate=10)  # 10 Hz sampling
    
    print("\n1. Recurrent Network:")
    print(f"   State ensemble: {state_ens.n_neurons} neurons")
    print(f"   Recurrent strength: 0.9")
    print(f"   Synaptic time constant: 100ms")
    
    # Simulate dynamics
    print("\n2. Simulating Dynamics:")
    
    # Initialize with REST state
    initial_state = vocab["REST"].vector
    
    # Run simulation (simplified)
    dt = 0.001  # 1ms timestep
    t_total = 2.0  # 2 seconds
    n_steps = int(t_total / dt)
    
    # Track state over time
    state_history = np.zeros((n_steps, 64))
    state_history[0] = initial_state
    
    # Add perturbation at t=0.5s
    perturbation_time = int(0.5 / dt)
    perturbation = vocab["ACTIVE"].vector * 0.5
    
    # Simple dynamics simulation
    for t in range(1, n_steps):
        # Recurrent dynamics: x(t+1) = decay * x(t) + input
        state_history[t] = 0.99 * state_history[t-1]
        
        # Add perturbation
        if t == perturbation_time:
            state_history[t] += perturbation
            print(f"   Perturbation added at t={t*dt:.1f}s")
    
    # Analyze state evolution
    print("\n3. State Analysis:")
    
    # Sample times for analysis
    sample_times = [0.0, 0.5, 1.0, 1.5]
    
    for t_sample in sample_times:
        idx = int(t_sample / dt)
        state_vec = state_history[idx]
        state_sp = SemanticPointer(state_vec, vocabulary=vocab)
        
        # Find dominant state
        similarities = {}
        for state in states:
            similarities[state] = state_sp.similarity(vocab[state])
        
        dominant = max(similarities, key=similarities.get)
        print(f"   t={t_sample}s: {dominant} (sim={similarities[dominant]:.3f})")
    
    # Plot state trajectory
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    time = np.arange(n_steps) * dt
    
    # Plot state similarities over time
    for state in states:
        state_sims = []
        for t in range(n_steps):
            sp = SemanticPointer(state_history[t], vocabulary=vocab)
            state_sims.append(sp.similarity(vocab[state]))
        ax1.plot(time, state_sims, label=state, linewidth=2)
    
    ax1.axvline(x=0.5, color='k', linestyle='--', alpha=0.5, label='Perturbation')
    ax1.set_ylabel('State Similarity')
    ax1.set_title('Neural State Dynamics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot energy over time
    energy = np.linalg.norm(state_history, axis=1)
    ax2.plot(time, energy, 'b-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('State Energy')
    ax2.set_title('Total Neural Activity')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return network, state_history


def demonstrate_biological_plausibility():
    """Demonstrate biologically plausible features."""
    print("\n\n=== Biological Plausibility ===")
    
    print("\n1. Neuron Models:")
    
    # Different neuron types
    neuron_types = {
        "Regular Spiking": NeuronParams(tau_rc=0.02, tau_ref=0.002),
        "Fast Spiking": NeuronParams(tau_rc=0.01, tau_ref=0.001),
        "Slow Adapting": NeuronParams(tau_rc=0.05, tau_ref=0.003)
    }
    
    for name, params in neuron_types.items():
        print(f"   {name}:")
        print(f"   - Membrane τ: {params.tau_rc*1000:.0f}ms")
        print(f"   - Refractory τ: {params.tau_ref*1000:.0f}ms")
    
    print("\n2. Dale's Principle:")
    print("   - Excitatory neurons: Only positive weights")
    print("   - Inhibitory neurons: Only negative weights")
    print("   - Typically 80% excitatory, 20% inhibitory")
    
    # Create mixed population
    n_neurons = 100
    n_excitatory = int(0.8 * n_neurons)
    n_inhibitory = n_neurons - n_excitatory
    
    # Neuron types
    neuron_types = np.array(['E'] * n_excitatory + ['I'] * n_inhibitory)
    
    print(f"\n   Population: {n_excitatory} excitatory, {n_inhibitory} inhibitory")
    
    print("\n3. Synaptic Dynamics:")
    
    # Different synapse types
    synapse_types = {
        "AMPA (fast excitatory)": 0.002,  # 2ms
        "NMDA (slow excitatory)": 0.050,  # 50ms
        "GABA-A (fast inhibitory)": 0.008,  # 8ms
        "GABA-B (slow inhibitory)": 0.030   # 30ms
    }
    
    for synapse, tau in synapse_types.items():
        print(f"   {synapse}: τ = {tau*1000:.0f}ms")
    
    print("\n4. Sparse Connectivity:")
    
    # Connection probability
    p_connect = 0.1  # 10% connectivity
    
    # Generate sparse connection matrix
    conn_matrix = np.random.rand(n_neurons, n_neurons) < p_connect
    
    # Apply Dale's principle
    for i in range(n_neurons):
        if neuron_types[i] == 'E':
            # Excitatory: positive weights
            conn_matrix[i, :] = conn_matrix[i, :] * np.random.uniform(0, 1, n_neurons)
        else:
            # Inhibitory: negative weights
            conn_matrix[i, :] = conn_matrix[i, :] * np.random.uniform(-1, 0, n_neurons)
    
    # Analyze connectivity
    actual_connections = np.sum(conn_matrix != 0)
    possible_connections = n_neurons * n_neurons
    actual_probability = actual_connections / possible_connections
    
    print(f"   Target connectivity: {p_connect:.1%}")
    print(f"   Actual connectivity: {actual_probability:.1%}")
    print(f"   Total connections: {actual_connections}")
    
    # Visualize connectivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Connection matrix
    im1 = ax1.imshow(conn_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xlabel('Post-synaptic')
    ax1.set_ylabel('Pre-synaptic')
    ax1.set_title('Sparse Connection Matrix')
    ax1.axhline(y=n_excitatory-0.5, color='k', linewidth=2)
    ax1.axvline(x=n_excitatory-0.5, color='k', linewidth=2)
    
    # Add E/I labels
    ax1.text(-5, n_excitatory/2, 'E', ha='center', va='center', fontsize=12)
    ax1.text(-5, n_excitatory + n_inhibitory/2, 'I', ha='center', va='center', fontsize=12)
    
    # Weight distribution
    weights = conn_matrix[conn_matrix != 0]
    ax2.hist(weights, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='k', linestyle='--')
    ax2.set_xlabel('Synaptic Weight')
    ax2.set_ylabel('Count')
    ax2.set_title('Weight Distribution (Dale\'s Principle)')
    
    plt.tight_layout()
    plt.show()
    
    return conn_matrix, neuron_types


def visualize_spa_network():
    """Visualize a complete SPA network."""
    print("\n\n=== Complete SPA Network Visualization ===")
    
    # Create network with multiple modules
    network = Network()
    
    # Add modules
    visual = network.add_ensemble("visual", n_neurons=500, dimensions=64)
    motor = network.add_ensemble("motor", n_neurons=500, dimensions=64) 
    memory = network.add_ensemble("memory", n_neurons=800, dimensions=64)
    control = network.add_ensemble("control", n_neurons=300, dimensions=32)
    
    # Add connections
    network.connect(visual, memory, transform=np.eye(64) * 0.8)
    network.connect(memory, motor, transform=np.eye(64) * 0.6)
    network.connect(control, memory, transform=np.random.randn(64, 32) * 0.1)
    network.connect(memory, memory, transform=np.eye(64) * 0.9)  # Recurrent
    
    print("\n1. Network Structure:")
    print(f"   Visual: {visual.n_neurons} neurons")
    print(f"   Motor: {motor.n_neurons} neurons")
    print(f"   Memory: {memory.n_neurons} neurons")
    print(f"   Control: {control.n_neurons} neurons")
    print(f"   Total: {visual.n_neurons + motor.n_neurons + memory.n_neurons + control.n_neurons} neurons")
    
    # Visualize network graph
    print("\n2. Network Connectivity:")
    
    fig, ax = plot_network_graph(network, layout="hierarchical")
    plt.title("SPA Neural Network Architecture")
    plt.show()
    
    print("\n   Network shows:")
    print("   - Hierarchical organization")
    print("   - Recurrent connections in memory")
    print("   - Top-down control from control module")
    print("   - Feed-forward visual to motor pathway")


def main():
    """Run all neural implementation demonstrations."""
    print("=" * 60)
    print("Neural Implementation of SPA")
    print("=" * 60)
    
    # Run demonstrations
    ensemble = demonstrate_ensemble_basics()
    ensemble_array, vocab1 = demonstrate_ensemble_arrays()
    conv_net, vocab2 = demonstrate_neural_binding()
    connection, errors = demonstrate_learning()
    network, dynamics = demonstrate_dynamics()
    conn_matrix, types = demonstrate_biological_plausibility()
    visualize_spa_network()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nKey Concepts Demonstrated:")
    print("- Neural ensembles with tuning curves")
    print("- Ensemble arrays for high-dimensional vectors")
    print("- Neural implementation of binding operations")
    print("- Learning through synaptic weight updates")
    print("- Recurrent dynamics and temporal processing")
    print("- Biologically plausible constraints")
    print("- Complete SPA network architecture")
    print("=" * 60)


if __name__ == "__main__":
    main()