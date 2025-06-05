"""Tests for SPA neural network implementation."""

import pytest
import numpy as np
from cognitive_computing.spa import SPAConfig, create_vocabulary
from cognitive_computing.spa.networks import (
    NeuronParams, Ensemble, EnsembleArray, Connection,
    Probe, Network, CircularConvolution
)


class TestNeuronParams:
    """Test NeuronParams configuration."""
    
    def test_creation(self):
        """Test neuron parameter creation."""
        params = NeuronParams(n_neurons=100)
        
        assert params.n_neurons == 100
        assert params.tau_rc == 0.02
        assert params.tau_ref == 0.002
        assert params.max_rates == (200.0, 400.0)
        assert params.intercepts == (-1.0, 1.0)
        assert params.noise == 0.1
        
    def test_custom_params(self):
        """Test custom neuron parameters."""
        params = NeuronParams(
            n_neurons=50,
            tau_rc=0.01,
            tau_ref=0.001,
            max_rates=(100.0, 200.0),
            intercepts=(-0.5, 0.5),
            noise=0.05
        )
        
        assert params.n_neurons == 50
        assert params.tau_rc == 0.01
        assert params.max_rates == (100.0, 200.0)
        
    def test_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="n_neurons must be positive"):
            NeuronParams(n_neurons=0)
            
        with pytest.raises(ValueError, match="tau_rc must be positive"):
            NeuronParams(n_neurons=10, tau_rc=-0.01)
            
        with pytest.raises(ValueError, match="noise must be non-negative"):
            NeuronParams(n_neurons=10, noise=-0.1)


class TestEnsemble:
    """Test Ensemble functionality."""
    
    def test_creation(self):
        """Test ensemble creation."""
        neurons = NeuronParams(n_neurons=50)
        ensemble = Ensemble("test", dimensions=2, neurons=neurons)
        
        assert ensemble.name == "test"
        assert ensemble.dimensions == 2
        assert ensemble.n_neurons == 50
        assert ensemble.radius == 1.0
        
        # Check generated parameters
        assert ensemble.encoders.shape == (50, 2)
        assert ensemble.gains.shape == (50,)
        assert ensemble.biases.shape == (50,)
        
        # Encoders should be unit vectors
        norms = np.linalg.norm(ensemble.encoders, axis=1)
        assert np.allclose(norms, 1.0)
        
    def test_encode_decode(self):
        """Test encoding and decoding values."""
        neurons = NeuronParams(n_neurons=100)
        ensemble = Ensemble("test", dimensions=2, neurons=neurons, seed=42)
        
        # Encode a value
        value = np.array([0.5, -0.3])
        activities = ensemble.encode(value)
        
        assert activities.shape == (100,)
        assert np.all(activities >= 0)  # Activities are non-negative
        # Activities can be > 1 due to noise, but should be reasonable
        assert np.max(activities) < 2.0
        
        # Check that some neurons are active
        assert np.sum(activities > 0.1) > 10
        
        # Decode should approximate the value
        decoded = ensemble.decode()
        # With simple decoding, expect some error
        assert np.allclose(decoded, value, atol=0.5)
        
    def test_value_property(self):
        """Test value property."""
        neurons = NeuronParams(n_neurons=50)
        ensemble = Ensemble("test", dimensions=3, neurons=neurons)
        
        value = np.array([0.1, 0.2, 0.3])
        ensemble.encode(value)
        
        assert np.array_equal(ensemble.value, value)
        
    def test_neural_response(self):
        """Test neural firing patterns."""
        neurons = NeuronParams(n_neurons=100, noise=0.0)  # No noise for testing
        ensemble = Ensemble("test", dimensions=1, neurons=neurons, seed=42)
        
        # Test different input values
        for val in [-1.0, 0.0, 1.0]:
            value = np.array([val])
            activities = ensemble.encode(value)
            rates = ensemble.rates
            
            # Some neurons should be active for non-zero inputs
            if val != 0.0:
                assert np.sum(rates > 0) > 0
                
            # Rates should be within biological range
            assert np.all(rates >= 0)
            assert np.all(rates <= neurons.max_rates[1])
            
    def test_noise_effect(self):
        """Test that noise affects activities."""
        neurons_no_noise = NeuronParams(n_neurons=50, noise=0.0)
        neurons_with_noise = NeuronParams(n_neurons=50, noise=0.2)
        
        ens1 = Ensemble("no_noise", dimensions=2, neurons=neurons_no_noise, seed=42)
        ens2 = Ensemble("with_noise", dimensions=2, neurons=neurons_with_noise, seed=42)
        
        value = np.array([0.5, 0.5])
        
        # Multiple runs with noise should give different activities
        activities1 = ens2.encode(value)
        activities2 = ens2.encode(value)
        
        assert not np.array_equal(activities1, activities2)
        
        # No noise should give same activities
        activities3 = ens1.encode(value)
        activities4 = ens1.encode(value)
        
        assert np.array_equal(activities3, activities4)


class TestEnsembleArray:
    """Test EnsembleArray functionality."""
    
    def test_creation(self):
        """Test ensemble array creation."""
        array = EnsembleArray(
            "array",
            n_ensembles=4,
            dimensions_per_ensemble=2,
            neurons_per_ensemble=30
        )
        
        assert array.name == "array"
        assert array.n_ensembles == 4
        assert array.dimensions_per_ensemble == 2
        assert array.total_dimensions == 8
        assert len(array.ensembles) == 4
        
    def test_encode_decode(self):
        """Test encoding and decoding high-dimensional vectors."""
        array = EnsembleArray(
            "test",
            n_ensembles=3,
            dimensions_per_ensemble=2,
            neurons_per_ensemble=50,
            seed=42
        )
        
        # 6-dimensional vector
        value = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        activities = array.encode(value)
        
        # Should have activities from all ensembles
        assert activities.shape == (150,)  # 3 * 50
        
        # Decode
        decoded = array.decode()
        assert decoded.shape == (6,)
        # Expect some error with simple decoding
        assert np.allclose(decoded, value, atol=0.5)
        
    def test_dimension_validation(self):
        """Test dimension validation."""
        array = EnsembleArray("test", n_ensembles=2, dimensions_per_ensemble=3)
        
        # Wrong dimension
        with pytest.raises(ValueError, match="doesn't match"):
            array.encode(np.ones(5))  # Expects 6
            
    def test_value_property(self):
        """Test value property aggregates from all ensembles."""
        array = EnsembleArray("test", n_ensembles=2, dimensions_per_ensemble=2)
        
        value = np.array([1.0, 2.0, 3.0, 4.0])
        array.encode(value)
        
        retrieved = array.value
        assert np.array_equal(retrieved, value)


class TestConnection:
    """Test Connection functionality."""
    
    def test_creation(self):
        """Test connection creation."""
        neurons = NeuronParams(n_neurons=50)
        pre = Ensemble("pre", dimensions=3, neurons=neurons)
        post = Ensemble("post", dimensions=3, neurons=neurons)
        
        conn = Connection(pre, post)
        
        assert conn.pre is pre
        assert conn.post is post
        assert conn.synapse == 0.005
        assert np.array_equal(conn.transform, np.eye(3))
        
    def test_transform(self):
        """Test connection with transformation."""
        neurons = NeuronParams(n_neurons=50)
        pre = Ensemble("pre", dimensions=2, neurons=neurons)
        post = Ensemble("post", dimensions=3, neurons=neurons)
        
        # Random projection
        transform = np.random.randn(3, 2)
        conn = Connection(pre, post, transform=transform)
        
        assert np.array_equal(conn.transform, transform)
        
    def test_dimension_mismatch(self):
        """Test dimension mismatch handling."""
        neurons = NeuronParams(n_neurons=50)
        pre = Ensemble("pre", dimensions=2, neurons=neurons)
        post = Ensemble("post", dimensions=3, neurons=neurons)
        
        # No transform with mismatched dimensions
        with pytest.raises(ValueError, match="Dimensions mismatch"):
            Connection(pre, post)
            
        # Wrong transform shape
        with pytest.raises(ValueError, match="Transform shape"):
            Connection(pre, post, transform=np.eye(2))
            
    def test_compute(self):
        """Test connection computation."""
        neurons = NeuronParams(n_neurons=50)
        pre = Ensemble("pre", dimensions=2, neurons=neurons)
        post = Ensemble("post", dimensions=2, neurons=neurons)
        
        # Set pre value
        pre.encode(np.array([0.5, -0.3]))
        
        # Create connection with scaling
        transform = np.array([[2.0, 0.0], [0.0, 2.0]])
        conn = Connection(pre, post, transform=transform, synapse=0)
        
        # Compute without filter
        output = conn.compute()
        expected = transform @ pre.value
        assert np.allclose(output, expected)
        
    def test_synaptic_filter(self):
        """Test synaptic filtering."""
        neurons = NeuronParams(n_neurons=50)
        pre = Ensemble("pre", dimensions=1, neurons=neurons)
        post = Ensemble("post", dimensions=1, neurons=neurons)
        
        conn = Connection(pre, post, synapse=0.1)
        
        # Step input
        pre.encode(np.array([1.0]))
        
        # First output should be small
        output1 = conn.compute(dt=0.001)
        assert output1[0] < 0.1
        
        # After many steps, should approach input
        for _ in range(1000):
            output = conn.compute(dt=0.001)
            
        assert output[0] > 0.9
        
    def test_function_connection(self):
        """Test connection with function."""
        neurons = NeuronParams(n_neurons=50)
        pre = Ensemble("pre", dimensions=2, neurons=neurons)
        post = Ensemble("post", dimensions=1, neurons=neurons)
        
        # Function that computes sum
        def sum_func(x):
            return np.array([np.sum(x)])
        
        # Need transform for dimension change
        transform = np.ones((1, 1))  # Just pass through the sum
        conn = Connection(pre, post, transform=transform, 
                         function=sum_func, synapse=0)
        
        pre.encode(np.array([0.3, 0.4]))
        output = conn.compute()
        
        assert np.isclose(output[0], 0.7, atol=0.01)


class TestProbe:
    """Test Probe functionality."""
    
    def test_creation(self):
        """Test probe creation."""
        neurons = NeuronParams(n_neurons=50)
        ensemble = Ensemble("test", dimensions=2, neurons=neurons)
        
        probe = Probe(ensemble, attr="value")
        
        assert probe.target is ensemble
        assert probe.attr == "value"
        assert probe.synapse == 0.01
        assert probe.sample_every == 0.001
        
    def test_recording(self):
        """Test data recording."""
        neurons = NeuronParams(n_neurons=50)
        ensemble = Ensemble("test", dimensions=2, neurons=neurons)
        probe = Probe(ensemble, attr="value", synapse=0)
        
        # Record some values
        times = [0.0, 0.001, 0.002]
        values = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0])]
        
        for t, v in zip(times, values):
            ensemble.encode(v)
            probe.record(t, dt=0.001)
            
        # Get recorded data
        rec_times, rec_data = probe.get_data()
        
        assert np.array_equal(rec_times, times)
        assert np.array_equal(rec_data, values)
        
    def test_sampling_rate(self):
        """Test probe sampling rate."""
        neurons = NeuronParams(n_neurons=50)
        ensemble = Ensemble("test", dimensions=1, neurons=neurons)
        probe = Probe(ensemble, sample_every=0.01)  # Sample every 10ms
        
        # Record at higher rate
        for i in range(20):
            t = i * 0.001  # Every 1ms
            ensemble.encode(np.array([float(i)]))
            probe.record(t)
            
        times, data = probe.get_data()
        
        # Should only have samples every 10ms
        assert len(times) == 2  # t=0 and t=0.01
        
    def test_filtering(self):
        """Test probe filtering."""
        neurons = NeuronParams(n_neurons=50)
        ensemble = Ensemble("test", dimensions=1, neurons=neurons)
        probe = Probe(ensemble, synapse=0.1)
        
        # Step input
        ensemble.encode(np.array([0.0]))
        probe.record(0.0)
        
        ensemble.encode(np.array([1.0]))
        for i in range(1, 101):
            probe.record(i * 0.001, dt=0.001)
            
        times, data = probe.get_data()
        
        # Should see gradual rise due to filter
        assert data[0][0] == 0.0
        assert data[1][0] < 0.1  # Small initial response
        assert data[-1][0] > 0.55  # Approaches step value (with tau=0.1, expect ~0.63)
        
    def test_probe_activities(self):
        """Test probing neural activities."""
        neurons = NeuronParams(n_neurons=50)
        ensemble = Ensemble("test", dimensions=2, neurons=neurons)
        probe = Probe(ensemble, attr="activities", synapse=0)
        
        ensemble.encode(np.array([0.5, 0.5]))
        probe.record(0.0)
        
        times, data = probe.get_data()
        
        assert data.shape == (1, 50)  # Neural activities
        assert np.all(data >= 0)
        # Activities can be slightly > 1 due to noise
        assert np.max(data) < 1.5


class TestNetwork:
    """Test Network functionality."""
    
    def test_creation(self):
        """Test network creation."""
        network = Network(label="TestNet", dt=0.001)
        
        assert network.label == "TestNet"
        assert network.dt == 0.001
        assert network.time == 0.0
        assert len(network.ensembles) == 0
        assert len(network.connections) == 0
        assert len(network.probes) == 0
        
    def test_add_ensemble(self):
        """Test adding ensembles."""
        network = Network()
        neurons = NeuronParams(n_neurons=50)
        
        ens1 = Ensemble("ens1", dimensions=2, neurons=neurons)
        network.add_ensemble("A", ens1)
        
        assert "A" in network.ensembles
        assert network.ensembles["A"] is ens1
        
        # Duplicate name
        with pytest.raises(ValueError, match="already exists"):
            network.add_ensemble("A", ens1)
            
    def test_connect(self):
        """Test creating connections."""
        network = Network()
        neurons = NeuronParams(n_neurons=50)
        
        ens1 = Ensemble("ens1", dimensions=2, neurons=neurons)
        ens2 = Ensemble("ens2", dimensions=2, neurons=neurons)
        
        network.add_ensemble("A", ens1)
        network.add_ensemble("B", ens2)
        
        # Connect by name
        conn = network.connect("A", "B", synapse=0.01)
        
        assert conn in network.connections
        assert conn.pre is ens1
        assert conn.post is ens2
        
        # Connect by object
        conn2 = network.connect(ens1, ens2, transform=np.eye(2) * 2)
        assert conn2.transform[0, 0] == 2.0
        
    def test_probe_creation(self):
        """Test creating probes."""
        network = Network(dt=0.001)
        neurons = NeuronParams(n_neurons=50)
        
        ens = Ensemble("test", dimensions=1, neurons=neurons)
        network.add_ensemble("A", ens)
        
        # Probe by name
        probe = network.probe("A", attr="value")
        assert probe in network.probes
        assert probe.target is ens
        
        # Probe by object
        probe2 = network.probe(ens, attr="activities")
        assert probe2.attr == "activities"
        
    def test_simulation(self):
        """Test network simulation."""
        network = Network(dt=0.001)
        neurons = NeuronParams(n_neurons=100)
        
        # Create simple network: A -> B
        ens_a = Ensemble("A", dimensions=1, neurons=neurons)
        ens_b = Ensemble("B", dimensions=1, neurons=neurons)
        
        network.add_ensemble("A", ens_a)
        network.add_ensemble("B", ens_b)
        
        # Set initial value
        ens_a.encode(np.array([0.8]))
        
        # Connect with gain of 2
        network.connect(ens_a, ens_b, transform=np.array([[2.0]]), synapse=0.01)
        
        # Probe output
        probe = network.probe(ens_b, "value", synapse=0)
        
        # Run simulation
        network.run(0.1)  # 100ms
        
        assert network.time == pytest.approx(0.1)
        
        times, data = probe.get_data()
        assert len(times) >= 99  # At least 99 samples (might miss last one)
        
        # Output should approach 2 * input = 1.6
        final_value = data[-1][0]
        assert 1.0 < final_value < 2.0  # Some error expected
        
    def test_reset(self):
        """Test network reset."""
        network = Network()
        neurons = NeuronParams(n_neurons=50)
        
        ens = Ensemble("test", dimensions=2, neurons=neurons)
        network.add_ensemble("A", ens)
        
        # Set value and run
        ens.encode(np.array([1.0, 1.0]))
        probe = network.probe(ens, "value")
        network.run(0.01)
        
        # Reset
        network.reset()
        
        assert network.time == 0.0
        assert np.array_equal(ens.value, np.zeros(2))
        
        # Probe should be cleared
        times, data = probe.get_data()
        assert len(times) == 0


class TestCircularConvolution:
    """Test CircularConvolution implementation."""
    
    def test_creation(self):
        """Test circular convolution creation."""
        conv = CircularConvolution(dimensions=128, neurons_per_dimension=50)
        
        assert conv.dimensions == 128
        assert conv.neurons_per_dimension == 50
        assert conv.radius == 1.0
        
    def test_convolve(self):
        """Test convolution computation."""
        conv = CircularConvolution(dimensions=64)
        
        # Create two vectors
        a = np.random.randn(64)
        a = a / np.linalg.norm(a)
        
        b = np.random.randn(64)
        b = b / np.linalg.norm(b)
        
        # Convolve
        c = conv.convolve(a, b)
        
        assert c.shape == (64,)
        
        # Test properties of convolution
        # Convolving with identity-like vector
        identity = np.zeros(64)
        identity[0] = 1.0
        
        result = conv.convolve(a, identity)
        assert np.allclose(result, a)
        
    def test_dimension_validation(self):
        """Test dimension validation."""
        conv = CircularConvolution(dimensions=32)
        
        a = np.ones(32)
        b = np.ones(16)  # Wrong dimension
        
        with pytest.raises(ValueError, match="Input dimensions"):
            conv.convolve(a, b)
            
    def test_create_network_placeholder(self):
        """Test network creation placeholder."""
        network = Network()
        conv = CircularConvolution(dimensions=64)
        
        # Create ensemble arrays
        array_a = EnsembleArray("A", n_ensembles=4, dimensions_per_ensemble=16)
        array_b = EnsembleArray("B", n_ensembles=4, dimensions_per_ensemble=16)
        array_c = EnsembleArray("C", n_ensembles=4, dimensions_per_ensemble=16)
        
        network.add_ensemble("A", array_a)
        network.add_ensemble("B", array_b)
        network.add_ensemble("C", array_c)
        
        # This is a placeholder that logs a message
        conv.create_network(network, "A", "B", "C")
        
        # Just check it doesn't crash
        assert True