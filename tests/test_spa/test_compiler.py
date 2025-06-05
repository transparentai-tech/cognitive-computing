"""Tests for SPA model compilation and building."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from cognitive_computing.spa.compiler import (
    ModuleSpec, ConnectionSpec, ActionSpec, SPAModel, ModelBuilder,
    compile_model, parse_actions, optimize_network
)
from cognitive_computing.spa.core import SPAConfig, Vocabulary, SPA
from cognitive_computing.spa.modules import State, Memory, Buffer, Gate
from cognitive_computing.spa.control import CognitiveControl
from cognitive_computing.spa.networks import Network


class TestModuleSpec:
    """Test module specification."""
    
    def test_creation(self):
        """Test module spec creation."""
        spec = ModuleSpec("motor", "state", 64, "actions", {"tau": 0.1})
        
        assert spec.name == "motor"
        assert spec.type == "state"
        assert spec.dimensions == 64
        assert spec.vocab == "actions"
        assert spec.params == {"tau": 0.1}
        
    def test_defaults(self):
        """Test default values."""
        spec = ModuleSpec("vision", "buffer", 128)
        
        assert spec.vocab is None
        assert spec.params == {}


class TestConnectionSpec:
    """Test connection specification."""
    
    def test_creation(self):
        """Test connection spec creation."""
        transform = np.random.randn(64, 64)
        spec = ConnectionSpec("vision", "motor", transform, "gate1")
        
        assert spec.source == "vision"
        assert spec.target == "motor"
        assert np.array_equal(spec.transform, transform)
        assert spec.gate == "gate1"
        
    def test_defaults(self):
        """Test default values."""
        spec = ConnectionSpec("a", "b")
        
        assert spec.transform is None
        assert spec.gate is None


class TestActionSpec:
    """Test action specification."""
    
    def test_creation(self):
        """Test action spec creation."""
        spec = ActionSpec(
            "grasp",
            "dot(vision, CIRCLE) > 0.5",
            "motor = GRASP",
            1.5
        )
        
        assert spec.name == "grasp"
        assert spec.condition == "dot(vision, CIRCLE) > 0.5"
        assert spec.effect == "motor = GRASP"
        assert spec.priority == 1.5
        
    def test_default_priority(self):
        """Test default priority."""
        spec = ActionSpec("look", "True", "gaze = TARGET")
        assert spec.priority == 0.0


class TestSPAModel:
    """Test SPA model specification."""
    
    def test_creation(self):
        """Test model creation."""
        config = SPAConfig(dimension=64)
        model = SPAModel("test_model", config)
        
        assert model.name == "test_model"
        assert model.config == config
        assert "default" in model.vocab_specs
        assert model.vocab_specs["default"] == 64
        
    def test_add_module(self):
        """Test adding modules."""
        config = SPAConfig(dimension=64)
        model = SPAModel("test", config)
        
        # Add with defaults
        model.add_module("vision", "state")
        assert "vision" in model.module_specs
        assert model.module_specs["vision"].dimensions == 64
        assert model.module_specs["vision"].vocab == "default"
        
        # Add with custom params
        model.add_module("motor", "buffer", 128, "actions", tau=0.1)
        assert model.module_specs["motor"].dimensions == 128
        assert model.module_specs["motor"].vocab == "actions"
        assert model.module_specs["motor"].params == {"tau": 0.1}
        
    def test_connect(self):
        """Test adding connections."""
        config = SPAConfig(dimension=64)
        model = SPAModel("test", config)
        
        model.connect("vision", "motor", "ROTATE_90", "gate1")
        
        assert len(model.connection_specs) == 1
        conn = model.connection_specs[0]
        assert conn.source == "vision"
        assert conn.target == "motor"
        assert conn.transform == "ROTATE_90"
        assert conn.gate == "gate1"
        
    def test_add_vocabulary(self):
        """Test adding vocabularies."""
        config = SPAConfig(dimension=64)
        model = SPAModel("test", config)
        
        model.add_vocabulary("actions", 32)
        assert model.vocab_specs["actions"] == 32
        
        model.add_vocabulary("concepts")  # Use default dimension
        assert model.vocab_specs["concepts"] == 64
        
    def test_add_action(self):
        """Test adding action rules."""
        config = SPAConfig(dimension=64)
        model = SPAModel("test", config)
        
        model.add_action(
            "grasp",
            "dot(vision, CIRCLE) > 0.5",
            "motor = GRASP",
            2.0
        )
        
        assert len(model.action_specs) == 1
        action = model.action_specs[0]
        assert action.name == "grasp"
        assert action.priority == 2.0
        
    def test_add_production(self):
        """Test adding production rules."""
        config = SPAConfig(dimension=64)
        model = SPAModel("test", config)
        
        model.add_production("IF state ~ READY THEN state = GO")
        assert len(model.production_specs) == 1
        assert model.production_specs[0] == "IF state ~ READY THEN state = GO"
        
    def test_module_dependencies(self):
        """Test dependency analysis."""
        config = SPAConfig(dimension=64)
        model = SPAModel("test", config)
        
        model.add_module("a", "state")
        model.add_module("b", "state")
        model.add_module("c", "state")
        model.add_module("gate", "gate")
        
        model.connect("a", "b")
        model.connect("b", "c", gate="gate")
        
        deps = model.get_module_dependencies()
        
        assert deps["a"] == set()
        assert deps["b"] == {"a"}
        assert deps["c"] == {"b", "gate"}
        assert deps["gate"] == set()


class TestModelBuilder:
    """Test model building."""
    
    def test_build_vocabularies(self):
        """Test vocabulary building."""
        config = SPAConfig(dimension=64)
        model = SPAModel("test", config)
        
        model.add_vocabulary("actions", 32)
        model.add_vocabulary("states", 64)
        
        builder = ModelBuilder(model)
        builder._build_vocabularies()
        
        assert len(builder.vocabularies) == 3  # default + 2 added
        assert "default" in builder.vocabularies
        assert "actions" in builder.vocabularies
        assert "states" in builder.vocabularies
        
        assert builder.vocabularies["actions"].dimension == 32
        assert builder.vocabularies["states"].dimension == 64
        
    def test_build_modules(self):
        """Test module building."""
        config = SPAConfig(dimension=64)
        model = SPAModel("test", config)
        
        model.add_module("vision", "state")
        model.add_module("motor", "buffer")
        model.add_module("gate1", "gate")
        model.add_module("control", "control")
        
        builder = ModelBuilder(model)
        builder._build_vocabularies()
        builder._build_modules()
        
        assert len(builder.modules) == 4
        assert isinstance(builder.modules["vision"], State)
        assert isinstance(builder.modules["motor"], Buffer)
        assert isinstance(builder.modules["gate1"], Gate)
        assert isinstance(builder.modules["control"], CognitiveControl)
        
    def test_build_connections(self):
        """Test connection building."""
        config = SPAConfig(dimension=64)
        model = SPAModel("test", config)
        
        model.add_module("a", "state")
        model.add_module("b", "state")
        
        model.connect("a", "b")
        
        builder = ModelBuilder(model)
        builder._build_vocabularies()
        builder._build_modules()
        builder._build_connections()
        
        # Check connection was added
        b_module = builder.modules["b"]
        assert "a_conn" in b_module.inputs
        
    def test_build_actions(self):
        """Test action system building."""
        config = SPAConfig(dimension=64)
        model = SPAModel("test", config)
        
        model.add_module("state", "state")
        model.add_action("go", "True", "state = GO")
        
        builder = ModelBuilder(model)
        builder._build_vocabularies()
        builder._build_modules()
        builder._build_actions()
        
        assert builder.action_selection is not None
        assert len(builder.action_selection.action_set.actions) == 1
        
    def test_build_complete(self):
        """Test complete model building."""
        config = SPAConfig(dimension=64)
        model = SPAModel("test", config)
        
        model.add_module("state", "state")
        model.add_module("motor", "state")
        model.connect("state", "motor")
        model.add_action("move", "state ~ READY", "motor = GO")
        
        builder = ModelBuilder(model)
        spa = builder.build()
        
        assert isinstance(spa, SPA)
        assert len(spa.modules) == 2
        assert spa.action_selection is not None
        
    def test_build_with_dependencies(self):
        """Test building with module dependencies."""
        config = SPAConfig(dimension=64)
        model = SPAModel("test", config)
        
        # Create dependency chain: c -> b -> a
        model.add_module("c", "state")
        model.add_module("b", "state")
        model.add_module("a", "state")
        
        model.connect("a", "b")
        model.connect("b", "c")
        
        builder = ModelBuilder(model)
        spa = builder.build()
        
        # All modules should be built despite order
        assert len(spa.modules) == 3
        assert all(name in spa.modules for name in ["a", "b", "c"])
        
    def test_unknown_module_type(self):
        """Test error on unknown module type."""
        config = SPAConfig(dimension=64)
        model = SPAModel("test", config)
        
        model.add_module("bad", "unknown_type")
        
        builder = ModelBuilder(model)
        builder._build_vocabularies()
        
        with pytest.raises(ValueError, match="Unknown module type"):
            builder._build_modules()


class TestCompileModel:
    """Test model compilation."""
    
    def test_compile_simple(self):
        """Test compiling simple model."""
        config = SPAConfig(dimension=64)
        model = SPAModel("simple", config)
        
        model.add_module("state", "state")
        model.add_module("motor", "state")
        
        network = compile_model(model)
        
        assert isinstance(network, Network)
        assert network.label == "compiled_simple"
        assert len(network.ensembles) == 2
        
    def test_compile_with_actions(self):
        """Test compiling model with actions."""
        config = SPAConfig(dimension=64)
        model = SPAModel("actions", config)
        
        model.add_module("vision", "state")
        model.add_module("motor", "state")
        model.add_action("look", "True", "motor = LOOK")
        
        network = compile_model(model)
        
        assert isinstance(network, Network)
        # Should still have ensemble arrays for modules
        assert len(network.ensembles) >= 2


class TestParseActions:
    """Test action parsing."""
    
    def test_parse_simple(self):
        """Test parsing simple action rules."""
        spec = """
        grasp: IF dot(vision, CIRCLE) > 0.5 THEN motor = GRASP
        look: IF True THEN gaze = TARGET
        """
        
        actions = parse_actions(spec)
        
        assert len(actions) == 2
        assert actions[0].name == "grasp"
        assert actions[0].condition == "dot(vision, CIRCLE) > 0.5"
        assert actions[0].effect == "motor = GRASP"
        assert actions[0].priority == 0.0
        
        assert actions[1].name == "look"
        assert actions[1].condition == "True"
        assert actions[1].effect == "gaze = TARGET"
        
    def test_parse_with_priority(self):
        """Test parsing with priorities."""
        spec = """
        high[2.0]: IF urgent THEN act = NOW
        low[0.5]: IF True THEN act = WAIT
        """
        
        actions = parse_actions(spec)
        
        assert len(actions) == 2
        assert actions[0].name == "high"
        assert actions[0].priority == 2.0
        assert actions[1].name == "low"
        assert actions[1].priority == 0.5
        
    def test_parse_multiline(self):
        """Test parsing multiline conditions/effects."""
        spec = """
        complex: IF dot(a, X) > 0.5 AND
                    dot(b, Y) > 0.5
                 THEN state = ACTIVE
        """
        
        actions = parse_actions(spec)
        
        assert len(actions) == 1
        assert "dot(a, X) > 0.5 AND" in actions[0].condition
        assert "dot(b, Y) > 0.5" in actions[0].condition
        
    def test_parse_empty(self):
        """Test parsing empty specification."""
        actions = parse_actions("")
        assert len(actions) == 0
        
    def test_parse_comments(self):
        """Test parsing with comments."""
        spec = """
        # This is a comment
        action1: IF True THEN go = True
        # Another comment
        """
        
        actions = parse_actions(spec)
        assert len(actions) == 1
        assert actions[0].name == "action1"


class TestOptimizeNetwork:
    """Test network optimization."""
    
    def test_optimize_basic(self):
        """Test basic optimization."""
        network = Network("test")
        
        optimized = optimize_network(network)
        
        assert optimized == network  # Currently just returns same network
        assert optimized.label == "test"
        
    @patch('cognitive_computing.spa.compiler.logger')
    def test_optimize_logging(self, mock_logger):
        """Test optimization logging."""
        network = Network("test")
        
        optimize_network(network)
        
        mock_logger.info.assert_called_once()
        assert "Optimized network 'test'" in str(mock_logger.info.call_args)


class TestIntegration:
    """Integration tests for compiler."""
    
    def test_full_model_specification(self):
        """Test complete model specification and building."""
        config = SPAConfig(dimension=64)
        model = SPAModel("cognitive_model", config)
        
        # Add vocabularies
        model.add_vocabulary("objects", 64)
        model.add_vocabulary("actions", 32)
        
        # Add modules
        model.add_module("vision", "state", vocab="objects")
        model.add_module("memory", "memory", vocab="objects")
        model.add_module("motor", "buffer", 32, vocab="actions")
        model.add_module("gate", "gate", 64)
        model.add_module("control", "control")
        
        # Add connections
        model.connect("vision", "memory")
        model.connect("memory", "motor", gate="gate")
        
        # Add actions
        model.add_action("store", "gate > 0.5", "memory = vision")
        model.add_action("recall", "vision ~ QUERY", "motor = memory")
        
        # Build model
        builder = ModelBuilder(model)
        spa = builder.build()
        
        # Verify structure
        assert len(spa.modules) == 5
        assert spa.action_selection is not None
        assert len(spa.action_selection.action_set.actions) == 2
        
        # Verify connections
        memory = spa.modules["memory"]
        assert "vision_conn" in memory.inputs
        
        motor = spa.modules["motor"]
        assert "memory_conn" in motor.inputs
        
    def test_model_with_production_system(self):
        """Test model with production rules."""
        config = SPAConfig(dimension=64)
        model = SPAModel("production_model", config)
        
        model.add_module("state", "state")
        model.add_module("goal", "state")
        
        model.add_production("IF state MATCHES IDLE THEN SET goal TO ACTIVE")
        model.add_production("IF goal MATCHES ACTIVE THEN SET state TO WORKING")
        
        builder = ModelBuilder(model)
        spa = builder.build()
        
        assert spa.production_system is not None
        assert len(spa.production_system.productions) == 2
        
    def test_compile_and_optimize(self):
        """Test compilation and optimization pipeline."""
        config = SPAConfig(dimension=64, neurons_per_dimension=50)
        model = SPAModel("pipeline_test", config)
        
        model.add_module("input", "state")
        model.add_module("hidden", "state")
        model.add_module("output", "state")
        
        model.connect("input", "hidden")
        model.connect("hidden", "output")
        
        # Compile to neural network
        network = compile_model(model)
        
        # Optimize
        optimized = optimize_network(network)
        
        assert isinstance(optimized, Network)
        assert len(optimized.ensembles) == 3