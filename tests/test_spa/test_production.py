"""Tests for SPA production system."""

import pytest
import numpy as np
from cognitive_computing.spa import (
    SPAConfig, SemanticPointer, Vocabulary, create_vocabulary
)
from cognitive_computing.spa.modules import State, Module
from cognitive_computing.spa.production import (
    Condition, MatchCondition, CompareCondition, CompoundCondition,
    Effect, SetEffect, BindEffect, CompoundEffect,
    Production, ProductionSystem, ConditionalModule,
    parse_production_rules
)


class TestConditions:
    """Test condition classes."""
    
    def test_match_condition_string_pattern(self):
        """Test match condition with string pattern."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("CIRCLE")
        vocab.create_pointer("SQUARE")
        
        vision = State("vision", 128, vocab)
        vision.set_semantic_pointer(vocab["CIRCLE"])
        
        context = {'modules': {'vision': vision}, 'vocab': vocab}
        
        # Match condition
        cond = MatchCondition("vision", "CIRCLE", threshold=0.7)
        strength = cond.evaluate(context)
        
        assert strength > 0.9  # Should be close to 1.0
        
        # Non-match
        cond2 = MatchCondition("vision", "SQUARE", threshold=0.7)
        strength2 = cond2.evaluate(context)
        
        assert strength2 == 0.0
        
    def test_match_condition_semantic_pointer(self):
        """Test match condition with SemanticPointer pattern."""
        vocab = create_vocabulary(128)
        sp = vocab.create_pointer("TEST")
        
        state = State("state", 128, vocab)
        state.set_semantic_pointer(sp)
        
        context = {'modules': {'state': state}, 'vocab': vocab}
        
        cond = MatchCondition("state", sp, threshold=0.8)
        strength = cond.evaluate(context)
        
        assert strength > 0.9
        
    def test_match_condition_vector_pattern(self):
        """Test match condition with vector pattern."""
        state = State("state", 64)
        pattern = np.random.randn(64)
        pattern = pattern / np.linalg.norm(pattern)
        
        state.state = pattern.copy()
        
        context = {'modules': {'state': state}}
        
        cond = MatchCondition("state", pattern, threshold=0.9)
        strength = cond.evaluate(context)
        
        assert strength > 0.9
        
    def test_match_condition_threshold_scaling(self):
        """Test threshold scaling behavior."""
        state = State("state", 64)
        pattern = np.random.randn(64)
        pattern = pattern / np.linalg.norm(pattern)
        
        # Set state to have known similarity with pattern
        # Use orthogonal vector for noise
        noise = np.random.randn(64)
        noise = noise - np.dot(noise, pattern) * pattern  # Make orthogonal
        noise = noise / np.linalg.norm(noise)
        
        # Create state with exactly 0.8 similarity
        state.state = 0.8 * pattern + 0.6 * noise  # 0.8^2 + 0.6^2 = 1
        state.state = state.state / np.linalg.norm(state.state)
        
        context = {'modules': {'state': state}}
        
        # Verify similarity is what we expect
        similarity = np.dot(state.state, pattern)
        assert np.isclose(similarity, 0.8, atol=0.01)
        
        # With threshold 0.7, should match
        cond = MatchCondition("state", pattern, threshold=0.7)
        strength = cond.evaluate(context)
        
        assert 0 < strength < 1  # Scaled strength
        # Should be (0.8 - 0.7) / (1 - 0.7) ≈ 0.33
        assert np.isclose(strength, 1/3, atol=0.1)
        
        # With threshold 0.9, should not match
        cond2 = MatchCondition("state", pattern, threshold=0.9)
        strength2 = cond2.evaluate(context)
        
        assert strength2 == 0.0
        
    def test_compare_condition(self):
        """Test compare condition between modules."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("A")
        
        state1 = State("state1", 128, vocab)
        state2 = State("state2", 128, vocab)
        
        state1.set_semantic_pointer(vocab["A"])
        state2.set_semantic_pointer(vocab["A"])
        
        context = {'modules': {'state1': state1, 'state2': state2}}
        
        cond = CompareCondition("state1", "state2", threshold=0.8)
        strength = cond.evaluate(context)
        
        assert strength > 0.9  # Same state
        
    def test_compound_condition_and(self):
        """Test compound AND condition."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("RED")
        vocab.create_pointer("CIRCLE")
        
        color = State("color", 128, vocab)
        shape = State("shape", 128, vocab)
        
        color.set_semantic_pointer(vocab["RED"])
        shape.set_semantic_pointer(vocab["CIRCLE"])
        
        context = {'modules': {'color': color, 'shape': shape}, 'vocab': vocab}
        
        # Both conditions match
        cond1 = MatchCondition("color", "RED", threshold=0.7)
        cond2 = MatchCondition("shape", "CIRCLE", threshold=0.7)
        
        and_cond = CompoundCondition([cond1, cond2], "and")
        strength = and_cond.evaluate(context)
        
        assert strength > 0.9  # Both match
        
        # One doesn't match
        shape.set_semantic_pointer(vocab.create_pointer("SQUARE"))
        strength2 = and_cond.evaluate(context)
        
        assert strength2 == 0.0  # AND fails
        
    def test_compound_condition_or(self):
        """Test compound OR condition."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("RED")
        vocab.create_pointer("BLUE")
        
        color = State("color", 128, vocab)
        color.set_semantic_pointer(vocab["RED"])
        
        context = {'modules': {'color': color}, 'vocab': vocab}
        
        cond1 = MatchCondition("color", "RED", threshold=0.7)
        cond2 = MatchCondition("color", "BLUE", threshold=0.7)
        
        or_cond = CompoundCondition([cond1, cond2], "or")
        strength = or_cond.evaluate(context)
        
        assert strength > 0.9  # First matches
        
    def test_compound_condition_not(self):
        """Test compound NOT condition."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("ENEMY")
        vocab.create_pointer("FRIEND")
        
        target = State("target", 128, vocab)
        target.set_semantic_pointer(vocab["FRIEND"])
        
        context = {'modules': {'target': target}, 'vocab': vocab}
        
        cond = MatchCondition("target", "ENEMY", threshold=0.7)
        not_cond = CompoundCondition([cond], "not")
        
        strength = not_cond.evaluate(context)
        
        assert strength == 1.0  # NOT enemy
        
    def test_condition_missing_module(self):
        """Test condition with missing module."""
        context = {'modules': {}}
        
        cond = MatchCondition("missing", "pattern")
        strength = cond.evaluate(context)
        
        assert strength == 0.0
        
    def test_condition_missing_vocab_item(self):
        """Test condition with missing vocabulary item."""
        vocab = create_vocabulary(128)
        state = State("state", 128, vocab)
        
        context = {'modules': {'state': state}, 'vocab': vocab}
        
        cond = MatchCondition("state", "MISSING", threshold=0.7)
        strength = cond.evaluate(context)
        
        assert strength == 0.0


class TestEffects:
    """Test effect classes."""
    
    def test_set_effect_string_value(self):
        """Test set effect with string value."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("GRASP")
        
        motor = State("motor", 128, vocab)
        context = {'modules': {'motor': motor}, 'vocab': vocab}
        
        effect = SetEffect("motor", "GRASP")
        effect.execute(context)
        
        assert np.allclose(motor.state, vocab["GRASP"].vector)
        
    def test_set_effect_semantic_pointer(self):
        """Test set effect with SemanticPointer value."""
        vocab = create_vocabulary(128)
        sp = vocab.create_pointer("ACTION")
        
        state = State("state", 128, vocab)
        context = {'modules': {'state': state}}
        
        effect = SetEffect("state", sp)
        effect.execute(context)
        
        assert np.allclose(state.state, sp.vector)
        
    def test_set_effect_vector_value(self):
        """Test set effect with vector value."""
        state = State("state", 64)
        value = np.random.randn(64)
        
        context = {'modules': {'state': state}}
        
        effect = SetEffect("state", value)
        effect.execute(context)
        
        assert np.array_equal(state.state, value)
        
    def test_bind_effect_modules(self):
        """Test bind effect with module sources."""
        vocab = create_vocabulary(128)
        role = vocab.create_pointer("ROLE")
        filler = vocab.create_pointer("FILLER")
        
        role_mod = State("role", 128, vocab)
        filler_mod = State("filler", 128, vocab)
        result_mod = State("result", 128, vocab)
        
        role_mod.set_semantic_pointer(role)
        filler_mod.set_semantic_pointer(filler)
        
        context = {'modules': {
            'role': role_mod,
            'filler': filler_mod,
            'result': result_mod
        }}
        
        effect = BindEffect("result", "role", "filler")
        effect.execute(context)
        
        # Check binding result
        expected = role * filler
        assert np.allclose(result_mod.state, expected.vector, atol=1e-6)
        
    def test_bind_effect_mixed_sources(self):
        """Test bind effect with mixed module/vocab sources."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("VERB")
        vocab.create_pointer("OBJECT")
        
        verb_mod = State("verb", 128, vocab)
        verb_mod.set_semantic_pointer(vocab["VERB"])
        
        result_mod = State("result", 128, vocab)
        
        context = {'modules': {
            'verb': verb_mod,
            'result': result_mod
        }, 'vocab': vocab}
        
        # Bind module state with vocab item
        effect = BindEffect("result", "verb", "OBJECT")
        effect.execute(context)
        
        expected = vocab["VERB"] * vocab["OBJECT"]
        assert np.allclose(result_mod.state, expected.vector, atol=1e-6)
        
    def test_compound_effect(self):
        """Test compound effect execution."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        
        state1 = State("state1", 128, vocab)
        state2 = State("state2", 128, vocab)
        
        context = {'modules': {
            'state1': state1,
            'state2': state2
        }, 'vocab': vocab}
        
        # Multiple effects
        effects = [
            SetEffect("state1", "A"),
            SetEffect("state2", "B")
        ]
        
        compound = CompoundEffect(effects)
        compound.execute(context)
        
        assert np.allclose(state1.state, vocab["A"].vector)
        assert np.allclose(state2.state, vocab["B"].vector)
        
    def test_effect_missing_module(self):
        """Test effect with missing module."""
        context = {'modules': {}}
        
        effect = SetEffect("missing", "value")
        # Should not raise, just log warning
        effect.execute(context)
        
    def test_effect_missing_value(self):
        """Test effect with missing vocabulary value."""
        vocab = create_vocabulary(128)
        state = State("state", 128, vocab)
        
        context = {'modules': {'state': state}, 'vocab': vocab}
        
        effect = SetEffect("state", "MISSING")
        # Should not raise, just log warning
        effect.execute(context)


class TestProduction:
    """Test Production class."""
    
    def test_production_creation(self):
        """Test creating a production."""
        cond = MatchCondition("vision", "CIRCLE", threshold=0.7)
        effect = SetEffect("motor", "GRASP")
        
        prod = Production("see_circle_grasp", cond, effect, priority=1.0)
        
        assert prod.name == "see_circle_grasp"
        assert prod.condition is cond
        assert prod.effect is effect
        assert prod.priority == 1.0
        assert prod._strength == 0.0
        assert prod._fired_count == 0
        
    def test_production_evaluate(self):
        """Test evaluating production strength."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("CIRCLE")
        
        vision = State("vision", 128, vocab)
        vision.set_semantic_pointer(vocab["CIRCLE"])
        
        context = {'modules': {'vision': vision}, 'vocab': vocab}
        
        cond = MatchCondition("vision", "CIRCLE", threshold=0.7)
        effect = SetEffect("motor", "GRASP")
        prod = Production("test", cond, effect)
        
        strength = prod.evaluate(context)
        
        assert strength > 0.9
        assert prod._strength > 0.9
        
    def test_production_fire(self):
        """Test firing a production."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("GRASP")
        
        motor = State("motor", 128, vocab)
        context = {'modules': {'motor': motor}, 'vocab': vocab}
        
        cond = MatchCondition("vision", "CIRCLE")  # Dummy condition
        effect = SetEffect("motor", "GRASP")
        prod = Production("test", cond, effect)
        
        prod.fire(context)
        
        assert np.allclose(motor.state, vocab["GRASP"].vector)
        assert prod._fired_count == 1
        
    def test_production_representation(self):
        """Test production string representation."""
        cond = MatchCondition("state", "pattern")
        effect = SetEffect("output", "value")
        prod = Production("test", cond, effect)
        
        repr_str = repr(prod)
        assert "test" in repr_str
        assert "strength=" in repr_str
        assert "fired=" in repr_str


class TestProductionSystem:
    """Test ProductionSystem class."""
    
    def test_production_system_creation(self):
        """Test creating production system."""
        system = ProductionSystem()
        
        assert len(system.productions) == 0
        assert system.conflict_resolution == "priority"
        assert len(system._fired_productions) == 0
        
    def test_add_production(self):
        """Test adding productions."""
        system = ProductionSystem()
        
        cond = MatchCondition("state", "pattern")
        effect = SetEffect("output", "value")
        prod = Production("test", cond, effect)
        
        system.add_production(prod)
        
        assert len(system.productions) == 1
        assert system.productions[0] is prod
        
    def test_set_context(self):
        """Test setting execution context."""
        system = ProductionSystem()
        
        vocab = create_vocabulary(128)
        modules = {'state': State("state", 128, vocab)}
        
        system.set_context(modules, vocab, custom_var="test")
        
        assert system._context['modules'] is modules
        assert system._context['vocab'] is vocab
        assert system._context['custom_var'] == "test"
        
    def test_evaluate_all(self):
        """Test evaluating all productions."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        
        state = State("state", 128, vocab)
        state.set_semantic_pointer(vocab["A"])
        
        # Create productions
        cond1 = MatchCondition("state", "A", threshold=0.7)
        cond2 = MatchCondition("state", "B", threshold=0.7)
        
        prod1 = Production("match_a", cond1, SetEffect("output", "X"))
        prod2 = Production("match_b", cond2, SetEffect("output", "Y"))
        
        system = ProductionSystem([prod1, prod2])
        system.set_context({'state': state}, vocab)
        
        active = system.evaluate_all()
        
        assert len(active) == 1  # Only match_a
        assert active[0][0] is prod1
        assert active[0][1] > 0.9
        
    def test_conflict_resolution_priority(self):
        """Test priority-based conflict resolution."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("A")
        
        state = State("state", 128, vocab)
        output = State("output", 128, vocab)
        state.set_semantic_pointer(vocab["A"])
        
        # Two matching productions with different priorities
        cond = MatchCondition("state", "A", threshold=0.5)
        
        prod1 = Production("low_priority", cond, SetEffect("output", "X"), priority=1.0)
        prod2 = Production("high_priority", cond, SetEffect("output", "Y"), priority=2.0)
        
        system = ProductionSystem([prod1, prod2], conflict_resolution="priority")
        system.set_context({'state': state, 'output': output}, vocab)
        
        selected = system.select_production()
        
        assert selected is prod2  # Higher priority
        
    def test_conflict_resolution_specificity(self):
        """Test specificity-based conflict resolution."""
        vocab = create_vocabulary(128)
        pattern = vocab.create_pointer("PATTERN")
        
        state = State("state", 128, vocab)
        # Set state to have controlled similarity
        noise = np.random.randn(128)
        noise = noise - np.dot(noise, pattern.vector) * pattern.vector
        noise = noise / np.linalg.norm(noise)
        
        # Create state with 0.8 similarity
        state.state = 0.8 * pattern.vector + 0.6 * noise
        state.state = state.state / np.linalg.norm(state.state)
        
        # Different thresholds give different specificities
        cond1 = MatchCondition("state", pattern, threshold=0.5)  # Less specific
        cond2 = MatchCondition("state", pattern, threshold=0.7)  # More specific
        
        prod1 = Production("general", cond1, SetEffect("output", "X"))
        prod2 = Production("specific", cond2, SetEffect("output", "Y"))
        
        system = ProductionSystem([prod1, prod2], conflict_resolution="specificity")
        system.set_context({'state': state}, vocab)
        
        # Evaluate to get strengths
        active = system.evaluate_all()
        
        # Both should match
        assert len(active) == 2
        
        # Specificity resolution selects highest strength
        # With scaling: general = (0.8-0.5)/(1-0.5) = 0.6
        #              specific = (0.8-0.7)/(1-0.7) = 0.33
        # So general should win
        selected = system.select_production()
        assert selected is prod1
        
    def test_step_execution(self):
        """Test single step execution."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("TRIGGER")
        vocab.create_pointer("RESULT")
        
        trigger = State("trigger", 128, vocab)
        output = State("output", 128, vocab)
        
        trigger.set_semantic_pointer(vocab["TRIGGER"])
        
        cond = MatchCondition("trigger", "TRIGGER", threshold=0.7)
        effect = SetEffect("output", "RESULT")
        prod = Production("rule", cond, effect)
        
        system = ProductionSystem([prod])
        system.set_context({'trigger': trigger, 'output': output}, vocab)
        
        fired = system.step()
        
        assert fired is True
        assert np.allclose(output.state, vocab["RESULT"].vector)
        assert system._fired_productions == ["rule"]
        
    def test_step_no_match(self):
        """Test step with no matching productions."""
        vocab = create_vocabulary(128)
        state = State("state", 128, vocab)
        
        cond = MatchCondition("state", "PATTERN", threshold=0.9)
        prod = Production("rule", cond, SetEffect("output", "X"))
        
        system = ProductionSystem([prod])
        system.set_context({'state': state}, vocab)
        
        fired = system.step()
        
        assert fired is False
        assert len(system._fired_productions) == 0
        
    def test_run_multiple_cycles(self):
        """Test running multiple production cycles."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        vocab.create_pointer("C")
        
        state = State("state", 128, vocab)
        state.set_semantic_pointer(vocab["A"])
        
        # Chain of productions
        prod1 = Production(
            "a_to_b",
            MatchCondition("state", "A", threshold=0.7),
            SetEffect("state", "B")
        )
        prod2 = Production(
            "b_to_c",
            MatchCondition("state", "B", threshold=0.7),
            SetEffect("state", "C")
        )
        
        system = ProductionSystem([prod1, prod2])
        system.set_context({'state': state}, vocab)
        
        cycles = system.run(max_cycles=10)
        
        assert cycles == 2  # A->B, B->C, no match
        assert np.allclose(state.state, vocab["C"].vector)
        assert system._fired_productions == ["a_to_b", "b_to_c"]
        
    def test_reset(self):
        """Test resetting production system."""
        cond = MatchCondition("state", "pattern")
        prod = Production("test", cond, SetEffect("output", "value"))
        prod._strength = 0.8
        prod._fired_count = 5
        
        system = ProductionSystem([prod])
        system._fired_productions = ["test", "test"]
        
        system.reset()
        
        assert prod._strength == 0.0
        assert prod._fired_count == 0
        assert len(system._fired_productions) == 0


class TestConditionalModule:
    """Test ConditionalModule class."""
    
    def test_conditional_module_creation(self):
        """Test creating conditional module."""
        base = State("base", 128)
        cond = MatchCondition("control", "ENABLE", threshold=0.7)
        
        conditional = ConditionalModule("conditional", 128, cond, base)
        
        assert conditional.name == "conditional"
        assert conditional.dimensions == 128
        assert conditional.condition is cond
        assert conditional.base_module is base
        assert not conditional._gated
        
    def test_conditional_update_gated_open(self):
        """Test update when condition is met."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("INPUT")
        pattern = vocab.create_pointer("PATTERN")
        
        base = State("base", 128, vocab)
        # Create a condition that will match
        cond = MatchCondition("conditional", pattern, threshold=0.7)
        
        conditional = ConditionalModule("conditional", 128, cond, base, vocab)
        # Set state to match the pattern
        conditional._state = pattern.vector.copy()
        
        # Set input to base
        base._input = vocab["INPUT"].vector
        
        conditional.update(0.01)
        
        assert conditional.is_gated
        assert np.allclose(conditional.state, base.state)
        
    def test_conditional_update_gated_closed(self):
        """Test update when condition is not met."""
        base = State("base", 128)
        cond = MatchCondition("conditional", "MISSING", threshold=0.9)
        
        conditional = ConditionalModule("conditional", 128, cond, base)
        
        # Set some initial state
        initial_state = np.random.randn(128)
        conditional._state = initial_state.copy()
        
        conditional.update(0.01)
        
        assert not conditional.is_gated
        assert np.array_equal(conditional.state, initial_state)  # Unchanged


class TestProductionParsing:
    """Test production rule parsing."""
    
    def test_parse_simple_rules(self):
        """Test parsing simple production rules."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("CIRCLE")
        vocab.create_pointer("GRASP")
        vocab.create_pointer("SQUARE")
        vocab.create_pointer("RELEASE")
        
        rules_text = """
        IF vision MATCHES CIRCLE THEN SET motor TO GRASP
        IF vision MATCHES SQUARE THEN SET motor TO RELEASE
        """
        
        productions = parse_production_rules(rules_text, vocab)
        
        assert len(productions) == 2
        
        # Check first production
        prod1 = productions[0]
        assert prod1.name == "rule_1"
        assert isinstance(prod1.condition, MatchCondition)
        assert prod1.condition.module_name == "vision"
        assert prod1.condition.pattern == "CIRCLE"
        assert isinstance(prod1.effect, SetEffect)
        assert prod1.effect.module_name == "motor"
        assert prod1.effect.value == "GRASP"
        
    def test_parse_with_comments(self):
        """Test parsing with comments and blank lines."""
        rules_text = """
        # This is a comment
        IF state MATCHES A THEN SET output TO X
        
        # Another comment
        IF state MATCHES B THEN SET output TO Y
        """
        
        productions = parse_production_rules(rules_text)
        
        assert len(productions) == 2
        
    def test_parse_empty_rules(self):
        """Test parsing empty rules."""
        productions = parse_production_rules("")
        assert len(productions) == 0
        
        productions = parse_production_rules("# Only comments")
        assert len(productions) == 0
        
    def test_parse_malformed_rules(self):
        """Test parsing malformed rules."""
        # Missing THEN
        rules_text = "IF state MATCHES pattern"
        productions = parse_production_rules(rules_text)
        assert len(productions) == 0
        
        # Missing MATCHES
        rules_text = "IF state pattern THEN SET output TO value"
        productions = parse_production_rules(rules_text)
        assert len(productions) == 0
        
        # Missing TO in effect
        rules_text = "IF state MATCHES pattern THEN SET output value"
        productions = parse_production_rules(rules_text)
        assert len(productions) == 0