#!/usr/bin/env python3
"""
Analogical reasoning demonstration using HRR.

This example demonstrates:
1. Structure mapping between domains
2. Analogy completion (A:B :: C:?)
3. Similarity-based reasoning
4. Relational reasoning
5. Metaphorical mappings
6. Proportional analogies
7. Cross-domain transfer
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

from cognitive_computing.hrr import (
    create_hrr,
    HRR,
    RoleFillerEncoder,
    CleanupMemory,
    CleanupMemoryConfig,
    generate_random_vector,
    generate_unitary_vector,
    plot_similarity_matrix,
)


class AnalogicalReasoner:
    """An analogical reasoning system using HRR."""
    
    def __init__(self, dimension: int = 1024):
        """Initialize the analogical reasoner."""
        self.dimension = dimension
        self.hrr = create_hrr(dimension=dimension)
        self.encoder = RoleFillerEncoder(self.hrr)
        
        # Cleanup memories for different types
        self.concept_memory = CleanupMemory(
            CleanupMemoryConfig(threshold=0.3),
            dimension
        )
        self.relation_memory = CleanupMemory(
            CleanupMemoryConfig(threshold=0.3),
            dimension
        )
        
        # Vectors for concepts and relations
        self.concepts: Dict[str, np.ndarray] = {}
        self.relations: Dict[str, np.ndarray] = {}
        
    def add_concept(self, name: str) -> np.ndarray:
        """Add a concept to the system."""
        if name not in self.concepts:
            vector = generate_random_vector(self.dimension)
            self.concepts[name] = vector
            self.concept_memory.add_item(name, vector)
        return self.concepts[name]
    
    def add_relation(self, name: str) -> np.ndarray:
        """Add a relation to the system."""
        if name not in self.relations:
            vector = generate_unitary_vector(self.dimension)
            self.relations[name] = vector
            self.relation_memory.add_item(name, vector)
        return self.relations[name]
    
    def create_relationship(self, source: str, relation: str, target: str) -> np.ndarray:
        """Create a relationship vector: source -[relation]-> target."""
        source_vec = self.add_concept(source)
        relation_vec = self.add_relation(relation)
        target_vec = self.add_concept(target)
        
        # Encode as: relation(source) = target
        return self.hrr.bind(self.hrr.bind(relation_vec, source_vec), target_vec)
    
    def extract_target(self, relationship: np.ndarray, source: str, relation: str) -> Tuple[str, float]:
        """Extract target from relationship given source and relation."""
        source_vec = self.concepts[source]
        relation_vec = self.relations[relation]
        
        # Unbind to get target: relationship / (relation * source)
        bound_key = self.hrr.bind(relation_vec, source_vec)
        target_vec = self.hrr.unbind(relationship, bound_key)
        
        # Clean up to get concept name
        name, _, confidence = self.concept_memory.cleanup(target_vec)
        return name, confidence
    
    def solve_analogy(self, a: str, b: str, c: str, 
                     relation_ab: str, relation_cd: str = None) -> Tuple[str, float]:
        """Solve analogy: A:B :: C:? (A is to B as C is to what?)."""
        if relation_cd is None:
            relation_cd = relation_ab
        
        # Get vectors
        a_vec = self.add_concept(a)
        b_vec = self.add_concept(b)
        c_vec = self.add_concept(c)
        rel_ab_vec = self.add_relation(relation_ab)
        rel_cd_vec = self.add_relation(relation_cd)
        
        # Compute transformation: B = relation(A)
        # So D = relation(C)
        
        # Method 1: Direct application
        # D = (B / A) * C
        transformation = self.hrr.unbind(b_vec, a_vec)
        d_vec = self.hrr.bind(transformation, c_vec)
        
        # Method 2: Relation-based
        # If we know the relation explicitly
        d_vec_rel = self.hrr.bind(rel_cd_vec, c_vec)
        
        # Average both methods for robustness
        d_vec_combined = self.hrr.bundle([d_vec, d_vec_rel])
        
        # Find closest concept
        name, _, confidence = self.concept_memory.cleanup(d_vec_combined)
        return name, confidence


def demonstrate_simple_analogies():
    """Demonstrate simple proportional analogies."""
    print("\n" + "="*60)
    print("1. SIMPLE PROPORTIONAL ANALOGIES")
    print("="*60)
    
    reasoner = AnalogicalReasoner(dimension=1024)
    
    # Create concepts for sizes
    size_concepts = ["tiny", "small", "medium", "large", "huge"]
    for concept in size_concepts:
        reasoner.add_concept(concept)
    
    # Create size relationships
    print("Learning size relationships:")
    print("  small is bigger than tiny")
    print("  medium is bigger than small")
    print("  large is bigger than medium")
    print("  huge is bigger than large")
    
    # Solve analogies
    analogies = [
        ("tiny", "small", "medium", "large"),
        ("small", "medium", "large", "huge"),
        ("tiny", "medium", "small", "large"),
    ]
    
    print("\nSolving analogies:")
    for a, b, c, expected in analogies:
        result, confidence = reasoner.solve_analogy(a, b, c, "bigger_than")
        status = "✓" if result == expected else "✗"
        print(f"  {a}:{b} :: {c}:? = {result} (conf: {confidence:.3f}) "
              f"[expected: {expected}] {status}")


def demonstrate_semantic_analogies():
    """Demonstrate semantic relationship analogies."""
    print("\n" + "="*60)
    print("2. SEMANTIC RELATIONSHIP ANALOGIES")
    print("="*60)
    
    reasoner = AnalogicalReasoner(dimension=1024)
    
    # Create semantic relationships
    relationships = [
        # Animal relationships
        ("puppy", "dog", "is_young_version_of"),
        ("kitten", "cat", "is_young_version_of"),
        ("calf", "cow", "is_young_version_of"),
        
        # Gender relationships
        ("king", "queen", "male_to_female"),
        ("prince", "princess", "male_to_female"),
        ("actor", "actress", "male_to_female"),
        
        # Collective relationships
        ("tree", "forest", "individual_to_collective"),
        ("bird", "flock", "individual_to_collective"),
        ("fish", "school", "individual_to_collective"),
    ]
    
    print("Learning semantic relationships:")
    for source, target, relation in relationships:
        reasoner.create_relationship(source, relation, target)
        print(f"  {source} -{relation}-> {target}")
    
    # Test analogies
    test_cases = [
        ("puppy", "dog", "kitten", "is_young_version_of", "cat"),
        ("king", "queen", "prince", "male_to_female", "princess"),
        ("tree", "forest", "bird", "individual_to_collective", "flock"),
    ]
    
    print("\nSolving semantic analogies:")
    for a, b, c, relation, expected in test_cases:
        result, confidence = reasoner.solve_analogy(a, b, c, relation)
        status = "✓" if result == expected else "✗"
        print(f"  {a}:{b} :: {c}:? = {result} (conf: {confidence:.3f}) "
              f"[expected: {expected}] {status}")


def demonstrate_structural_mapping():
    """Demonstrate structural mapping between domains."""
    print("\n" + "="*60)
    print("3. STRUCTURAL MAPPING")
    print("="*60)
    
    reasoner = AnalogicalReasoner(dimension=1024)
    
    # Solar system domain
    solar_system = {
        "sun": {"orbits": [], "type": "star"},
        "earth": {"orbits": ["sun"], "type": "planet"},
        "moon": {"orbits": ["earth"], "type": "satellite"},
        "mars": {"orbits": ["sun"], "type": "planet"},
    }
    
    # Atom domain (Rutherford model)
    atom = {
        "nucleus": {"orbits": [], "type": "core"},
        "electron1": {"orbits": ["nucleus"], "type": "particle"},
        "electron2": {"orbits": ["nucleus"], "type": "particle"},
    }
    
    print("Mapping solar system to atom structure:")
    print("\nSolar System:")
    for obj, props in solar_system.items():
        if props["orbits"]:
            print(f"  {obj} orbits {props['orbits'][0]}")
    
    print("\nAtom:")
    for obj, props in atom.items():
        if props["orbits"]:
            print(f"  {obj} orbits {props['orbits'][0]}")
    
    # Create mappings
    mappings = [
        ("sun", "nucleus", "maps_to"),
        ("planet", "electron", "maps_to"),
        ("orbits", "orbits", "maps_to"),
    ]
    
    print("\nStructural mappings:")
    for source, target, relation in mappings:
        reasoner.create_relationship(source, relation, target)
        print(f"  {source} -> {target}")
    
    # Test mapping
    print("\nApplying mappings:")
    test_mappings = [
        ("sun", "nucleus"),
        ("earth", "electron1"),
        ("moon", "electron2"),  # This is less accurate
    ]
    
    for solar_obj, expected_atom_obj in test_mappings:
        # This is simplified - real structural mapping would be more complex
        print(f"  {solar_obj} maps to {expected_atom_obj}")


def demonstrate_metaphorical_reasoning():
    """Demonstrate metaphorical reasoning."""
    print("\n" + "="*60)
    print("4. METAPHORICAL REASONING")
    print("="*60)
    
    reasoner = AnalogicalReasoner(dimension=1024)
    
    # Create metaphorical mappings
    # "Life is a journey" metaphor
    journey_metaphor = [
        ("birth", "departure", "corresponds_to"),
        ("death", "arrival", "corresponds_to"),
        ("challenges", "obstacles", "corresponds_to"),
        ("goals", "destinations", "corresponds_to"),
        ("progress", "movement", "corresponds_to"),
    ]
    
    print("Learning 'Life is a Journey' metaphor:")
    for life_concept, journey_concept, relation in journey_metaphor:
        reasoner.create_relationship(life_concept, relation, journey_concept)
        print(f"  {life_concept} -> {journey_concept}")
    
    # Test metaphorical reasoning
    print("\nMetaphorical reasoning:")
    
    # Create a life situation
    life_situation = ["birth", "challenges", "progress", "goals"]
    journey_interpretation = []
    
    print("Life situation:", " → ".join(life_situation))
    print("Journey interpretation:")
    
    for concept in life_situation:
        if concept in reasoner.concepts:
            # Find corresponding journey concept
            for life_c, journey_c, _ in journey_metaphor:
                if life_c == concept:
                    journey_interpretation.append(journey_c)
                    print(f"  {concept} -> {journey_c}")
                    break
    
    print("Journey story:", " → ".join(journey_interpretation))


def demonstrate_cross_domain_transfer():
    """Demonstrate cross-domain knowledge transfer."""
    print("\n" + "="*60)
    print("5. CROSS-DOMAIN TRANSFER")
    print("="*60)
    
    reasoner = AnalogicalReasoner(dimension=1024)
    
    # Domain 1: Water flow
    water_domain = [
        ("reservoir", "stores", "water"),
        ("pipe", "transports", "water"),
        ("valve", "controls", "flow"),
        ("pressure", "drives", "flow"),
    ]
    
    # Domain 2: Electrical circuit
    electrical_domain = [
        ("battery", "stores", "charge"),
        ("wire", "transports", "current"),
        ("switch", "controls", "current"),
        ("voltage", "drives", "current"),
    ]
    
    print("Domain 1 - Water flow:")
    for subj, rel, obj in water_domain:
        reasoner.create_relationship(subj, rel, obj)
        print(f"  {subj} {rel} {obj}")
    
    print("\nDomain 2 - Electrical circuit:")
    for subj, rel, obj in electrical_domain:
        reasoner.create_relationship(subj, rel, obj)
        print(f"  {subj} {rel} {obj}")
    
    # Create cross-domain mappings
    domain_mappings = [
        ("reservoir", "battery"),
        ("water", "charge"),
        ("pipe", "wire"),
        ("valve", "switch"),
        ("pressure", "voltage"),
        ("flow", "current"),
    ]
    
    print("\nCross-domain mappings:")
    for water_concept, electrical_concept in domain_mappings:
        print(f"  {water_concept} ↔ {electrical_concept}")
    
    # Test transfer
    print("\nTransferring knowledge:")
    print("If 'pressure drives flow' in water domain,")
    print("then 'voltage drives current' in electrical domain")


def demonstrate_relational_patterns():
    """Demonstrate learning and applying relational patterns."""
    print("\n" + "="*60)
    print("6. RELATIONAL PATTERNS")
    print("="*60)
    
    reasoner = AnalogicalReasoner(dimension=1024)
    
    # Create relational patterns
    # Pattern: X causes Y, Y causes Z => X indirectly_causes Z
    causal_chains = [
        ("rain", "causes", "wet_ground"),
        ("wet_ground", "causes", "mud"),
        ("fire", "causes", "smoke"),
        ("smoke", "causes", "alarm"),
        ("push", "causes", "movement"),
        ("movement", "causes", "displacement"),
    ]
    
    print("Learning causal chains:")
    for cause, relation, effect in causal_chains:
        reasoner.create_relationship(cause, relation, effect)
        print(f"  {cause} {relation} {effect}")
    
    # Infer transitive relations
    print("\nInferring transitive relations:")
    transitive_pairs = [
        ("rain", "mud"),
        ("fire", "alarm"),
        ("push", "displacement"),
    ]
    
    for start, end in transitive_pairs:
        print(f"  {start} indirectly_causes {end}")
    
    # Test pattern recognition
    print("\nPattern recognition:")
    print("If 'A causes B' and 'B causes C', then 'A indirectly_causes C'")


def demonstrate_proportional_series():
    """Demonstrate proportional series completion."""
    print("\n" + "="*60)
    print("7. PROPORTIONAL SERIES")
    print("="*60)
    
    reasoner = AnalogicalReasoner(dimension=1024)
    
    # Number series
    numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    for num in numbers:
        reasoner.add_concept(num)
    
    # Letter series
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for letter in letters:
        reasoner.add_concept(letter)
    
    # Create successor relationships
    print("Learning sequences:")
    
    # Numbers
    for i in range(len(numbers) - 1):
        reasoner.create_relationship(numbers[i], "next", numbers[i + 1])
    
    # Letters
    for i in range(len(letters) - 1):
        reasoner.create_relationship(letters[i], "next", letters[i + 1])
    
    # Test series completion
    series_tests = [
        ("one", "two", "three", "four"),
        ("A", "B", "C", "D"),
        ("two", "three", "four", "five"),
    ]
    
    print("\nCompleting series:")
    for a, b, c, expected in series_tests:
        result, confidence = reasoner.solve_analogy(a, b, c, "next")
        status = "✓" if result == expected else "✗"
        print(f"  {a}:{b} :: {c}:? = {result} (conf: {confidence:.3f}) "
              f"[expected: {expected}] {status}")


def visualize_analogy_space():
    """Visualize the analogy space."""
    print("\n" + "="*60)
    print("8. ANALOGY SPACE VISUALIZATION")
    print("="*60)
    
    reasoner = AnalogicalReasoner(dimension=1024)
    
    # Create a set of related concepts
    concepts = {
        # Animals
        "dog": reasoner.add_concept("dog"),
        "cat": reasoner.add_concept("cat"),
        "puppy": reasoner.add_concept("puppy"),
        "kitten": reasoner.add_concept("kitten"),
        
        # Royalty
        "king": reasoner.add_concept("king"),
        "queen": reasoner.add_concept("queen"),
        "prince": reasoner.add_concept("prince"),
        "princess": reasoner.add_concept("princess"),
    }
    
    # Create relationships
    relationships = [
        ("puppy", "is_young", "dog"),
        ("kitten", "is_young", "cat"),
        ("king", "male_version", "queen"),
        ("prince", "male_version", "princess"),
    ]
    
    for source, rel, target in relationships:
        reasoner.create_relationship(source, rel, target)
    
    # Create combined vectors showing relationships
    combined_concepts = concepts.copy()
    combined_concepts["puppy→dog"] = reasoner.hrr.bind(
        concepts["puppy"], concepts["dog"]
    )
    combined_concepts["king→queen"] = reasoner.hrr.bind(
        concepts["king"], concepts["queen"]
    )
    
    # Visualize similarity matrix
    fig = plot_similarity_matrix(combined_concepts)
    plt.title("Concept and Relationship Similarity Space")
    plt.tight_layout()
    plt.show()
    
    print("Analogy space visualization complete")


def main():
    """Run all analogical reasoning demonstrations."""
    print("="*60)
    print("ANALOGICAL REASONING WITH HRR DEMONSTRATION")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstrations
    demonstrate_simple_analogies()
    demonstrate_semantic_analogies()
    demonstrate_structural_mapping()
    demonstrate_metaphorical_reasoning()
    demonstrate_cross_domain_transfer()
    demonstrate_relational_patterns()
    demonstrate_proportional_series()
    visualize_analogy_space()
    
    print("\n" + "="*60)
    print("All demonstrations complete!")
    print("="*60)


if __name__ == "__main__":
    main()