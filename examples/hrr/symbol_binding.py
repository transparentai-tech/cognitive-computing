#!/usr/bin/env python3
"""
Symbol binding demonstration using HRR.

This example demonstrates:
1. Role-filler binding for structured representations
2. Variable binding and substitution
3. Compositional structures
4. Complex symbolic reasoning
5. Binding multiple roles simultaneously
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

from cognitive_computing.hrr import (
    create_hrr,
    HRR,
    RoleFillerEncoder,
    CleanupMemory,
    CleanupMemoryConfig,
    generate_random_vector,
    generate_unitary_vector,
    generate_orthogonal_set,
    plot_similarity_matrix,
)


class SymbolicReasoner:
    """A symbolic reasoning system using HRR."""
    
    def __init__(self, dimension: int = 1024):
        """Initialize the symbolic reasoner."""
        self.dimension = dimension
        self.hrr = create_hrr(dimension=dimension)
        self.encoder = RoleFillerEncoder(self.hrr)
        
        # Symbol registries
        self.symbols: Dict[str, np.ndarray] = {}
        self.roles: Dict[str, np.ndarray] = {}
        self.cleanup = CleanupMemory(CleanupMemoryConfig(threshold=0.3), dimension)
        
    def create_symbol(self, name: str) -> np.ndarray:
        """Create a new symbol vector."""
        if name not in self.symbols:
            vector = generate_random_vector(self.dimension)
            self.symbols[name] = vector
            self.cleanup.add_item(name, vector)
        return self.symbols[name]
    
    def create_role(self, name: str) -> np.ndarray:
        """Create a new role vector (unitary for clean unbinding)."""
        if name not in self.roles:
            self.roles[name] = generate_unitary_vector(self.dimension)
        return self.roles[name]
    
    def bind_structure(self, structure: Dict[str, str]) -> np.ndarray:
        """Bind a role-filler structure."""
        role_filler_pairs = {}
        for role_name, filler_name in structure.items():
            role = self.create_role(role_name)
            filler = self.create_symbol(filler_name)
            role_filler_pairs[role_name] = (role, filler)
        
        # Create the composite representation
        return self.encoder.encode_structure({
            role_name: filler 
            for role_name, (role, filler) in role_filler_pairs.items()
        })
    
    def query_role(self, structure: np.ndarray, role_name: str) -> Tuple[str, float]:
        """Query a role in a structure."""
        role = self.roles.get(role_name)
        if role is None:
            raise ValueError(f"Unknown role: {role_name}")
        
        # Unbind the role to get the filler
        filler = self.encoder.decode_filler(structure, role)
        
        # Clean up to get the symbol name
        name, _, confidence = self.cleanup.cleanup(filler)
        return name, confidence


def demonstrate_basic_role_filler():
    """Demonstrate basic role-filler binding."""
    print("\n" + "="*60)
    print("1. BASIC ROLE-FILLER BINDING")
    print("="*60)
    
    reasoner = SymbolicReasoner(dimension=1024)
    
    # Create a simple structure: "John loves Mary"
    structure = {
        "agent": "John",
        "action": "loves",
        "patient": "Mary"
    }
    
    # Bind the structure
    sentence = reasoner.bind_structure(structure)
    
    print("Original structure:")
    for role, filler in structure.items():
        print(f"  {role}: {filler}")
    
    print("\nQuerying the structure:")
    for role in structure.keys():
        retrieved, confidence = reasoner.query_role(sentence, role)
        print(f"  {role}: {retrieved} (confidence: {confidence:.3f})")


def demonstrate_variable_binding():
    """Demonstrate variable binding and substitution."""
    print("\n" + "="*60)
    print("2. VARIABLE BINDING AND SUBSTITUTION")
    print("="*60)
    
    reasoner = SymbolicReasoner(dimension=1024)
    
    # Create a template: "X loves Y"
    print("Creating template: X loves Y")
    
    # Create variable placeholders
    X = reasoner.create_symbol("X")
    Y = reasoner.create_symbol("Y")
    loves = reasoner.create_symbol("loves")
    
    # Create roles
    agent_role = reasoner.create_role("agent")
    action_role = reasoner.create_role("action")
    patient_role = reasoner.create_role("patient")
    
    # Create template
    template = reasoner.hrr.bundle([
        reasoner.hrr.bind(agent_role, X),
        reasoner.hrr.bind(action_role, loves),
        reasoner.hrr.bind(patient_role, Y)
    ])
    
    # Create substitutions
    substitutions = [
        {"X": "John", "Y": "Mary"},
        {"X": "Alice", "Y": "Bob"},
        {"X": "Dog", "Y": "Cat"}
    ]
    
    print("\nApplying substitutions:")
    
    for subs in substitutions:
        # Create substitution mapping
        X_sub = reasoner.create_symbol(subs["X"])
        Y_sub = reasoner.create_symbol(subs["Y"])
        
        # Perform substitution by unbinding variables and binding new values
        # This is a simplified approach - in practice you'd use more sophisticated methods
        instance = reasoner.hrr.bundle([
            reasoner.hrr.bind(agent_role, X_sub),
            reasoner.hrr.bind(action_role, loves),
            reasoner.hrr.bind(patient_role, Y_sub)
        ])
        
        # Query the instantiated structure
        agent, _ = reasoner.query_role(instance, "agent")
        patient, _ = reasoner.query_role(instance, "patient")
        action, _ = reasoner.query_role(instance, "action")
        
        print(f"  {agent} {action} {patient}")


def demonstrate_compositional_structures():
    """Demonstrate compositional structures."""
    print("\n" + "="*60)
    print("3. COMPOSITIONAL STRUCTURES")
    print("="*60)
    
    reasoner = SymbolicReasoner(dimension=1024)
    
    # Create nested structure: "John believes [Mary loves Bob]"
    # First create the embedded proposition
    embedded_prop = reasoner.bind_structure({
        "agent": "Mary",
        "action": "loves",
        "patient": "Bob"
    })
    
    # Create a symbol for the embedded proposition
    reasoner.symbols["mary_loves_bob"] = embedded_prop
    reasoner.cleanup.add_item("mary_loves_bob", embedded_prop)
    
    # Create the main proposition
    main_prop = reasoner.bind_structure({
        "agent": "John",
        "action": "believes",
        "proposition": "mary_loves_bob"
    })
    
    print("Nested structure: John believes [Mary loves Bob]")
    print("\nQuerying main proposition:")
    
    for role in ["agent", "action", "proposition"]:
        retrieved, confidence = reasoner.query_role(main_prop, role)
        print(f"  {role}: {retrieved} (confidence: {confidence:.3f})")
    
    # Extract and query the embedded proposition
    print("\nExtracting embedded proposition:")
    prop_role = reasoner.roles["proposition"]
    embedded_retrieved = reasoner.hrr.unbind(main_prop, prop_role)
    
    # Query the embedded structure
    # Note: This requires the embedded structure to be properly stored
    print("Embedded proposition structure:")
    print("  (Would need additional machinery to fully decompose)")


def demonstrate_multiple_bindings():
    """Demonstrate binding multiple role-filler pairs."""
    print("\n" + "="*60)
    print("4. MULTIPLE SIMULTANEOUS BINDINGS")
    print("="*60)
    
    reasoner = SymbolicReasoner(dimension=1024)
    
    # Create a complex structure with many roles
    person_description = {
        "name": "Alice",
        "age": "thirty",
        "occupation": "scientist",
        "location": "Boston",
        "hobby": "painting",
        "pet": "cat"
    }
    
    # Bind the structure
    alice = reasoner.bind_structure(person_description)
    
    print("Complex structure with multiple roles:")
    for role, filler in person_description.items():
        print(f"  {role}: {filler}")
    
    print("\nQuerying all roles:")
    correct_retrievals = 0
    
    for role, expected in person_description.items():
        retrieved, confidence = reasoner.query_role(alice, role)
        is_correct = retrieved == expected
        correct_retrievals += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"  {role}: {retrieved} (confidence: {confidence:.3f}) {status}")
    
    accuracy = correct_retrievals / len(person_description) * 100
    print(f"\nRetrieval accuracy: {accuracy:.1f}%")


def demonstrate_symbolic_arithmetic():
    """Demonstrate symbolic arithmetic operations."""
    print("\n" + "="*60)
    print("5. SYMBOLIC ARITHMETIC")
    print("="*60)
    
    dimension = 1024
    hrr = create_hrr(dimension=dimension)
    
    # Create number symbols
    numbers = {}
    for i in range(10):
        numbers[str(i)] = generate_random_vector(dimension)
    
    # Create operation symbols
    plus = generate_unitary_vector(dimension)
    equals = generate_unitary_vector(dimension)
    
    # Encode arithmetic facts: "2 + 3 = 5"
    facts = []
    
    # Simple addition facts
    addition_facts = [
        (2, 3, 5),
        (1, 1, 2),
        (4, 2, 6),
        (3, 4, 7)
    ]
    
    print("Encoding arithmetic facts:")
    for a, b, c in addition_facts:
        # Encode: a + b = c
        fact = hrr.bundle([
            hrr.bind(numbers[str(a)], plus),
            hrr.bind(plus, numbers[str(b)]),
            hrr.bind(equals, numbers[str(c)])
        ])
        facts.append(fact)
        print(f"  {a} + {b} = {c}")
    
    # Bundle all facts into knowledge base
    knowledge = hrr.bundle(facts)
    
    # Query the knowledge base
    print("\nQuerying arithmetic knowledge:")
    
    # Query: What is 2 + 3?
    query = hrr.bundle([
        hrr.bind(numbers["2"], plus),
        hrr.bind(plus, numbers["3"]),
        equals  # Looking for what equals binds to
    ])
    
    # Find the answer by checking similarity with all numbers
    print("\nQuery: 2 + 3 = ?")
    similarities = {}
    
    for num, vector in numbers.items():
        bound_with_equals = hrr.bind(equals, vector)
        sim = hrr.similarity(knowledge, bound_with_equals)
        similarities[num] = sim
    
    # Find the most similar
    answer = max(similarities.items(), key=lambda x: x[1])
    print(f"Answer: {answer[0]} (similarity: {answer[1]:.3f})")


def demonstrate_role_relationships():
    """Demonstrate relationships between roles."""
    print("\n" + "="*60)
    print("6. ROLE RELATIONSHIPS AND CONSTRAINTS")
    print("="*60)
    
    reasoner = SymbolicReasoner(dimension=1024)
    
    # Create related structures
    structures = []
    
    # Family relationships
    family_relations = [
        {"parent": "John", "child": "Alice"},
        {"parent": "John", "child": "Bob"},
        {"parent": "Mary", "child": "Alice"},
        {"parent": "Mary", "child": "Bob"},
    ]
    
    print("Encoding family relationships:")
    for relation in family_relations:
        structure = reasoner.bind_structure(relation)
        structures.append(structure)
        print(f"  {relation['parent']} is parent of {relation['child']}")
    
    # Bundle all relationships
    family_knowledge = reasoner.hrr.bundle(structures)
    
    # Query: Who are John's children?
    print("\nQuerying: Who are John's children?")
    
    john = reasoner.symbols["John"]
    parent_role = reasoner.roles["parent"]
    child_role = reasoner.roles["child"]
    
    # Create a partial structure with John as parent
    john_as_parent = reasoner.hrr.bind(parent_role, john)
    
    # Check all known people to see who could be the child
    people = ["Alice", "Bob", "Charlie", "David"]
    
    for person in people:
        if person in reasoner.symbols:
            person_vec = reasoner.symbols[person]
        else:
            person_vec = reasoner.create_symbol(person)
        
        # Create a structure with this person as child
        test_structure = reasoner.hrr.bundle([
            john_as_parent,
            reasoner.hrr.bind(child_role, person_vec)
        ])
        
        # Check similarity with knowledge base
        similarity = reasoner.hrr.similarity(family_knowledge, test_structure)
        
        if similarity > 0.3:  # Threshold for considering it a match
            print(f"  {person} (similarity: {similarity:.3f})")


def visualize_symbol_space(reasoner: SymbolicReasoner):
    """Visualize the symbol space."""
    print("\n" + "="*60)
    print("7. SYMBOL SPACE VISUALIZATION")
    print("="*60)
    
    # Create various symbols and their relationships
    symbols = {}
    
    # Basic symbols
    for name in ["John", "Mary", "loves", "believes", "happy", "sad"]:
        symbols[name] = reasoner.create_symbol(name)
    
    # Create some composite symbols
    symbols["John_loves_Mary"] = reasoner.hrr.bind(
        reasoner.hrr.bind(reasoner.create_role("agent"), symbols["John"]),
        reasoner.hrr.bind(reasoner.create_role("patient"), symbols["Mary"])
    )
    
    symbols["Mary_is_happy"] = reasoner.hrr.bind(
        symbols["Mary"], symbols["happy"]
    )
    
    # Visualize similarity matrix
    fig = plot_similarity_matrix(symbols)
    plt.tight_layout()
    plt.show()
    
    print("Symbol space visualization complete")


def main():
    """Run all symbol binding demonstrations."""
    print("="*60)
    print("SYMBOL BINDING WITH HRR DEMONSTRATION")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstrations
    demonstrate_basic_role_filler()
    demonstrate_variable_binding()
    demonstrate_compositional_structures()
    demonstrate_multiple_bindings()
    demonstrate_symbolic_arithmetic()
    demonstrate_role_relationships()
    
    # Create reasoner for visualization
    reasoner = SymbolicReasoner(dimension=1024)
    visualize_symbol_space(reasoner)
    
    print("\n" + "="*60)
    print("All demonstrations complete!")
    print("="*60)


if __name__ == "__main__":
    main()