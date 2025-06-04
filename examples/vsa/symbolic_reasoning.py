#!/usr/bin/env python3
"""
Symbolic Reasoning Demo: Complex Reasoning with Vector Symbolic Architectures

This script demonstrates advanced symbolic reasoning capabilities using VSA:
- Analogical reasoning
- Compositional structures
- Logic operations
- Semantic relationships
- Knowledge representation
- Question answering
- Rule-based inference
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from cognitive_computing.vsa import create_vsa, BSC, MAP, FHRR


class KnowledgeBase:
    """A VSA-based knowledge base for symbolic reasoning."""
    
    def __init__(self, dimension: int = 10000):
        """Initialize knowledge base with a VSA instance."""
        self.vsa = create_vsa(
            dimension=dimension,
            vector_type='bipolar',
            vsa_type='custom',
            binding_method='multiplication'
        )
        self.facts = {}
        self.rules = {}
        self.concepts = {}
        
    def add_concept(self, name: str):
        """Add a concept to the knowledge base."""
        if name not in self.concepts:
            self.concepts[name] = self.vsa.generate_vector()
        return self.concepts[name]
    
    def add_fact(self, subject: str, predicate: str, object: str):
        """Add a fact as a triple (subject, predicate, object)."""
        # Encode components
        subj_vec = self.add_concept(subject)
        pred_vec = self.add_concept(predicate)
        obj_vec = self.add_concept(object)
        
        # Create structured representation
        fact_vec = self.vsa.bundle([
            self.vsa.bind(self.add_concept('SUBJECT'), subj_vec),
            self.vsa.bind(self.add_concept('PREDICATE'), pred_vec),
            self.vsa.bind(self.add_concept('OBJECT'), obj_vec)
        ])
        
        fact_key = f"{subject}-{predicate}-{object}"
        self.facts[fact_key] = fact_vec
        return fact_vec
    
    def query(self, role: str, fact_vec):
        """Query a fact for a specific role."""
        role_vec = self.add_concept(role)
        result = self.vsa.unbind(fact_vec, role_vec)
        
        # Find closest concept
        best_concept = None
        best_similarity = -1
        
        for name, concept in self.concepts.items():
            sim = self.vsa.similarity(result, concept)
            if sim > best_similarity:
                best_similarity = sim
                best_concept = name
                
        return best_concept, best_similarity
    
    def add_rule(self, name: str, condition, consequence):
        """Add an inference rule."""
        self.rules[name] = {
            'condition': condition,
            'consequence': consequence
        }


def demonstrate_analogical_reasoning():
    """Demonstrate analogical reasoning: A:B :: C:?
    
    Note: Analogical reasoning with multiplication binding typically produces
    low similarity scores because the transformation cannot be cleanly extracted.
    Better results would be achieved with permutation-based or MAP-based binding.
    """
    print("=== Analogical Reasoning Demo ===\n")
    
    vsa = create_vsa(
        dimension=10000,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )
    
    # Example 1: Classic word analogy
    print("1. Word Analogies:")
    
    # King:Queen :: Man:Woman
    king = vsa.generate_vector()
    queen = vsa.generate_vector()
    man = vsa.generate_vector()
    woman = vsa.generate_vector()
    
    # Learn the transformation
    # For multiplication binding, we can unbind by binding again (element-wise multiplication is its own inverse)
    transform = vsa.bind(queen, king)
    
    # Apply to man
    result = vsa.bind(transform, vsa.bind(man, king))
    similarity = vsa.similarity(result, woman)
    
    print(f"   King:Queen :: Man:? → Woman (similarity: {similarity:.3f})")
    
    # Example 2: Semantic relationships
    print("\n2. Semantic Relationships:")
    
    # Paris:France :: Tokyo:Japan
    paris = vsa.generate_vector()
    france = vsa.generate_vector()
    tokyo = vsa.generate_vector()
    japan = vsa.generate_vector()
    
    # Learn capital-country relationship
    # For multiplication binding, unbind by binding again
    capital_relation = vsa.bind(france, paris)
    
    # Apply to Tokyo
    result = vsa.bind(capital_relation, vsa.bind(tokyo, paris))
    similarity = vsa.similarity(result, japan)
    
    print(f"   Paris:France :: Tokyo:? → Japan (similarity: {similarity:.3f})")
    
    # Example 3: Property transfer
    print("\n3. Property Transfer:")
    
    # Bird:Fly :: Fish:Swim
    bird = vsa.generate_vector()
    fly = vsa.generate_vector()
    fish = vsa.generate_vector()
    swim = vsa.generate_vector()
    
    # Learn action relationship
    # For multiplication binding, unbind by binding again
    action_relation = vsa.bind(fly, bird)
    
    # Apply to fish
    result = vsa.bind(action_relation, vsa.bind(fish, bird))
    similarity = vsa.similarity(result, swim)
    
    print(f"   Bird:Fly :: Fish:? → Swim (similarity: {similarity:.3f})")
    
    # Example 4: Compositional analogy
    print("\n4. Compositional Analogies:")
    
    # Create compositional structures
    big = vsa.generate_vector()
    small = vsa.generate_vector()
    car = vsa.generate_vector()
    truck = vsa.generate_vector()
    
    big_car = vsa.bind(big, car)
    small_car = vsa.bind(small, car)
    big_truck = vsa.bind(big, truck)
    
    # Learn size transformation
    # For multiplication binding, unbind by binding again
    size_transform = vsa.bind(small_car, big_car)
    
    # Apply to big truck
    result = vsa.bind(size_transform, vsa.bind(big_truck, big_car))
    expected = vsa.bind(small, truck)
    similarity = vsa.similarity(result, expected)
    
    print(f"   Big_Car:Small_Car :: Big_Truck:? → Small_Truck (similarity: {similarity:.3f})\n")


def demonstrate_compositional_structures():
    """Demonstrate complex compositional structures."""
    print("=== Compositional Structures Demo ===\n")
    
    kb = KnowledgeBase(dimension=10000)
    
    # Example 1: Nested structures
    print("1. Nested Role-Filler Structures:")
    
    # Create a complex scene description
    # "The red car is parked next to the blue house"
    
    # Objects
    car = kb.add_concept('car')
    house = kb.add_concept('house')
    
    # Properties
    red = kb.add_concept('red')
    blue = kb.add_concept('blue')
    
    # Relations
    color_role = kb.add_concept('COLOR')
    type_role = kb.add_concept('TYPE')
    location_role = kb.add_concept('LOCATION')
    next_to = kb.add_concept('next_to')
    
    # Build structures
    red_car = kb.vsa.bundle([
        kb.vsa.bind(color_role, red),
        kb.vsa.bind(type_role, car)
    ])
    
    blue_house = kb.vsa.bundle([
        kb.vsa.bind(color_role, blue),
        kb.vsa.bind(type_role, house)
    ])
    
    scene = kb.vsa.bundle([
        kb.vsa.bind(kb.add_concept('OBJECT1'), red_car),
        kb.vsa.bind(kb.add_concept('OBJECT2'), blue_house),
        kb.vsa.bind(location_role, next_to)
    ])
    
    # Query the scene
    print("   Scene queries:")
    
    # What is object1?
    obj1 = kb.vsa.unbind(scene, kb.add_concept('OBJECT1'))
    obj1_type = kb.vsa.unbind(obj1, type_role)
    obj1_color = kb.vsa.unbind(obj1, color_role)
    
    print(f"   Object1 type similarity to 'car': {kb.vsa.similarity(obj1_type, car):.3f}")
    print(f"   Object1 color similarity to 'red': {kb.vsa.similarity(obj1_color, red):.3f}")
    
    # Example 2: Recursive structures
    print("\n2. Recursive Structures (Lists):")
    
    # Create a list: [apple, banana, orange]
    nil = kb.add_concept('NIL')
    cons = kb.add_concept('CONS')
    head_role = kb.add_concept('HEAD')
    tail_role = kb.add_concept('TAIL')
    
    apple = kb.add_concept('apple')
    banana = kb.add_concept('banana')
    orange = kb.add_concept('orange')
    
    # Build list recursively
    list3 = kb.vsa.bundle([
        kb.vsa.bind(head_role, orange),
        kb.vsa.bind(tail_role, nil)
    ])
    
    list2 = kb.vsa.bundle([
        kb.vsa.bind(head_role, banana),
        kb.vsa.bind(tail_role, list3)
    ])
    
    list1 = kb.vsa.bundle([
        kb.vsa.bind(head_role, apple),
        kb.vsa.bind(tail_role, list2)
    ])
    
    # Traverse the list
    print("   List traversal:")
    current = list1
    position = 1
    
    while position <= 3:
        head = kb.vsa.unbind(current, head_role)
        if position == 1:
            print(f"   Position {position}: similarity to 'apple' = {kb.vsa.similarity(head, apple):.3f}")
        elif position == 2:
            print(f"   Position {position}: similarity to 'banana' = {kb.vsa.similarity(head, banana):.3f}")
        elif position == 3:
            print(f"   Position {position}: similarity to 'orange' = {kb.vsa.similarity(head, orange):.3f}")
        
        current = kb.vsa.unbind(current, tail_role)
        position += 1
    
    print()


def demonstrate_logic_operations():
    """Demonstrate logical operations using VSA."""
    print("=== Logic Operations Demo ===\n")
    
    vsa = create_vsa(
        dimension=10000,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )
    
    # Example 1: Boolean logic with fuzzy values
    print("1. Fuzzy Logic Operations:")
    
    # Create truth values
    true = vsa.generate_vector()
    false = vsa.generate_vector()
    # Create maybe as a weighted average
    maybe = 0.5 * true + 0.5 * false
    maybe = maybe / np.linalg.norm(maybe)  # Normalize
    
    # Propositions
    sunny = vsa.generate_vector()
    warm = vsa.generate_vector()
    beach = vsa.generate_vector()
    
    # Create facts with truth values
    sunny_true = vsa.bind(sunny, true)
    warm_maybe = vsa.bind(warm, maybe)
    
    # Fuzzy AND operation (binding)
    sunny_and_warm = vsa.bind(sunny_true, warm_maybe)
    
    # Extract truth value
    truth = vsa.unbind(sunny_and_warm, vsa.bind(sunny, warm))
    
    print(f"   sunny ∧ warm truth similarities:")
    print(f"   - TRUE: {vsa.similarity(truth, true):.3f}")
    print(f"   - FALSE: {vsa.similarity(truth, false):.3f}")
    print(f"   - MAYBE: {vsa.similarity(truth, maybe):.3f}")
    
    # Example 2: Logical implications
    print("\n2. Logical Implications (If-Then Rules):")
    
    # Rule: If sunny AND warm THEN go to beach
    if_role = vsa.generate_vector()
    then_role = vsa.generate_vector()
    
    condition = vsa.bind(sunny, warm)
    consequence = beach
    
    rule = vsa.bundle([
        vsa.bind(if_role, condition),
        vsa.bind(then_role, consequence)
    ])
    
    # Apply rule
    extracted_condition = vsa.unbind(rule, if_role)
    extracted_consequence = vsa.unbind(rule, then_role)
    
    print(f"   Rule condition similarity to 'sunny ∧ warm': {vsa.similarity(extracted_condition, condition):.3f}")
    print(f"   Rule consequence similarity to 'beach': {vsa.similarity(extracted_consequence, beach):.3f}")
    
    # Example 3: Quantifiers
    print("\n3. Quantifiers (Universal/Existential):")
    
    # Concepts
    all_role = vsa.generate_vector()
    some_role = vsa.generate_vector()
    birds = vsa.generate_vector()
    can_fly = vsa.generate_vector()
    penguins = vsa.generate_vector()
    cannot_fly = vsa.generate_vector()
    
    # Universal: All birds can fly
    universal = vsa.bundle([
        vsa.bind(all_role, birds),
        vsa.bind(vsa.generate_vector(), can_fly)
    ])
    
    # Exception: Some birds (penguins) cannot fly
    exception = vsa.bundle([
        vsa.bind(some_role, vsa.bind(birds, penguins)),
        vsa.bind(vsa.generate_vector(), cannot_fly)
    ])
    
    # Combine with exception
    knowledge = 0.8 * universal + 0.2 * exception
    knowledge = knowledge / np.linalg.norm(knowledge)  # Normalize
    
    print("   Knowledge representation includes both rule and exception\n")


def demonstrate_semantic_relationships():
    """Demonstrate semantic relationship modeling."""
    print("=== Semantic Relationships Demo ===\n")
    
    kb = KnowledgeBase(dimension=10000)
    
    # Example 1: Taxonomic relationships
    print("1. Taxonomic Hierarchy:")
    
    # Build taxonomy
    kb.add_fact('dog', 'is_a', 'mammal')
    kb.add_fact('cat', 'is_a', 'mammal')
    kb.add_fact('mammal', 'is_a', 'animal')
    kb.add_fact('bird', 'is_a', 'animal')
    kb.add_fact('sparrow', 'is_a', 'bird')
    
    # Properties inheritance
    kb.add_fact('mammal', 'has', 'fur')
    kb.add_fact('bird', 'has', 'feathers')
    kb.add_fact('animal', 'can', 'move')
    
    # Query relationships
    print("   Direct relationships:")
    dog_mammal = kb.facts['dog-is_a-mammal']
    subject, sim = kb.query('SUBJECT', dog_mammal)
    print(f"   - Query subject of 'X is_a mammal': {subject} (sim: {sim:.3f})")
    
    # Example 2: Part-whole relationships
    print("\n2. Part-Whole Relationships:")
    
    # Build part-whole structure
    kb.add_fact('wheel', 'part_of', 'car')
    kb.add_fact('engine', 'part_of', 'car')
    kb.add_fact('window', 'part_of', 'car')
    kb.add_fact('leaf', 'part_of', 'tree')
    kb.add_fact('branch', 'part_of', 'tree')
    
    # Create composite representation
    car_parts = kb.vsa.bundle([
        kb.facts['wheel-part_of-car'],
        kb.facts['engine-part_of-car'],
        kb.facts['window-part_of-car']
    ])
    
    print("   Car has multiple parts bundled together")
    
    # Example 3: Causal relationships
    print("\n3. Causal Relationships:")
    
    # Build causal chain
    kb.add_fact('rain', 'causes', 'wet_ground')
    kb.add_fact('wet_ground', 'causes', 'slippery')
    kb.add_fact('slippery', 'causes', 'dangerous')
    
    # Trace causal chain
    print("   Causal chain: rain → wet_ground → slippery → dangerous")
    
    # Example 4: Spatial relationships
    print("\n4. Spatial Relationships:")
    
    # Build spatial scene
    kb.add_fact('book', 'on', 'table')
    kb.add_fact('table', 'in', 'room')
    kb.add_fact('lamp', 'above', 'table')
    kb.add_fact('chair', 'next_to', 'table')
    
    # Query spatial relations
    book_table = kb.facts['book-on-table']
    obj, sim = kb.query('OBJECT', book_table)
    print(f"   - What is on the table? {obj} (sim: {sim:.3f})\n")


def demonstrate_question_answering():
    """Demonstrate question answering with VSA."""
    print("=== Question Answering Demo ===\n")
    
    kb = KnowledgeBase(dimension=10000)
    
    # Build knowledge base
    print("1. Building Knowledge Base:")
    
    # Add facts about people
    kb.add_fact('John', 'occupation', 'doctor')
    kb.add_fact('Mary', 'occupation', 'teacher')
    kb.add_fact('Bob', 'occupation', 'engineer')
    
    kb.add_fact('John', 'lives_in', 'New_York')
    kb.add_fact('Mary', 'lives_in', 'Boston')
    kb.add_fact('Bob', 'lives_in', 'New_York')
    
    kb.add_fact('John', 'age', '35')
    kb.add_fact('Mary', 'age', '28')
    kb.add_fact('Bob', 'age', '42')
    
    print("   Added facts about people, occupations, locations, and ages")
    
    # Answer questions
    print("\n2. Answering Questions:")
    
    # Question 1: What is John's occupation?
    johns_occupation = kb.facts['John-occupation-doctor']
    occupation, sim = kb.query('OBJECT', johns_occupation)
    print(f"   Q: What is John's occupation?")
    print(f"   A: {occupation} (confidence: {sim:.3f})")
    
    # Question 2: Who lives in New York?
    print("\n   Q: Who lives in New York?")
    ny_residents = []
    for fact_key, fact_vec in kb.facts.items():
        if 'lives_in-New_York' in fact_key:
            resident, sim = kb.query('SUBJECT', fact_vec)
            if sim > 0.4:  # Lower threshold for multiplication binding
                ny_residents.append(resident)
    print(f"   A: {', '.join(ny_residents)}")
    
    # Question 3: Complex query - Find doctors in New York
    print("\n   Q: Which doctors live in New York?")
    
    # Create query vector
    doctor_vec = kb.concepts['doctor']
    ny_vec = kb.concepts['New_York']
    occupation_role = kb.concepts['occupation']
    location_role = kb.concepts['lives_in']
    
    # Find people who are doctors
    doctors = []
    for fact_key, fact_vec in kb.facts.items():
        if 'occupation-doctor' in fact_key:
            person, sim = kb.query('SUBJECT', fact_vec)
            if sim > 0.4:  # Lower threshold for multiplication binding
                doctors.append(person)
    
    # Check which doctors live in NY
    doctors_in_ny = []
    for doctor in doctors:
        fact_key = f"{doctor}-lives_in-New_York"
        if fact_key in kb.facts:
            doctors_in_ny.append(doctor)
    
    print(f"   A: {', '.join(doctors_in_ny) if doctors_in_ny else 'None found'}\n")


def demonstrate_rule_based_inference():
    """Demonstrate rule-based inference with VSA."""
    print("=== Rule-Based Inference Demo ===\n")
    
    kb = KnowledgeBase(dimension=10000)
    
    # Define rules
    print("1. Defining Rules:")
    
    # Rule 1: If X is a bird, then X can fly
    def bird_fly_rule(kb, entity):
        # Check if entity is a bird
        for fact_key, fact_vec in kb.facts.items():
            if f"{entity}-is_a-bird" in fact_key:
                # Add flying ability
                kb.add_fact(entity, 'can', 'fly')
                return True
        return False
    
    # Rule 2: If X is a mammal, then X is warm-blooded
    def mammal_warm_rule(kb, entity):
        for fact_key, fact_vec in kb.facts.items():
            if f"{entity}-is_a-mammal" in fact_key:
                kb.add_fact(entity, 'is', 'warm_blooded')
                return True
        return False
    
    # Add rules
    kb.add_rule('bird_can_fly', 
                lambda e: f"{e}-is_a-bird" in kb.facts,
                lambda e: kb.add_fact(e, 'can', 'fly'))
    
    kb.add_rule('mammal_warm_blooded',
                lambda e: f"{e}-is_a-mammal" in kb.facts,
                lambda e: kb.add_fact(e, 'is', 'warm_blooded'))
    
    print("   - Rule 1: If X is a bird, then X can fly")
    print("   - Rule 2: If X is a mammal, then X is warm-blooded")
    
    # Add base facts
    print("\n2. Adding Base Facts:")
    kb.add_fact('robin', 'is_a', 'bird')
    kb.add_fact('eagle', 'is_a', 'bird')
    kb.add_fact('dog', 'is_a', 'mammal')
    kb.add_fact('cat', 'is_a', 'mammal')
    print("   - Added: robin and eagle are birds")
    print("   - Added: dog and cat are mammals")
    
    # Apply rules
    print("\n3. Applying Inference Rules:")
    
    # Apply bird rule
    for entity in ['robin', 'eagle']:
        if bird_fly_rule(kb, entity):
            print(f"   - Inferred: {entity} can fly")
    
    # Apply mammal rule
    for entity in ['dog', 'cat']:
        if mammal_warm_rule(kb, entity):
            print(f"   - Inferred: {entity} is warm-blooded")
    
    # Verify inferences
    print("\n4. Verifying Inferences:")
    robin_fly = kb.facts.get('robin-can-fly')
    if robin_fly is not None:
        obj, sim = kb.query('OBJECT', robin_fly)
        print(f"   - Robin can: {obj} (confidence: {sim:.3f})")
    
    dog_warm = kb.facts.get('dog-is-warm_blooded')
    if dog_warm is not None:
        obj, sim = kb.query('OBJECT', dog_warm)
        print(f"   - Dog is: {obj} (confidence: {sim:.3f})\n")


def demonstrate_cognitive_operations():
    """Demonstrate cognitive operations like attention and memory."""
    print("=== Cognitive Operations Demo ===\n")
    
    vsa = create_vsa(
        dimension=10000,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )
    
    # Example 1: Attention mechanism
    print("1. Attention Mechanism:")
    
    # Create scene with multiple objects
    color_role = vsa.generate_vector()
    type_role = vsa.generate_vector()
    red = vsa.generate_vector()
    blue = vsa.generate_vector()
    green = vsa.generate_vector()
    car = vsa.generate_vector()
    house = vsa.generate_vector()
    tree = vsa.generate_vector()
    
    objects = {
        'red_car': vsa.bundle([
            vsa.bind(color_role, red),
            vsa.bind(type_role, car)
        ]),
        'blue_house': vsa.bundle([
            vsa.bind(color_role, blue),
            vsa.bind(type_role, house)
        ]),
        'green_tree': vsa.bundle([
            vsa.bind(color_role, green),
            vsa.bind(type_role, tree)
        ])
    }
    
    # Create scene
    scene = vsa.bundle(list(objects.values()))
    
    # Attention query: Find red objects
    red_query = vsa.bind(color_role, red)
    
    # Check each object
    print("   Looking for red objects:")
    for name, obj in objects.items():
        similarity = vsa.similarity(obj, red_query)
        print(f"   - {name}: attention weight = {similarity:.3f}")
    
    # Example 2: Working memory
    print("\n2. Working Memory Operations:")
    
    # Create memory slots
    slot1 = vsa.generate_vector()
    slot2 = vsa.generate_vector()
    slot3 = vsa.generate_vector()
    
    # Store items in memory
    item1 = vsa.generate_vector()
    item2 = vsa.generate_vector()
    item3 = vsa.generate_vector()
    
    working_memory = vsa.bundle([
        vsa.bind(slot1, item1),
        vsa.bind(slot2, item2),
        vsa.bind(slot3, item3)
    ])
    
    # Retrieve from memory
    print("   Working memory contents:")
    for i, slot in enumerate([slot1, slot2, slot3], 1):
        retrieved = vsa.unbind(working_memory, slot)
        items = [('apple', item1), ('banana', item2), ('orange', item3)]
        for item_name, item_vec in items:
            sim = vsa.similarity(retrieved, item_vec)
            if sim > 0.3:  # Lower threshold for multiplication binding
                print(f"   - Slot {i}: {item_name} (sim: {sim:.3f})")
                break
    
    # Example 3: Priming effects
    print("\n3. Semantic Priming:")
    
    # Create semantic network
    concepts = {
        'doctor': ['hospital', 'medicine', 'patient'],
        'teacher': ['school', 'student', 'lesson'],
        'chef': ['kitchen', 'food', 'cooking']
    }
    
    # Build associations
    associations = {}
    concept_vecs = {}
    
    # Generate vectors for all concepts
    for main_concept, related in concepts.items():
        concept_vecs[main_concept] = vsa.generate_vector()
        for r in related:
            if r not in concept_vecs:
                concept_vecs[r] = vsa.generate_vector()
    
    # Build associations
    for main_concept, related in concepts.items():
        main_vec = concept_vecs[main_concept]
        related_vecs = [concept_vecs[r] for r in related]
        associations[main_concept] = vsa.bundle([
            vsa.bind(main_vec, r) for r in related_vecs
        ])
    
    # Test priming
    prime = 'doctor'
    target = 'hospital'
    
    prime_vec = concept_vecs[prime]
    target_vec = concept_vecs[target]
    
    # Measure association strength
    association = associations[prime]
    priming_effect = vsa.similarity(
        vsa.bind(prime_vec, target_vec),
        association
    )
    
    print(f"   Priming effect: '{prime}' → '{target}'")
    print(f"   Association strength: {priming_effect:.3f}\n")


def main():
    """Run all symbolic reasoning demonstrations."""
    print("\n" + "="*60)
    print("VSA SYMBOLIC REASONING DEMONSTRATION")
    print("="*60 + "\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstrations
    demonstrate_analogical_reasoning()
    print("-"*60 + "\n")
    
    demonstrate_compositional_structures()
    print("-"*60 + "\n")
    
    demonstrate_logic_operations()
    print("-"*60 + "\n")
    
    demonstrate_semantic_relationships()
    print("-"*60 + "\n")
    
    demonstrate_question_answering()
    print("-"*60 + "\n")
    
    demonstrate_rule_based_inference()
    print("-"*60 + "\n")
    
    demonstrate_cognitive_operations()
    
    # Summary
    print("=== Summary ===\n")
    print("VSA enables sophisticated symbolic reasoning through:")
    print("• Analogical reasoning via transformation learning")
    print("• Compositional structures with role-filler binding")
    print("• Fuzzy logic operations and quantifiers")
    print("• Semantic relationship modeling")
    print("• Question answering over structured knowledge")
    print("• Rule-based inference")
    print("• Cognitive operations like attention and memory\n")
    
    print("Key advantages:")
    print("• Graceful degradation with noise")
    print("• Continuous similarity measures")
    print("• Fixed-size representations")
    print("• Parallelizable operations")
    print("• Biologically plausible\n")
    
    print("="*60)
    print("Symbolic Reasoning Demo Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()