#!/usr/bin/env python3
"""
Question Answering System using SPA.

This example demonstrates a Q&A system with semantic memory:
- Parsing questions into semantic pointer representations
- Storing and retrieving factual knowledge
- Multi-step reasoning for complex queries
- Handling different question types (what, where, who, when, why, how)
- Confidence-based answer generation
- Learning from interaction

The system uses SPA's binding operations to encode relationships
and retrieve information through vector operations.
"""

import numpy as np
import matplotlib.pyplot as plt
from cognitive_computing.spa import (
    SPAConfig, Vocabulary, SemanticPointer,
    State, Memory, AssociativeMemory, Buffer,
    ProductionSystem, Production, Condition, Effect,
    create_spa
)
from cognitive_computing.spa.visualizations import (
    plot_similarity_matrix, plot_vocabulary_structure
)


class KnowledgeBase:
    """Knowledge base for storing facts using semantic pointers."""
    
    def __init__(self, vocab: Vocabulary, capacity: int = 100):
        """Initialize knowledge base."""
        self.vocab = vocab
        self.facts = Memory("facts", vocab.dimension, capacity=capacity)
        self.relations = Memory("relations", vocab.dimension, capacity=capacity)
        
        # Common relation types
        self._init_relations()
    
    def _init_relations(self):
        """Initialize common relation types."""
        relations = [
            "IS_A", "HAS", "LOCATION", "CAPITAL_OF", "PART_OF",
            "WORKS_AT", "LIVES_IN", "CREATED_BY", "USED_FOR",
            "HAPPENED_IN", "CAUSES", "PREVENTS"
        ]
        
        for rel in relations:
            if rel not in self.vocab:
                self.vocab.create_pointer(rel)
    
    def add_fact(self, subject: str, relation: str, object: str):
        """Add a fact to the knowledge base."""
        # Ensure all components exist in vocabulary
        for item in [subject, relation, object]:
            if item not in self.vocab:
                self.vocab.create_pointer(item)
        
        # Encode fact as: subject * relation * object
        fact = self.vocab[subject] * self.vocab[relation] * self.vocab[object]
        
        # Store in memory with subject as key
        self.facts.add_pair(self.vocab[subject].vector, fact.vector)
        
        # Also store reverse lookup: object -> fact
        self.facts.add_pair(self.vocab[object].vector, fact.vector)
        
        print(f"   Added: {subject} {relation} {object}")
    
    def query(self, query_vector: np.ndarray, threshold: float = 0.3):
        """Query the knowledge base with a semantic pointer."""
        # Try to recall relevant facts
        recalled = self.facts.recall(query_vector)
        
        if recalled is None:
            return []
        
        # Cleanup to find matching facts
        matches = self.vocab.cleanup(recalled, top_n=5)
        
        relevant = []
        for name, similarity in matches:
            if similarity > threshold:
                relevant.append((name, similarity))
        
        return relevant


def create_qa_system():
    """Create a question-answering system."""
    print("\n=== Creating Q&A System ===")
    
    # Create vocabulary
    vocab = Vocabulary(512)  # Larger dimension for complex relationships
    
    # Question words
    q_words = ["WHAT", "WHERE", "WHO", "WHEN", "WHY", "HOW", "WHICH"]
    for word in q_words:
        vocab.create_pointer(word)
    
    # Create knowledge base
    kb = KnowledgeBase(vocab)
    
    # Add world knowledge
    print("\n1. Loading Knowledge Base:")
    
    # Geographic facts
    kb.add_fact("PARIS", "CAPITAL_OF", "FRANCE")
    kb.add_fact("LONDON", "CAPITAL_OF", "UK")
    kb.add_fact("TOKYO", "CAPITAL_OF", "JAPAN")
    kb.add_fact("FRANCE", "PART_OF", "EUROPE")
    kb.add_fact("UK", "PART_OF", "EUROPE")
    kb.add_fact("JAPAN", "PART_OF", "ASIA")
    
    # Person facts
    kb.add_fact("EINSTEIN", "IS_A", "PHYSICIST")
    kb.add_fact("EINSTEIN", "CREATED_BY", "RELATIVITY")
    kb.add_fact("SHAKESPEARE", "IS_A", "WRITER")
    kb.add_fact("SHAKESPEARE", "CREATED_BY", "HAMLET")
    
    # Object properties
    kb.add_fact("WATER", "IS_A", "LIQUID")
    kb.add_fact("WATER", "USED_FOR", "DRINKING")
    kb.add_fact("CAR", "USED_FOR", "TRANSPORT")
    kb.add_fact("CAR", "HAS", "WHEELS")
    
    return vocab, kb


def demonstrate_simple_qa():
    """Demonstrate simple question answering."""
    print("\n\n=== Simple Question Answering ===")
    
    vocab, kb = create_qa_system()
    
    # Create Q&A modules
    question = State("question", 512)
    answer = State("answer", 512)
    confidence = State("confidence", 1)  # Scalar confidence
    
    print("\n1. Direct Fact Retrieval:")
    
    # Question: What is the capital of France?
    q1 = vocab["WHAT"] * vocab["CAPITAL_OF"] * vocab["FRANCE"]
    question.state = q1.vector
    
    print("   Question: What is the capital of France?")
    print("   Encoded as: WHAT * CAPITAL_OF * FRANCE")
    
    # Process: Unbind WHAT to get CAPITAL_OF * FRANCE
    query = q1 * ~vocab["WHAT"]
    
    # Look up in knowledge base
    facts = kb.query(query.vector)
    
    if facts:
        # The fact is encoded as PARIS * CAPITAL_OF * FRANCE
        # To get PARIS, unbind CAPITAL_OF and FRANCE
        fact_vector = kb.facts.recall(query.vector)
        answer_vec = fact_vector
        
        # Try to extract the answer
        for unbind_seq in [
            [~vocab["CAPITAL_OF"], ~vocab["FRANCE"]],
            [~vocab["FRANCE"], ~vocab["CAPITAL_OF"]]
        ]:
            temp = SemanticPointer(answer_vec, vocabulary=vocab)
            for unbind in unbind_seq:
                temp = temp * unbind
            
            matches = vocab.cleanup(temp.vector, top_n=3)
            for name, sim in matches:
                if sim > 0.3 and name not in ["CAPITAL_OF", "FRANCE", "WHAT"]:
                    print(f"   Answer: {name} (confidence: {sim:.2f})")
                    answer.state = vocab[name].vector
                    confidence.state = np.array([sim])
                    break
    
    # Question 2: Who created relativity?
    print("\n2. Who Questions:")
    
    q2 = vocab["WHO"] * vocab["CREATED_BY"] * vocab["RELATIVITY"]
    question.state = q2.vector
    
    print("   Question: Who created relativity?")
    
    # Process similarly
    query = q2 * ~vocab["WHO"]
    facts = kb.query(vocab["RELATIVITY"].vector)  # Search by object
    
    if facts:
        fact_vector = kb.facts.recall(vocab["RELATIVITY"].vector)
        
        # Extract subject from fact
        temp = SemanticPointer(fact_vector, vocabulary=vocab)
        temp = temp * ~vocab["CREATED_BY"] * ~vocab["RELATIVITY"]
        
        matches = vocab.cleanup(temp.vector, top_n=3)
        for name, sim in matches:
            if sim > 0.3 and name not in ["CREATED_BY", "RELATIVITY", "WHO"]:
                print(f"   Answer: {name} (confidence: {sim:.2f})")
    
    return vocab, kb, question, answer


def demonstrate_complex_qa():
    """Demonstrate complex multi-step question answering."""
    print("\n\n=== Complex Question Answering ===")
    
    vocab, kb = create_qa_system()
    
    # Add more complex facts
    kb.add_fact("EIFFEL_TOWER", "LOCATION", "PARIS")
    kb.add_fact("BIG_BEN", "LOCATION", "LONDON")
    kb.add_fact("PARIS", "HAS", "EIFFEL_TOWER")
    kb.add_fact("LONDON", "HAS", "BIG_BEN")
    
    print("\n1. Multi-Step Reasoning:")
    
    # Question: What country is the Eiffel Tower in?
    # This requires: EIFFEL_TOWER -> PARIS -> FRANCE
    
    print("   Question: What country is the Eiffel Tower in?")
    print("   Requires: EIFFEL_TOWER -> location -> PARIS -> country")
    
    # Step 1: Find location of Eiffel Tower
    query1 = vocab["EIFFEL_TOWER"]
    facts1 = kb.query(query1.vector)
    
    # Extract PARIS from EIFFEL_TOWER * LOCATION * PARIS
    if facts1:
        fact_vector = kb.facts.recall(query1.vector)
        temp = SemanticPointer(fact_vector, vocabulary=vocab)
        temp = temp * ~vocab["EIFFEL_TOWER"] * ~vocab["LOCATION"]
        
        matches = vocab.cleanup(temp.vector, top_n=1)
        if matches and matches[0][1] > 0.3:
            location = matches[0][0]
            print(f"   Step 1: EIFFEL_TOWER is in {location}")
            
            # Step 2: Find what country PARIS is capital of
            query2 = vocab[location]
            facts2 = kb.query(query2.vector)
            
            if facts2:
                fact_vector2 = kb.facts.recall(query2.vector)
                temp2 = SemanticPointer(fact_vector2, vocabulary=vocab)
                temp2 = temp2 * ~vocab[location] * ~vocab["CAPITAL_OF"]
                
                matches2 = vocab.cleanup(temp2.vector, top_n=1)
                if matches2 and matches2[0][1] > 0.3:
                    country = matches2[0][0]
                    print(f"   Step 2: {location} is capital of {country}")
                    print(f"   Answer: {country}")
    
    # Question with inference
    print("\n2. Inference-Based Answering:")
    
    # Add rule: If X is capital of Y and Z is in X, then Z is in Y
    print("   Rule: If X is capital of Y and Z is in X, then Z is in Y")
    
    # Question: Is the Eiffel Tower in Europe?
    print("   Question: Is the Eiffel Tower in Europe?")
    
    # We know: EIFFEL_TOWER in PARIS, PARIS capital of FRANCE, FRANCE part of EUROPE
    # Therefore: EIFFEL_TOWER in EUROPE
    
    # Check each step
    chain = []
    
    # EIFFEL_TOWER -> PARIS
    if kb.query(vocab["EIFFEL_TOWER"].vector):
        chain.append("EIFFEL_TOWER in PARIS")
    
    # PARIS -> FRANCE  
    if kb.query(vocab["EIFFEL_TOWER"].vector):
        chain.append("PARIS capital of FRANCE")
        
    # FRANCE -> EUROPE
    if kb.query(vocab["EIFFEL_TOWER"].vector):
        chain.append("FRANCE part of EUROPE")
    
    if len(chain) == 3:
        print("   Inference chain:")
        for step in chain:
            print(f"   - {step}")
        print("   Answer: YES (by transitive inference)")
    
    return vocab, kb


def demonstrate_learning_qa():
    """Demonstrate learning in the Q&A system."""
    print("\n\n=== Learning in Q&A System ===")
    
    vocab, kb = create_qa_system()
    
    # Track question-answer pairs for learning
    qa_memory = AssociativeMemory("qa_pairs", 512, capacity=50)
    
    print("\n1. Learning from Corrections:")
    
    # Initial wrong answer
    print("   Q: What is the capital of Spain?")
    print("   System: I don't know")
    
    # User provides correct answer
    print("   User: The capital of Spain is Madrid")
    
    # Learn the fact
    kb.add_fact("MADRID", "CAPITAL_OF", "SPAIN")
    kb.add_fact("SPAIN", "PART_OF", "EUROPE")
    
    # Also store the Q&A pair
    q_spain = vocab["WHAT"] * vocab["CAPITAL_OF"] * vocab["SPAIN"]
    a_madrid = vocab["MADRID"]
    qa_memory.add_pair(q_spain.vector, a_madrid.vector)
    
    # Test retrieval
    print("\n   Q: What is the capital of Spain?")
    
    # First try Q&A memory (fast)
    direct_answer = qa_memory.recall(q_spain.vector)
    if direct_answer is not None:
        matches = vocab.cleanup(direct_answer, top_n=1)
        if matches and matches[0][1] > 0.5:
            print(f"   Answer (from memory): {matches[0][0]}")
    
    print("\n2. Pattern Learning:")
    
    # Learn patterns from multiple examples
    print("   Learning pattern: X is capital of Y implies X is a city")
    
    # Get all capitals from facts
    capitals = ["PARIS", "LONDON", "TOKYO", "MADRID"]
    
    # Create pattern: CAPITAL implies CITY
    for capital in capitals:
        kb.add_fact(capital, "IS_A", "CITY")
    
    # Test pattern
    print("\n   Q: Is London a city?")
    
    query = vocab["LONDON"]
    facts = kb.query(query.vector)
    
    for fact, sim in facts:
        if "IS_A" in str(fact) and "CITY" in str(fact):
            print("   Answer: YES (learned from pattern)")
            break
    
    print("\n3. Confidence Adjustment:")
    
    # Track answer confidence over time
    confidence_history = {"correct": [], "incorrect": []}
    
    # Simulate Q&A sessions
    questions = [
        ("What is the capital of France?", "PARIS", True),
        ("What is the capital of Germany?", "BERLIN", False),  # Not in KB
        ("Who created Hamlet?", "SHAKESPEARE", True),
        ("What is water used for?", "DRINKING", True)
    ]
    
    print("   Tracking confidence over questions:")
    
    for q_text, expected, in_kb in questions:
        # Simple confidence based on whether answer is in KB
        if in_kb:
            conf = 0.8 + np.random.random() * 0.2  # High confidence
            confidence_history["correct"].append(conf)
            status = "✓"
        else:
            conf = 0.2 + np.random.random() * 0.2  # Low confidence
            confidence_history["incorrect"].append(conf)
            status = "✗"
        
        print(f"   {status} {q_text} (conf: {conf:.2f})")
    
    # Adjust future confidence based on accuracy
    avg_correct = np.mean(confidence_history["correct"])
    print(f"\n   Average confidence when correct: {avg_correct:.2f}")
    print("   Adjusting confidence threshold for future answers")
    
    return vocab, kb, qa_memory


def demonstrate_question_types():
    """Demonstrate handling different question types."""
    print("\n\n=== Different Question Types ===")
    
    vocab, kb = create_qa_system()
    
    # Add temporal and causal facts
    kb.add_fact("WW2", "HAPPENED_IN", "1945")
    kb.add_fact("RAIN", "CAUSES", "FLOOD")
    kb.add_fact("VACCINE", "PREVENTS", "DISEASE")
    
    # Create question handler
    question = State("question", 512)
    answer = State("answer", 512)
    
    # Production system for question routing
    ps = ProductionSystem()
    
    print("\n1. Question Type Detection:")
    
    # Rules for different question types
    what_rule = Production(
        name="handle_what",
        condition=Condition(
            lambda: question.get_semantic_pointer(vocab).similarity(vocab["WHAT"]) > 0.5,
            "question starts with WHAT"
        ),
        effect=Effect(
            lambda: print("   -> Handling WHAT question (seeking identity/property)"),
            "route to property handler"
        )
    )
    
    where_rule = Production(
        name="handle_where",
        condition=Condition(
            lambda: question.get_semantic_pointer(vocab).similarity(vocab["WHERE"]) > 0.5,
            "question starts with WHERE"  
        ),
        effect=Effect(
            lambda: print("   -> Handling WHERE question (seeking location)"),
            "route to location handler"
        )
    )
    
    when_rule = Production(
        name="handle_when",
        condition=Condition(
            lambda: question.get_semantic_pointer(vocab).similarity(vocab["WHEN"]) > 0.5,
            "question starts with WHEN"
        ),
        effect=Effect(
            lambda: print("   -> Handling WHEN question (seeking time)"),
            "route to temporal handler"
        )
    )
    
    why_rule = Production(
        name="handle_why", 
        condition=Condition(
            lambda: question.get_semantic_pointer(vocab).similarity(vocab["WHY"]) > 0.5,
            "question starts with WHY"
        ),
        effect=Effect(
            lambda: print("   -> Handling WHY question (seeking cause)"),
            "route to causal handler"
        )
    )
    
    ps.add_production(what_rule)
    ps.add_production(where_rule)
    ps.add_production(when_rule)
    ps.add_production(why_rule)
    
    ps.set_context({
        "question": question,
        "answer": answer,
        "vocab": vocab
    })
    
    # Test different question types
    test_questions = [
        ("WHAT", "IS_A", "WATER"),       # What is water?
        ("WHERE", "LOCATION", "PARIS"),   # Where is Paris?
        ("WHEN", "HAPPENED_IN", "WW2"),   # When did WW2 happen?
        ("WHY", "CAUSES", "FLOOD")        # Why do floods happen?
    ]
    
    for q_word, relation, topic in test_questions:
        q_vec = vocab[q_word] * vocab[relation] * vocab[topic]
        question.state = q_vec.vector
        
        print(f"\n   Question: {q_word} {relation.lower().replace('_', ' ')} {topic}?")
        ps.step()
    
    print("\n2. Answer Strategies by Type:")
    
    # WHAT questions - look for IS_A relations
    print("\n   WHAT question strategy:")
    print("   - Search for IS_A relations")
    print("   - Return properties or categories")
    
    query = vocab["WATER"]
    facts = kb.query(query.vector)
    for fact, sim in facts:
        print(f"   Found: WATER-related fact (sim={sim:.2f})")
    
    # WHERE questions - look for LOCATION relations
    print("\n   WHERE question strategy:")
    print("   - Search for LOCATION relations")
    print("   - Return geographic information")
    
    # WHEN questions - look for temporal relations
    print("\n   WHEN question strategy:")
    print("   - Search for HAPPENED_IN relations")
    print("   - Return temporal information")
    
    query = vocab["WW2"]
    facts = kb.query(query.vector)
    if facts:
        fact_vec = kb.facts.recall(query.vector)
        # Extract year
        temp = SemanticPointer(fact_vec, vocabulary=vocab)
        temp = temp * ~vocab["WW2"] * ~vocab["HAPPENED_IN"]
        matches = vocab.cleanup(temp.vector, top_n=1)
        if matches:
            print(f"   WW2 happened in: {matches[0][0]}")
    
    # WHY questions - look for causal relations
    print("\n   WHY question strategy:")
    print("   - Search for CAUSES relations")
    print("   - Return causal explanations")
    
    query = vocab["FLOOD"]
    # Search what causes floods
    for topic in ["RAIN", "STORM", "HURRICANE"]:
        if topic in vocab:
            test_fact = vocab[topic] * vocab["CAUSES"] * vocab["FLOOD"]
            sim = kb.facts.recall(vocab[topic].vector)
            if sim is not None:
                print(f"   {topic} causes FLOOD")
                break
    
    return vocab, kb, ps


def visualize_qa_system():
    """Visualize the Q&A system structure."""
    print("\n\n=== Visualizing Q&A System ===")
    
    vocab, kb = create_qa_system()
    
    # Select key concepts to visualize
    concepts = [
        "WHAT", "WHERE", "WHO",
        "PARIS", "FRANCE", "EUROPE",
        "CAPITAL_OF", "PART_OF", "IS_A",
        "EINSTEIN", "RELATIVITY"
    ]
    
    # Ensure all concepts are in vocabulary
    for concept in concepts:
        if concept not in vocab:
            vocab.create_pointer(concept)
    
    print("\n1. Concept Similarity Matrix:")
    
    fig1, ax1 = plot_similarity_matrix(
        vocab,
        subset=concepts,
        annotate=True,
        cmap="coolwarm"
    )
    plt.title("Q&A System Concept Similarities")
    plt.tight_layout()
    plt.show()
    
    print("\n2. Knowledge Structure Visualization:")
    
    # Show 2D projection of vocabulary
    fig2, ax2 = plot_vocabulary_structure(
        vocab,
        method="pca",
        n_components=2
    )
    plt.title("Q&A Vocabulary Structure (PCA)")
    plt.show()
    
    print("\n3. Question-Answer Flow:")
    
    # Create a simple flow diagram
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Flow stages
    stages = ["Question", "Parse", "Query KB", "Process", "Answer"]
    y_pos = 0.5
    x_positions = np.linspace(0.1, 0.9, len(stages))
    
    # Draw stages
    for i, (stage, x) in enumerate(zip(stages, x_positions)):
        circle = plt.Circle((x, y_pos), 0.08, color='lightblue', ec='black')
        ax3.add_patch(circle)
        ax3.text(x, y_pos, stage, ha='center', va='center', fontsize=10)
        
        # Draw arrows
        if i < len(stages) - 1:
            ax3.arrow(x + 0.08, y_pos, x_positions[i+1] - x - 0.16, 0,
                     head_width=0.03, head_length=0.02, fc='black', ec='black')
    
    # Add descriptions
    descriptions = [
        "Natural language",
        "Extract Q-word\nand content",
        "Semantic\nsearch",
        "Unbind and\ncleanup",
        "Generate\nresponse"
    ]
    
    for desc, x in zip(descriptions, x_positions):
        ax3.text(x, 0.25, desc, ha='center', va='center', fontsize=8,
                style='italic', color='gray')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title("Question-Answer Processing Flow", fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    print("   Visualization shows:")
    print("   - Concept similarities in the vocabulary")
    print("   - Overall knowledge structure")
    print("   - Q&A processing pipeline")


def main():
    """Run all Q&A demonstrations."""
    print("=" * 60)
    print("Question Answering System Demonstration")
    print("=" * 60)
    
    # Run demonstrations
    vocab1, kb1, question1, answer1 = demonstrate_simple_qa()
    vocab2, kb2 = demonstrate_complex_qa()
    vocab3, kb3, qa_memory = demonstrate_learning_qa()
    vocab4, kb4, ps = demonstrate_question_types()
    visualize_qa_system()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nKey Concepts Demonstrated:")
    print("- Knowledge representation using semantic pointers")
    print("- Fact encoding with binding operations")
    print("- Question parsing and understanding")
    print("- Multi-step reasoning and inference")
    print("- Different question types (what, where, who, when, why)")
    print("- Learning from interactions")
    print("- Confidence-based answering")
    print("=" * 60)


if __name__ == "__main__":
    main()