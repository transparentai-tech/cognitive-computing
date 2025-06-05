#!/usr/bin/env python3
"""Fix remaining syntax errors in SPA examples."""

import re
import os

def fix_file(filepath):
    """Fix syntax errors in a file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix patterns where we have extra ) after .vector
    patterns = [
        # Fix lines like: state.state = vocab["DOG"].vector)
        (r'(\w+\.state = [^)]+\.vector)\)', r'\1'),
        # Fix lines like: outcome.state = vocab["REWARD"].vector)
        (r'(\.state = vocab\[[^\]]+\]\.vector)\)', r'\1'),
        # Fix lines like: confidence.state = np.array([sim]))
        (r'(confidence\.state = np\.array\(\[[^\]]+\]\))\)', r'\1'),
        # Fix lines like: classification.state = np.zeros(256))
        (r'(\.state = np\.zeros\(\d+\))\)', r'\1'),
        # Fix lines like: perception.state = vocab["SEE_FOOD"].vector)
        (r'(perception\.state = vocab\[[^\]]+\]\.vector)\)', r'\1'),
        # Fix lines like: question.state = q1.vector)
        (r'(question\.state = \w+\.vector)\)', r'\1'),
        # Fix lines like: input_state.state = vocab["URGENT"].vector)
        (r'(input_state\.state = [^)]+\.vector)\)', r'\1'),
        # Fix lines like: control.set_current_goal(vocab["COLOR_TASK"].vector)
        (r'(control\.\w+\(vocab\[[^\]]+\]\.vector)\)(\s|$)', r'\1)\2'),
        # Fix lines like: control.push_task(vocab["COLOR_TASK"].vector)
        (r'(control\.push_task\(vocab\[[^\]]+\]\.vector)\)(\s|$)', r'\1)\2'),
        # Fix lines like: control.is_task_active(vocab["COLOR_TASK"].vector)
        (r'(control\.is_task_active\(vocab\[[^\]]+\]\.vector)\)(\s|$)', r'\1)\2'),
        # Fix lines like: control.is_goal_active(vocab["FIND_FOOD"].vector)
        (r'(control\.is_goal_active\(vocab\[[^\]]+\]\.vector)\)(\s|$)', r'\1)\2'),
        # Fix lines ending with .vector)
        (r'= ([^=]+)\.vector\)(\s*$)', r'= \1.vector\2'),
        # Fix lines like: query = vocab["LONDON"]
        (r'(query = vocab\[[^\]]+\])\n', r'\1\n'),
        # Fix lines like: query = vocab["WW2"]
        (r'(query\d* = vocab\[[^\]]+\])\n', r'\1\n'),
        # Fix lines like: facts = kb.query(query.vector)
        (r'(facts\d* = kb\.query\([^)]+\.vector)\)(\s|$)', r'\1)\2'),
        # Fix lines like: facts = kb.query(vocab["RELATIVITY"].vector
        (r'(facts = kb\.query\(vocab\[[^\]]+\]\.vector)(\s|$)', r'\1)\2'),
        # Fix lines like: if kb.query(vocab["EIFFEL_TOWER"].vector:
        (r'if kb\.query\(vocab\[[^\]]+\]\.vector:', r'if kb.query(vocab["EIFFEL_TOWER"].vector):'),
        # Fix lines like: if kb.query(vocab["PARIS"].vector:
        (r'if kb\.query\(vocab\[[^\]]+\]\.vector:', r'if kb.query(vocab["PARIS"].vector):'),
        # Fix lines like: if kb.query(vocab["FRANCE"].vector:
        (r'if kb\.query\(vocab\[[^\]]+\]\.vector:', r'if kb.query(vocab["FRANCE"].vector):'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

# Fix all SPA examples
spa_examples = [
    'examples/spa/basic_spa_demo.py',
    'examples/spa/cognitive_control.py', 
    'examples/spa/production_system.py',
    'examples/spa/question_answering.py',
    'examples/spa/sequential_behavior.py',
    'examples/spa/neural_implementation.py'
]

for example in spa_examples:
    if os.path.exists(example):
        fix_file(example)