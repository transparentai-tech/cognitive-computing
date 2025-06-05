#!/usr/bin/env python3
"""Fix missing closing parentheses in examples."""

import re
import os

def fix_file(filepath):
    """Fix missing closing parentheses in a file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix patterns where we have missing closing parenthesis after .vector
    patterns = [
        # Fix lines ending with .vector but missing )
        (r'(\.vector)\n', r'.vector)\n'),
        # Fix lines with .state = something.vector (missing closing paren)
        (r'(\.state = .*\.vector)\n', r'\1)\n'),
        # Fix lines with recall(...vector missing closing paren
        (r'(\.recall\([^)]+\.vector)\n', r'\1)\n'),
        # Fix lines with add_pair(...vector missing closing paren  
        (r'(\.add_pair\([^)]+\.vector, [^)]+\.vector)\n', r'\1)\n'),
        # Fix lines with query(...vector missing closing paren
        (r'(\.query\([^)]+\.vector)\n', r'\1)\n'),
        # Fix lines with lambda missing closing paren
        (r'(control\.is_goal_active\(vocab\[[^\]]+\]\.vector) \* ', r'\1) * '),
        (r'(control\.is_task_active\(vocab\[[^\]]+\]\.vector)\n', r'\1)\n'),
        (r'(control\.[^(]+\(vocab\[[^\]]+\]\.vector)\n', r'\1)\n'),
        # Fix saved_state.state = current)
        (r'saved_state\.state = current\)', r'saved_state.state = current'),
        # Fix action.state = current)
        (r'action\.state = current\)', r'action.state = current'),
        # Fix other .state = patterns
        (r'(\w+)\.state = (\w+)\)', r'\1.state = \2'),
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