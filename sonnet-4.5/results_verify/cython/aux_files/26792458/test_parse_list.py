#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Dependencies import parse_list

# Test various edge cases
test_cases = [
    '[""]',  # Empty string in quotes
    '["a"]',  # Single quoted string
    '[a, ""]',  # Mix
    '["\'"]',  # Quote inside quotes
    '["\\"a\\""]',  # Escaped quotes
    '["]',  # The bug report case - unclosed quote
    '["\'"]',  # Single quote in double quotes
]

for test in test_cases:
    try:
        result = parse_list(test)
        print(f"{test:20} -> {result}")
    except Exception as e:
        print(f"{test:20} -> ERROR: {type(e).__name__}: {e}")