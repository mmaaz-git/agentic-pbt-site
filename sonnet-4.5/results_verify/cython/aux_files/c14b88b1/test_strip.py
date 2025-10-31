#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Dependencies import strip_string_literals

# Test strip_string_literals with incomplete quotes
test_cases = [
    "'",
    "['']",
    "[']",
    '"',
    '[""]',
    '["]]',
    "'hello",
    '"hello',
    "normal text",
    "'complete string'",
    '"complete string"',
]

for test in test_cases:
    print(f"\nInput: {repr(test)}")
    try:
        result, literals = strip_string_literals(test)
        print(f"  Result: {repr(result)}")
        print(f"  Literals: {literals}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")