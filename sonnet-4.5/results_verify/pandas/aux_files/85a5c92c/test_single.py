#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.api import types
import re

def test_specific_pattern(pattern_str):
    """Test that is_re_compilable should never raise an exception"""
    print(f"Testing pattern: '{pattern_str}'")

    try:
        result = types.is_re_compilable(pattern_str)
        print(f"  Result: {result}")
        assert isinstance(result, bool), f"Result should be bool, got {type(result)}"

        # If it returns True, we should be able to compile it
        if result:
            re.compile(pattern_str)
            print(f"  Pattern compiles successfully")
        return True
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False

# Test specific failing cases
test_cases = ['?', '*', '+', '\\', '[', '(', ')']

print("Testing specific patterns that should NOT crash:")
print("-" * 50)

failures = []
for pattern in test_cases:
    if not test_specific_pattern(pattern):
        failures.append(pattern)

if failures:
    print(f"\n{len(failures)} patterns crashed when they should have returned False:")
    for p in failures:
        print(f"  '{p}'")
else:
    print("\nAll patterns handled correctly (no crashes)")