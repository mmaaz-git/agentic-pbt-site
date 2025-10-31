#!/usr/bin/env python3
"""Test to reproduce the is_valid_tag bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

# First, test the manual reproduction
from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

print("Testing manual reproduction:")
print(f"is_valid_tag('.0') as str: {is_valid_tag('.0')}")
print(f"is_valid_tag('.123') as str: {is_valid_tag('.123')}")
print(f"is_valid_tag(EncodedString('.0')): {is_valid_tag(EncodedString('.0'))}")
print(f"is_valid_tag(EncodedString('.123')): {is_valid_tag(EncodedString('.123'))}")

# Now test with hypothesis
print("\nTesting with Hypothesis:")
try:
    from hypothesis import given, strategies as st

    @given(st.integers(min_value=0, max_value=999999))
    def test_is_valid_tag_decimal_pattern(n):
        name = f".{n}"
        result = is_valid_tag(name)
        assert result is False, f"Failed for input: {name}, got {result}"

    # Run the test
    test_is_valid_tag_decimal_pattern()
    print("Hypothesis test completed - no failures detected")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
except Exception as e:
    print(f"Test failed with exception: {e}")