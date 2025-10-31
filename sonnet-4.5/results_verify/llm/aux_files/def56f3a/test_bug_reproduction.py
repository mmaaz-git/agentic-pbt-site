#!/usr/bin/env python3
"""Test script to reproduce the not_nulls bug"""

import sys
import traceback

# Add the package path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

print("=" * 60)
print("TEST 1: Reproducing the bug with a regular dict")
print("=" * 60)

try:
    from llm.default_plugins.openai_models import not_nulls

    # Test with the example from the bug report
    data = {"temperature": 0.7, "max_tokens": None, "seed": 42}
    print(f"Input data: {data}")
    result = not_nulls(data)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 2: Running the hypothesis test")
print("=" * 60)

try:
    from hypothesis import given, strategies as st
    from llm.default_plugins.openai_models import not_nulls

    @given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
    def test_not_nulls_with_dict(data):
        """Property: not_nulls should filter out None values from a dict"""
        result = not_nulls(data)
        assert all(v is not None for v in result.values())

    # Run the test with the specific failing input
    failing_input = {"a": 1, "b": None}
    print(f"Testing with failing input: {failing_input}")
    result = not_nulls(failing_input)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 3: Testing how the function actually gets used")
print("=" * 60)

# Let me check how it's actually used in the codebase
print("Checking line 658 where not_nulls is used...")