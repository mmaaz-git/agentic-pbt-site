#!/usr/bin/env python3
"""Test script to reproduce the bug reported in dask.diagnostics.profile_visualize.unquote"""

from hypothesis import given, strategies as st
from dask.diagnostics.profile_visualize import unquote

# First, let's test the Hypothesis-based property test
@given(
    items=st.lists(st.tuples(st.text(), st.integers()), min_size=0, max_size=5)
)
def test_unquote_handles_dict(items):
    expr = (dict, [items])
    result = unquote(expr)
    assert isinstance(result, dict)

# Test with the specific failing input mentioned
print("Testing Hypothesis property test with empty list:")
try:
    test_unquote_handles_dict()
    print("Hypothesis test passed (unexpected)")
except Exception as e:
    print(f"Hypothesis test failed with: {type(e).__name__}: {e}")

# Now test the specific reproducing cases
print("\nTesting direct reproduction cases:")

print("\n1. Testing unquote((dict, [])):")
try:
    result = unquote((dict, []))
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError raised: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

print("\n2. Testing unquote((dict, [[]])):")
try:
    result = unquote((dict, [[]]))
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

# Let's also test some working cases to understand the function
print("\n3. Testing valid dictionary representations:")
try:
    # Test with a valid dictionary representation
    result = unquote((dict, [[('key1', 'value1'), ('key2', 'value2')]]))
    print(f"Valid dict result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n4. Testing other collection types:")
try:
    # Test list
    result = unquote((list, [[1, 2, 3]]))
    print(f"List result: {result}")

    # Test tuple
    result = unquote((tuple, [[1, 2, 3]]))
    print(f"Tuple result: {result}")

    # Test set
    result = unquote((set, [[1, 2, 3]]))
    print(f"Set result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")