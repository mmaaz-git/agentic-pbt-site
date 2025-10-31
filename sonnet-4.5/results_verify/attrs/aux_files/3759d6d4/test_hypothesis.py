#!/usr/bin/env python3
"""Property-based testing for attrs.converters.to_bool"""

from hypothesis import given, strategies as st, settings
import attrs
from attrs import converters

# Test 1: to_bool should reject most floats
@given(st.floats().filter(lambda x: x not in [1.0, 0.0]))
@settings(max_examples=100)
def test_to_bool_rejects_undocumented_floats(x):
    """Test that to_bool rejects floats other than 1.0 and 0.0"""
    try:
        result = converters.to_bool(x)
        print(f"UNEXPECTED: to_bool({x}) = {result} (should have raised ValueError)")
        assert False, f"to_bool should reject float {x}"
    except ValueError:
        pass  # Expected behavior

# Test 2: to_bool accepts 1.0 and 0.0
@given(st.sampled_from([1.0, 0.0]))
def test_to_bool_accepts_some_floats(x):
    """Test that to_bool accepts 1.0 and 0.0"""
    result = converters.to_bool(x)
    expected = (x == 1.0)
    assert result == expected, f"to_bool({x}) returned {result}, expected {expected}"
    print(f"to_bool({x}) = {result} (as expected)")

if __name__ == "__main__":
    print("Running property-based tests...")
    print("\nTest 1: Testing that most floats are rejected")
    test_to_bool_rejects_undocumented_floats()

    print("\nTest 2: Testing that 1.0 and 0.0 are accepted")
    test_to_bool_accepts_some_floats()

    print("\nAll property tests completed successfully!")