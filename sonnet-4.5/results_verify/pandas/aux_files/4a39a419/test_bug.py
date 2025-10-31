#!/usr/bin/env python3
"""Test script to reproduce the reported bug in nested_to_record"""

from pandas.io.json._normalize import nested_to_record
import traceback

print("Testing pandas.io.json._normalize.nested_to_record with non-string keys")
print("=" * 70)

# Test 1: Simple nested dict with integer keys
print("\nTest 1: Simple nested dict {1: {2: 'value'}}")
try:
    d = {1: {2: 'value'}}
    result = nested_to_record(d)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Failed with {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 2: Mixed dict with nested integer keys
print("\n" + "=" * 70)
print("Test 2: Mixed dict {1: 'a', 2: 'b', 3: {4: 'c', 5: 'd'}}")
try:
    d = {1: 'a', 2: 'b', 3: {4: 'c', 5: 'd'}}
    result = nested_to_record(d)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Failed with {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 3: Dict with string keys (should work)
print("\n" + "=" * 70)
print("Test 3: Normal dict with string keys {'a': {'b': 'value'}}")
try:
    d = {'a': {'b': 'value'}}
    result = nested_to_record(d)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Failed with {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 4: Dict with integer keys but no nesting
print("\n" + "=" * 70)
print("Test 4: Flat dict with integer keys {1: 'a', 2: 'b', 3: 'c'}")
try:
    d = {1: 'a', 2: 'b', 3: 'c'}
    result = nested_to_record(d)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Failed with {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 5: Run the hypothesis test
print("\n" + "=" * 70)
print("Test 5: Running property-based test from bug report")
try:
    from hypothesis import given, strategies as st

    @given(st.dictionaries(
        keys=st.integers(),
        values=st.one_of(
            st.text(),
            st.dictionaries(keys=st.integers(), values=st.text(), max_size=3)
        ),
        max_size=5
    ))
    def test_nested_to_record_handles_non_string_keys(d):
        result = nested_to_record(d)
        assert isinstance(result, dict)
        return True

    # Run a few examples
    test_nested_to_record_handles_non_string_keys()
    print("Hypothesis test completed successfully!")

except Exception as e:
    print(f"Hypothesis test failed: {e}")
    traceback.print_exc()