#!/usr/bin/env python3
"""Test the reported bug in dask.dataframe.utils.valid_divisions"""

from hypothesis import given, strategies as st
from dask.dataframe.utils import valid_divisions
import traceback

print("=" * 60)
print("Testing valid_divisions with small inputs")
print("=" * 60)

# Test 1: Empty list
print("\n1. Testing valid_divisions([]):")
try:
    result = valid_divisions([])
    print(f"   Result: {result}")
    print(f"   Type: {type(result)}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 2: Single element list
print("\n2. Testing valid_divisions([1]):")
try:
    result = valid_divisions([1])
    print(f"   Result: {result}")
    print(f"   Type: {type(result)}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 3: Two element list (should work)
print("\n3. Testing valid_divisions([1, 2]):")
try:
    result = valid_divisions([1, 2])
    print(f"   Result: {result}")
    print(f"   Type: {type(result)}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

# Test 4: Property-based test from the bug report
print("\n4. Running property-based test:")
@given(st.lists(st.integers(), max_size=1))
def test_valid_divisions_small_lists(divisions):
    try:
        result = valid_divisions(divisions)
        assert isinstance(result, bool), f"Should return bool, got {type(result)}"
        print(f"   PASS for {divisions}: returned {result}")
    except IndexError as e:
        print(f"   FAIL for {divisions}: IndexError - {e}")
        raise AssertionError(f"Should not crash on {divisions}")
    except Exception as e:
        print(f"   UNEXPECTED ERROR for {divisions}: {type(e).__name__} - {e}")
        raise

# Run a few examples from the property test
try:
    test_valid_divisions_small_lists([])
except AssertionError as e:
    print(f"   Property test failed: {e}")

try:
    test_valid_divisions_small_lists([42])
except AssertionError as e:
    print(f"   Property test failed: {e}")

print("\n" + "=" * 60)
print("Analysis of the crash location")
print("=" * 60)

# Let's look at what line causes the crash
print("\nThe function crashes at line 715:")
print("    return divisions[-2] <= divisions[-1]")
print("\nFor an empty list []:")
print("  - divisions[-2] tries to access element at index -2")
print("  - Empty list has no elements, so IndexError")
print("\nFor a single-element list [1]:")
print("  - divisions[-2] tries to access element at index -2")
print("  - List [1] only has indices 0 and -1, so IndexError")