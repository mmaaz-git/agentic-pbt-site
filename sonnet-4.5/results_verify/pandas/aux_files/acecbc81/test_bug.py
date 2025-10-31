#!/usr/bin/env python3
"""Test script to reproduce the reported bug with surrogate characters"""

import numpy as np
import sys
import traceback

# First, let's understand what surrogate characters are
print("Testing surrogate character handling in pandas.hash_array")
print("=" * 60)

# Check Python's handling of surrogate characters
surrogate_char = '\ud800'
print(f"Python string with surrogate: repr={repr(surrogate_char)}")
print(f"Length of string: {len(surrogate_char)}")
print(f"Is valid Python string: {isinstance(surrogate_char, str)}")
print()

# Try to encode it to UTF-8
print("Attempting to encode surrogate to UTF-8:")
try:
    encoded = surrogate_char.encode('utf-8')
    print(f"Success: {encoded}")
except UnicodeEncodeError as e:
    print(f"Failed with UnicodeEncodeError: {e}")
print()

# Now test with pandas
print("Testing pandas.hash_array with surrogate character:")
print("-" * 40)

from pandas.core.util.hashing import hash_array

# Test 1: Simple surrogate character
print("\nTest 1: Array with single surrogate character '\\ud800'")
try:
    arr = np.array(['\ud800'], dtype=object)
    print(f"Created numpy array: {arr}")
    print(f"Array dtype: {arr.dtype}")
    result = hash_array(arr)
    print(f"Hash result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 2: Mixed array with normal and surrogate characters
print("\nTest 2: Mixed array with normal string and surrogate")
try:
    arr = np.array(['hello', '\ud800'], dtype=object)
    result = hash_array(arr)
    print(f"Hash result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 3: Multiple different surrogates
print("\nTest 3: Multiple different surrogate characters")
try:
    arr = np.array(['\ud800', '\udfff', '\udc00'], dtype=object)
    result = hash_array(arr)
    print(f"Hash result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 4: Test with categorize=False
print("\nTest 4: Same test with categorize=False")
try:
    arr = np.array(['\ud800'], dtype=object)
    result = hash_array(arr, categorize=False)
    print(f"Hash result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test the Hypothesis test from the bug report
print("\n" + "=" * 60)
print("Running the Hypothesis test from bug report:")
print("-" * 40)

from hypothesis import given, strategies as st

@given(st.lists(st.text(max_size=100, alphabet=st.characters(min_codepoint=128, max_codepoint=0x10FFFF)), min_size=1, max_size=50))
def test_hash_array_unicode_strings(values):
    arr = np.array(values, dtype=object)
    hash1 = hash_array(arr)
    hash2 = hash_array(arr)
    assert np.array_equal(hash1, hash2)

print("\nTesting with the specific failing input from bug report: values=['\\ud800']")
try:
    test_hash_array_unicode_strings(['\ud800'])
    print("Test passed!")
except Exception as e:
    print(f"Test failed: {type(e).__name__}: {e}")
    traceback.print_exc()