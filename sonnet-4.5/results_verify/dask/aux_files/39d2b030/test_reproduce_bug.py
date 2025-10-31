#!/usr/bin/env python3
"""Test script to reproduce the reported bug in dask.array.slicing.check_index"""

from hypothesis import given, strategies as st
import numpy as np
from dask.array.slicing import check_index

# First, let's run the simple reproduction case
print("=== Simple Reproduction Case ===")
bool_array = np.array([True, True, True])
dimension = 1

try:
    check_index(0, bool_array, dimension)
except IndexError as e:
    print(f"Error raised: {e}")
    print(f"Array size: {bool_array.size}, Dimension size: {dimension}")
    print(f"Array is {'too long' if bool_array.size > dimension else 'too short' if bool_array.size < dimension else 'correct size'}")
    print()

# Now let's run the property-based test
print("=== Property-Based Test ===")

@given(st.integers(min_value=1, max_value=100))
def test_check_index_error_message_accuracy(dim_size):
    too_long_array = np.array([True] * (dim_size + 1))

    try:
        check_index(0, too_long_array, dim_size)
        assert False, "Should have raised IndexError"
    except IndexError as e:
        error_msg = str(e)

        if "not long enough" in error_msg and too_long_array.size > dim_size:
            raise AssertionError(
                f"Error message says 'not long enough' but array size "
                f"{too_long_array.size} is greater than dimension {dim_size}"
            )

# Run the property-based test
try:
    test_check_index_error_message_accuracy()
    print("Property-based test failed to find the bug (unexpected)")
except AssertionError as e:
    print(f"Property-based test found the bug: {e}")

# Let's also test the case where the array is actually too short
print("\n=== Testing when array is actually too short ===")
short_array = np.array([True])
dimension = 3

try:
    check_index(0, short_array, dimension)
except IndexError as e:
    print(f"Error raised: {e}")
    print(f"Array size: {short_array.size}, Dimension size: {dimension}")
    print(f"Array is {'too long' if short_array.size > dimension else 'too short' if short_array.size < dimension else 'correct size'}")