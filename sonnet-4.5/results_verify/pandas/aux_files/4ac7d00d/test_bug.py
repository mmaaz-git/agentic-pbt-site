#!/usr/bin/env python3
"""Test the reported bug about SparseArray.density division by zero"""

from pandas.arrays import SparseArray
from hypothesis import given, strategies as st
import math
import traceback

# First, test the hypothesis test from the bug report
@st.composite
def sparse_arrays(draw, min_size=0, max_size=100):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    data = draw(st.lists(st.integers(), min_size=size, max_size=size))
    return SparseArray(data)

@given(sparse_arrays())
def test_density_property(arr):
    expected_density = arr.npoints / arr.sp_index.length if arr.sp_index.length > 0 else 0.0
    assert math.isclose(arr.density, expected_density)

print("Testing with hypothesis...")
try:
    test_density_property()
    print("Hypothesis test passed")
except Exception as e:
    print(f"Hypothesis test failed: {e}")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Now test the specific failing case
print("Testing the specific failing case: SparseArray([])")
try:
    empty_arr = SparseArray([])
    print(f"Created empty SparseArray: {empty_arr}")
    print(f"Empty array length: {len(empty_arr)}")
    print(f"Empty array sp_index.length: {empty_arr.sp_index.length}")
    print(f"Empty array sp_index.npoints: {empty_arr.sp_index.npoints}")

    # This should raise ZeroDivisionError according to the bug report
    density = empty_arr.density
    print(f"Empty array density: {density}")
except ZeroDivisionError as e:
    print(f"✓ ZeroDivisionError raised as reported: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Different error raised: {e}")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test with other edge cases
print("Testing other edge cases...")

# Test with single element
try:
    single = SparseArray([1])
    print(f"Single element array density: {single.density}")
except Exception as e:
    print(f"Single element failed: {e}")

# Test with all fill values
try:
    all_zeros = SparseArray([0, 0, 0, 0])
    print(f"All zeros array density: {all_zeros.density}")
except Exception as e:
    print(f"All zeros failed: {e}")

# Test with no fill values
try:
    no_zeros = SparseArray([1, 2, 3, 4])
    print(f"No zeros array density: {no_zeros.density}")
except Exception as e:
    print(f"No zeros failed: {e}")

print("\n" + "="*50 + "\n")

# Let's verify what density means mathematically
print("Understanding density calculation:")
test_arr = SparseArray([0, 0, 1, 1, 1], fill_value=0)
print(f"Test array: {test_arr}")
print(f"sp_index.npoints (non-fill values): {test_arr.sp_index.npoints}")
print(f"sp_index.length (total length): {test_arr.sp_index.length}")
print(f"Density (npoints/length): {test_arr.sp_index.npoints}/{test_arr.sp_index.length} = {test_arr.density}")
print(f"This matches the documentation example: 3/5 = 0.6")