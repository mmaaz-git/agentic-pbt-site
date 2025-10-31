#!/usr/bin/env python3
"""Test to reproduce the xarray broadcast bug with string exclude parameter."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/results/xarray')

from hypothesis import given, strategies as st, assume
import numpy as np
import xarray as xr
from xarray.structure.alignment import broadcast

print("=" * 70)
print("Testing xarray.structure.alignment.broadcast() String Exclude Parameter")
print("=" * 70)

# First, run the hypothesis test
print("\n1. Running Hypothesis property test...")
print("-" * 50)

test_passed = True

@given(
    dim_name=st.text(min_size=2, max_size=5, alphabet=st.characters(whitelist_categories=('Ll',))),
    other_dim=st.text(min_size=1, max_size=1, alphabet=st.characters(whitelist_categories=('Ll',)))
)
def test_broadcast_exclude_string_exact_match(dim_name, other_dim):
    """Property: When exclude is a string dimension name, only that exact dimension should be excluded."""
    global test_passed
    assume(other_dim not in dim_name)
    assume(dim_name != other_dim)

    da1 = xr.DataArray([1, 2], dims=[dim_name])
    da2 = xr.DataArray([3, 4, 5], dims=[other_dim])

    result1, result2 = broadcast(da1, da2, exclude=dim_name)

    if other_dim not in result1.dims:
        print(f"BUG FOUND: '{other_dim}' was excluded when exclude='{dim_name}'")
        print(f"  dim_name='{dim_name}', other_dim='{other_dim}'")
        print(f"  Character '{other_dim}' is {'IN' if other_dim in dim_name else 'NOT IN'} string '{dim_name}'")
        test_passed = False

try:
    test_broadcast_exclude_string_exact_match()
    if test_passed:
        print("Hypothesis test PASSED - no bug found")
except Exception as e:
    print(f"Hypothesis test failed with error: {e}")

print("\n2. Running specific reproduction case...")
print("-" * 50)

# Now run the specific reproduction case
da_x = xr.DataArray([1, 2, 3], dims=['x'])
da_y = xr.DataArray([4, 5], dims=['y'])

print(f"Input data arrays:")
print(f"  da_x dims: {da_x.dims}")
print(f"  da_y dims: {da_y.dims}")
print(f"  exclude parameter: 'xy'")

result_x, result_y = broadcast(da_x, da_y, exclude='xy')

print(f"\nResult after broadcast with exclude='xy':")
print(f"  result_x dims: {result_x.dims}")
print(f"  result_y dims: {result_y.dims}")

print("\nChecking expected behavior:")
# Expected: Only dimension 'xy' (which doesn't exist) should be excluded
# Dimensions 'x' and 'y' should be broadcast normally
expected_x_in_result_y = True
expected_y_in_result_x = True

actual_x_in_result_y = 'x' in result_y.dims
actual_y_in_result_x = 'y' in result_x.dims

print(f"  Dimension 'x' in result_y: expected={expected_x_in_result_y}, actual={actual_x_in_result_y}")
print(f"  Dimension 'y' in result_x: expected={expected_y_in_result_x}, actual={actual_y_in_result_x}")

if actual_x_in_result_y != expected_x_in_result_y or actual_y_in_result_x != expected_y_in_result_x:
    print("\nBUG CONFIRMED: Dimensions 'x' and 'y' were incorrectly excluded!")
    print("This happens because Python's 'in' operator treats string 'xy' as a sequence")
    print("So 'x' in 'xy' returns True, and 'y' in 'xy' returns True")
else:
    print("\nNo bug found - dimensions were broadcast correctly")

print("\n3. Demonstrating the root cause...")
print("-" * 50)
print(f"When exclude='xy' is passed as a string:")
print(f"  'x' in 'xy' = {'x' in 'xy'}")
print(f"  'y' in 'xy' = {'y' in 'xy'}")
print(f"  'z' in 'xy' = {'z' in 'xy'}")
print(f"This is why both 'x' and 'y' dimensions get excluded incorrectly")