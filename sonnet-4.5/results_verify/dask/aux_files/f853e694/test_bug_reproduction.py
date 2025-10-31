#!/usr/bin/env python3
"""Test to reproduce the sorted_division_locations bug"""

import sys
import traceback
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st

print("=" * 60)
print("Testing sorted_division_locations with different input types")
print("=" * 60)

# First, let's test the basic bug claim
from dask.dataframe.io.io import sorted_division_locations

# Test 1: Plain Python list (as shown in docstring examples)
print("\n1. Testing with plain Python list (as shown in docstring):")
print("Code: L = ['A', 'B', 'C', 'D', 'E', 'F']")
print("Code: sorted_division_locations(L, chunksize=2)")

L = ['A', 'B', 'C', 'D', 'E', 'F']
try:
    result = sorted_division_locations(L, chunksize=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 2: NumPy array (should work)
print("\n2. Testing with NumPy array:")
print("Code: L_np = np.array(['A', 'B', 'C', 'D', 'E', 'F'])")
print("Code: sorted_division_locations(L_np, chunksize=2)")

L_np = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
try:
    result = sorted_division_locations(L_np, chunksize=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Test 3: Pandas Series (should work)
print("\n3. Testing with Pandas Series:")
print("Code: L_pd = pd.Series(['A', 'B', 'C', 'D', 'E', 'F'])")
print("Code: sorted_division_locations(L_pd, chunksize=2)")

L_pd = pd.Series(['A', 'B', 'C', 'D', 'E', 'F'])
try:
    result = sorted_division_locations(L_pd, chunksize=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Test 4: Run the hypothesis test from the bug report
print("\n4. Running the Hypothesis test from the bug report:")
print("Code: Test that sorted_division_locations accepts lists as documented")

@given(seq=st.lists(st.text(min_size=1, max_size=1), min_size=1, max_size=10))
def test_sorted_division_locations_accepts_lists_as_documented(seq):
    seq_sorted = sorted(seq)
    divisions, locations = sorted_division_locations(seq_sorted, chunksize=2)

    assert divisions[0] == seq_sorted[0]
    assert divisions[-1] == seq_sorted[-1]

try:
    test_sorted_division_locations_accepts_lists_as_documented()
    print("Hypothesis test PASSED")
except Exception as e:
    print(f"Hypothesis test FAILED: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Summary:")
print("The function's docstring shows examples using plain Python lists,")
print("but the implementation fails when given a list.")
print("=" * 60)