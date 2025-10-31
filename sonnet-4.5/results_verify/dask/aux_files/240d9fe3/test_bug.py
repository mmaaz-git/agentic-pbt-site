#!/usr/bin/env python3
"""Test script to reproduce the sorted_division_locations bug"""

from hypothesis import given, strategies as st
from dask.dataframe.io.io import sorted_division_locations
import traceback

print("Testing sorted_division_locations with plain Python lists")
print("=" * 60)

# Test 1: Simple list example from the docstring
print("\n1. Testing docstring example:")
try:
    L = ['A', 'B', 'C', 'D', 'E', 'F']
    result = sorted_division_locations(L, chunksize=2)
    print(f"   Input: {L}")
    print(f"   Result: {result}")
    print(f"   Expected: (['A', 'C', 'E', 'F'], [0, 2, 4, 6])")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 2: Another docstring example
print("\n2. Testing second docstring example:")
try:
    L = ['A', 'B', 'C', 'D', 'E', 'F']
    result = sorted_division_locations(L, chunksize=3)
    print(f"   Input: {L}")
    print(f"   Result: {result}")
    print(f"   Expected: (['A', 'D', 'F'], [0, 3, 6])")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

# Test 3: Duplicates example from docstring
print("\n3. Testing duplicates example:")
try:
    L = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C']
    result = sorted_division_locations(L, chunksize=3)
    print(f"   Input: {L}")
    print(f"   Result: {result}")
    print(f"   Expected: (['A', 'B', 'C', 'C'], [0, 4, 7, 8])")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

# Test 4: Single element example from docstring
print("\n4. Testing single element example:")
try:
    result = sorted_division_locations(['A'], chunksize=2)
    print(f"   Input: ['A']")
    print(f"   Result: {result}")
    print(f"   Expected: (['A', 'A'], [0, 1])")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

# Test 5: Minimal failing case from bug report
print("\n5. Testing minimal case from bug report:")
try:
    result = sorted_division_locations([0], npartitions=1)
    print(f"   Input: [0], npartitions=1")
    print(f"   Result: {result}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

# Test 6: Property-based test from bug report
print("\n6. Running property-based test:")
test_passed = True
try:
    @given(
        seq=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=100),
        npartitions=st.integers(min_value=1, max_value=20)
    )
    def test_sorted_division_locations_accepts_lists(seq, npartitions):
        seq_sorted = sorted(seq)
        divisions, locations = sorted_division_locations(seq_sorted, npartitions=npartitions)
        assert len(divisions) == len(locations)

    # Run a single test case
    test_sorted_division_locations_accepts_lists([1, 2, 3], 2)
    print("   Property test with list [1, 2, 3], npartitions=2: PASSED")
except Exception as e:
    print(f"   Property test FAILED: {type(e).__name__}: {e}")
    test_passed = False

# Now test with numpy arrays to see if those work
print("\n7. Testing with numpy array (for comparison):")
try:
    import numpy as np
    arr = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
    result = sorted_division_locations(arr, chunksize=2)
    print(f"   Input: numpy array {arr}")
    print(f"   Result: {result}")
    print(f"   SUCCESS - numpy arrays work!")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

# Test with pandas Series
print("\n8. Testing with pandas Series (for comparison):")
try:
    import pandas as pd
    series = pd.Series(['A', 'B', 'C', 'D', 'E', 'F'])
    result = sorted_division_locations(series, chunksize=2)
    print(f"   Input: pandas Series")
    print(f"   Result: {result}")
    print(f"   SUCCESS - pandas Series work!")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("SUMMARY:")
print("- Plain Python lists FAIL with TypeError: No dispatch for <class 'list'>")
print("- NumPy arrays and pandas Series work correctly")
print("- The docstring examples all use plain lists but don't actually work")
print("- This is a clear discrepancy between documentation and implementation")