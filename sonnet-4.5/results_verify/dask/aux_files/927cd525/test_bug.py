#!/usr/bin/env python3
"""Test the reported bug with sorted_division_locations"""

# First, let's try the exact example from the docstring
from dask.dataframe.io.io import sorted_division_locations

print("Testing example from docstring:")
try:
    L = ['A', 'B', 'C', 'D', 'E', 'F']
    result = sorted_division_locations(L, chunksize=2)
    print(f"SUCCESS: {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\nTesting with integer list:")
try:
    L = [1, 2, 3, 4, 5, 6]
    result = sorted_division_locations(L, chunksize=2)
    print(f"SUCCESS: {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\nTesting minimal case from bug report:")
try:
    result = sorted_division_locations([0], chunksize=1)
    print(f"SUCCESS: {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\nTesting the hypothesis property test:")
from hypothesis import given, strategies as st

@given(
    seq=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=50)
)
def test_sorted_division_locations_with_lists(seq, chunksize):
    seq_sorted = sorted(seq)
    divisions, locations = sorted_division_locations(seq_sorted, chunksize=chunksize)
    assert divisions[0] == seq_sorted[0]
    assert divisions[-1] == seq_sorted[-1]
    return True

try:
    test_sorted_division_locations_with_lists()
    print("Hypothesis test PASSED")
except Exception as e:
    print(f"Hypothesis test FAILED: {type(e).__name__}: {e}")