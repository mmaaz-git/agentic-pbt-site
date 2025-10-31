#!/usr/bin/env python3
"""Reproduce the reported bug in sorted_division_locations"""

import pandas as pd
from dask.dataframe.io.io import sorted_division_locations

print("=== Reproducing the Bug ===")
print()

# Test Case 1: The specific failing example from the bug report
print("Test Case 1: seq=pd.Series([0, -1]), chunksize=1")
print("-" * 50)
seq = pd.Series([0, -1])
divisions, locations = sorted_division_locations(seq, chunksize=1)

print(f"Input: {seq.tolist()}")
print(f"Divisions returned: {divisions}")
print(f"Expected (sorted): {sorted(divisions)}")
print(f"Are divisions sorted? {divisions == sorted(divisions)}")
print()

# Test Case 2: Let's try with a numpy array
print("Test Case 2: seq=np.array([0, -1]), chunksize=1")
print("-" * 50)
import numpy as np
seq2 = np.array([0, -1])
try:
    divisions2, locations2 = sorted_division_locations(seq2, chunksize=1)
    print(f"Input: {seq2.tolist()}")
    print(f"Divisions returned: {divisions2}")
    print(f"Expected (sorted): {sorted(divisions2)}")
    print(f"Are divisions sorted? {divisions2 == sorted(divisions2)}")
except Exception as e:
    print(f"Failed with error: {e}")
print()

# Test Case 3: Let's test with sorted input (control case)
print("Test Case 3: seq=pd.Series([-1, 0]), chunksize=1 (sorted input)")
print("-" * 50)
seq3 = pd.Series([-1, 0])
divisions3, locations3 = sorted_division_locations(seq3, chunksize=1)

print(f"Input: {seq3.tolist()}")
print(f"Divisions returned: {divisions3}")
print(f"Expected (sorted): {sorted(divisions3)}")
print(f"Are divisions sorted? {divisions3 == sorted(divisions3)}")
print()

# Test Case 4: More complex unsorted case
print("Test Case 4: seq=[5, 3, 1, 4, 2], chunksize=2")
print("-" * 50)
seq4 = [5, 3, 1, 4, 2]
divisions4, locations4 = sorted_division_locations(seq4, chunksize=2)

print(f"Input: {seq4}")
print(f"Divisions returned: {divisions4}")
print(f"Locations returned: {locations4}")
print(f"Expected (sorted): {sorted(divisions4)}")
print(f"Are divisions sorted? {divisions4 == sorted(divisions4)}")
print()

# Test Case 5: Run the hypothesis test example
print("Test Case 5: Running Hypothesis Test")
print("-" * 50)
from hypothesis import given, strategies as st, settings
import pandas as pd

failed_cases = []

@given(
    seq=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1).map(pd.Series),
    chunksize=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=100)
def test_sorted_division_locations_divisions_sorted(seq, chunksize):
    try:
        divisions, locations = sorted_division_locations(seq, chunksize=chunksize)
        if divisions != sorted(divisions):
            failed_cases.append({
                'seq': seq.tolist(),
                'chunksize': chunksize,
                'divisions': divisions,
                'sorted_divisions': sorted(divisions)
            })
    except Exception as e:
        print(f"Exception: {e}")

# Run the test
try:
    test_sorted_division_locations_divisions_sorted()
    if failed_cases:
        print(f"Found {len(failed_cases)} failing cases:")
        for i, case in enumerate(failed_cases[:5], 1):  # Show first 5
            print(f"  Case {i}: seq={case['seq'][:10]}{'...' if len(case['seq']) > 10 else ''}, "
                  f"chunksize={case['chunksize']}")
            print(f"    Divisions: {case['divisions'][:10]}{'...' if len(case['divisions']) > 10 else ''}")
            print(f"    Expected:  {case['sorted_divisions'][:10]}{'...' if len(case['sorted_divisions']) > 10 else ''}")
    else:
        print("No failures found in hypothesis test (100 examples)")
except Exception as e:
    print(f"Hypothesis test failed with error: {e}")

print()
print("=== Reproduction Complete ===")