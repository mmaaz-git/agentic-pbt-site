#!/usr/bin/env python3
"""Reproduce the sorted_division_locations bug"""

import numpy as np
from dask.dataframe.io.io import sorted_division_locations

print("=== Test 1: Simple reproduction case ===")
seq = np.array([0, 0])
npartitions = 2
divisions, locations = sorted_division_locations(seq, npartitions=npartitions)

print(f"Input sequence: {seq}")
print(f"Requested npartitions: {npartitions}")
print(f"Expected divisions length: {npartitions + 1}")
print(f"Actual divisions: {divisions}")
print(f"Actual divisions length: {len(divisions)}")
print(f"Locations: {locations}")
print()

print("=== Test 2: More duplicate values ===")
seq2 = np.array([5, 5, 5, 5])
npartitions2 = 3
divisions2, locations2 = sorted_division_locations(seq2, npartitions=npartitions2)
print(f"Input sequence: {seq2}")
print(f"Requested npartitions: {npartitions2}")
print(f"Expected divisions length: {npartitions2 + 1}")
print(f"Actual divisions: {divisions2}")
print(f"Actual divisions length: {len(divisions2)}")
print()

print("=== Test 3: Mixed values with duplicates ===")
seq3 = np.array([1, 1, 2, 2, 3, 3])
npartitions3 = 4
divisions3, locations3 = sorted_division_locations(seq3, npartitions=npartitions3)
print(f"Input sequence: {seq3}")
print(f"Requested npartitions: {npartitions3}")
print(f"Expected divisions length: {npartitions3 + 1}")
print(f"Actual divisions: {divisions3}")
print(f"Actual divisions length: {len(divisions3)}")
print()

print("=== Test 4: No duplicates ===")
seq4 = np.array([1, 2, 3, 4, 5])
npartitions4 = 2
divisions4, locations4 = sorted_division_locations(seq4, npartitions=npartitions4)
print(f"Input sequence: {seq4}")
print(f"Requested npartitions: {npartitions4}")
print(f"Expected divisions length: {npartitions4 + 1}")
print(f"Actual divisions: {divisions4}")
print(f"Actual divisions length: {len(divisions4)}")