#!/usr/bin/env python3
"""Test script to reproduce the reported bug"""

import pandas.core.indexers as indexers

# Test 1: Basic reproduction of the bug
target = []
slc = slice(None, -1, None)

print("Test 1: Slicing empty list with negative stop")
print(f"target = {target}")
print(f"slc = {slc}")
print(f"target[slc] = {target[slc]}")
print(f"len(target[slc]) = {len(target[slc])}")
print(f"length_of_indexer(slc, target) = {indexers.length_of_indexer(slc, target)}")
print()

# Test 2: Try with a non-empty list
target2 = [0, 1, 2, 3, 4]
slc2 = slice(None, -1, None)

print("Test 2: Slicing non-empty list with negative stop")
print(f"target = {target2}")
print(f"slc = {slc2}")
print(f"target[slc] = {target2[slc2]}")
print(f"len(target[slc]) = {len(target2[slc2])}")
print(f"length_of_indexer(slc, target) = {indexers.length_of_indexer(slc2, target2)}")
print()

# Test 3: Various slices on empty list
empty_list = []
test_slices = [
    slice(None, -1, None),
    slice(None, -2, None),
    slice(None, -5, None),
    slice(None, 0, None),
    slice(None, 1, None),
    slice(1, 3, None),
    slice(-1, None, None),
]

print("Test 3: Various slices on empty list")
for slc in test_slices:
    expected = len(empty_list[slc])
    calculated = indexers.length_of_indexer(slc, empty_list)
    match = "✓" if expected == calculated else "✗"
    print(f"{match} slice{slc}: expected={expected}, calculated={calculated}")