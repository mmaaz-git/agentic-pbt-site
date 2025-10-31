#!/usr/bin/env python3
"""Test how Python's built-in slicing handles edge cases"""

# Test 1: Empty list with various slice parameters
empty = []

test_cases = [
    (None, -1, None),
    (None, -2, None),
    (None, 0, None),
    (None, 1, None),
    (0, -1, None),
    (0, -2, None),
    (1, 3, None),
    (1, None, None),
    (-1, None, None),
    (5, None, None),
    (5, 10, None),
    (-5, -3, None),
]

print("Python's slice behavior on empty list:")
print("=" * 50)
for start, stop, step in test_cases:
    slc = slice(start, stop, step)
    result = empty[slc]
    print(f"[]slice({start}, {stop}, {step}) = {result}, len = {len(result)}")

print("\n\nAnalyzing slice indices conversion for empty list:")
print("=" * 50)
for start, stop, step in test_cases:
    slc = slice(start, stop, step)
    # Python's built-in slice.indices method returns (start, stop, step)
    # adjusted for the given sequence length
    normalized = slc.indices(len(empty))
    print(f"slice({start}, {stop}, {step}).indices(0) = {normalized}")

print("\n\nComparing with pandas length_of_indexer:")
print("=" * 50)
import pandas.core.indexers as indexers
for start, stop, step in test_cases:
    slc = slice(start, stop, step)
    expected = len(empty[slc])
    calculated = indexers.length_of_indexer(slc, empty)
    match = "✓" if expected == calculated else "✗"
    print(f"{match} slice({start}, {stop}, {step}): expected={expected}, calculated={calculated}")