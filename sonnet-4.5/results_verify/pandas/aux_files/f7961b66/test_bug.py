#!/usr/bin/env python3
"""Reproduce the reported bug in pandas.core.indexers.length_of_indexer"""

import numpy as np
from pandas.core.indexers import length_of_indexer

print("=" * 60)
print("Testing pandas.core.indexers.length_of_indexer")
print("=" * 60)

# Test the specific failing case from the bug report
print("\n1. Testing the specific failing case from bug report:")
target = np.arange(0)
slc = slice(1, None, None)

computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])

print(f"Target: np.arange(0) - empty array")
print(f"Slice: slice(1, None, None)")
print(f"Computed length (from length_of_indexer): {computed_length}")
print(f"Actual length (from len(target[slc])): {actual_length}")
print(f"Match: {computed_length == actual_length}")
print(f"Bug confirmed: {computed_length} != {actual_length}" if computed_length != actual_length else "No bug")

# Test all the additional test cases from the bug report
print("\n2. Testing additional cases from bug report:")
test_cases = [
    (0, slice(1, None), "Empty array, start=1"),
    (0, slice(5, 10), "Empty array, start=5, stop=10"),
    (3, slice(5, None), "Array[3], start=5"),
    (5, slice(3, 2), "Array[5], start=3, stop=2"),
]

for target_len, slc, desc in test_cases:
    target = np.arange(target_len)
    computed = length_of_indexer(slc, target)
    actual = len(target[slc])
    print(f"{desc}:")
    print(f"  Target: np.arange({target_len})")
    print(f"  Slice: {slc}")
    print(f"  Computed: {computed}, Actual: {actual}, Match: {computed == actual}")

# Additional edge case testing
print("\n3. Additional edge case testing:")
edge_cases = [
    (10, slice(5, 3), "Normal array, backwards slice (start > stop)"),
    (10, slice(-1, -5), "Negative indices, backwards"),
    (10, slice(100, 200), "Out of bounds slice"),
    (0, slice(None, None, -1), "Empty array, negative step"),
    (5, slice(None, None, -1), "Array[5], negative step"),
]

for target_len, slc, desc in edge_cases:
    target = np.arange(target_len)
    computed = length_of_indexer(slc, target)
    actual = len(target[slc])
    print(f"{desc}:")
    print(f"  Target: np.arange({target_len})")
    print(f"  Slice: {slc}")
    print(f"  Computed: {computed}, Actual: {actual}, Match: {computed == actual}")

# Verify the mathematical invariant: length should always be >= 0
print("\n4. Checking mathematical invariant (length >= 0):")
violations = []
test_configs = [
    (0, slice(1, None)),
    (0, slice(5, 10)),
    (3, slice(5, None)),
    (5, slice(3, 2)),
    (10, slice(5, 3)),
    (10, slice(100, 200)),
]

for target_len, slc in test_configs:
    target = np.arange(target_len)
    computed = length_of_indexer(slc, target)
    if computed < 0:
        violations.append((target_len, slc, computed))

if violations:
    print(f"Found {len(violations)} violations where length < 0:")
    for target_len, slc, computed in violations:
        print(f"  np.arange({target_len})[{slc}] -> length = {computed}")
else:
    print("No violations found - all lengths are non-negative")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
if violations:
    print(f"BUG CONFIRMED: {len(violations)} cases return negative lengths")
    print("This violates the mathematical invariant that lengths must be >= 0")
else:
    print("NO BUG: All computed lengths are non-negative")