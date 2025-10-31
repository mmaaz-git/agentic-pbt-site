#!/usr/bin/env python3
"""
Demonstrating the bug in pandas.core.indexers.length_of_indexer for range objects.
This shows that the function returns incorrect lengths when step > 1.
"""

from pandas.core.indexers import length_of_indexer

# Test various range configurations
test_cases = [
    (0, 1, 2),   # Single element [0], step > span
    (0, 5, 3),   # Two elements [0, 3]
    (0, 10, 7),  # Two elements [0, 7]
    (5, 10, 1),  # Five elements [5, 6, 7, 8, 9]
    (0, 0, 1),   # Empty range
    (10, 20, 3), # Four elements [10, 13, 16, 19]
    (0, 100, 7), # Should have 15 elements
    (1, 0, 1),   # Empty range (start > stop)
    (0, 10, 100),# Single element [0], large step
]

print("Testing pandas.core.indexers.length_of_indexer with range objects:")
print("=" * 70)

for start, stop, step in test_cases:
    indexer = range(start, stop, step)

    # Get the computed result from pandas
    computed = length_of_indexer(indexer)

    # Get the expected result using Python's len()
    expected = len(indexer)

    # Also show the actual elements for clarity
    elements = list(indexer)

    # Check if they match
    match = "✓" if computed == expected else "✗"

    print(f"{match} range({start}, {stop}, {step}):")
    print(f"  Computed: {computed}")
    print(f"  Expected: {expected}")
    print(f"  Elements: {elements}")
    if computed != expected:
        print(f"  ERROR: Off by {abs(computed - expected)}")
    print()