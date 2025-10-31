#!/usr/bin/env python3
"""
Test Python's built-in slice.indices() method to understand correct behavior
"""

# Test various slices with negative steps
test_cases = [
    (slice(None, None, -1), 1),
    (slice(None, None, -1), 5),
    (slice(None, None, -2), 4),
    (slice(5, None, -1), 7),
    (slice(None, 2, -1), 6),
]

print("Testing Python's slice.indices() method:")
print("=" * 60)

for slc, target_len in test_cases:
    target = list(range(target_len))

    # Get the indices from slice.indices()
    start, stop, step = slc.indices(target_len)

    # Calculate the length using range
    length_via_range = len(range(start, stop, step))

    # Get actual length via slicing
    actual_len = len(target[slc])

    print(f"\nSlice: {slc}, Target length: {target_len}")
    print(f"  slice.indices() returns: start={start}, stop={stop}, step={step}")
    print(f"  len(range({start}, {stop}, {step})) = {length_via_range}")
    print(f"  Actual len(target[slice]) = {actual_len}")
    print(f"  Target[slice] = {target[slc]}")
    print(f"  Lengths match: {length_via_range == actual_len}")

print("\n" + "=" * 60)
print("Testing the proposed fix:")
print("Using: len(range(*slice.indices(target_len)))")
print()

from pandas.core.indexers import length_of_indexer

for slc, target_len in test_cases:
    target = list(range(target_len))

    # Current implementation
    current_result = length_of_indexer(slc, target)

    # Proposed fix
    proposed_result = len(range(*slc.indices(target_len)))

    # Actual
    actual_result = len(target[slc])

    print(f"Slice: {slc}, Target length: {target_len}")
    print(f"  Current implementation: {current_result}")
    print(f"  Proposed fix: {proposed_result}")
    print(f"  Actual: {actual_result}")
    print(f"  Fix is correct: {proposed_result == actual_result}")
    print()