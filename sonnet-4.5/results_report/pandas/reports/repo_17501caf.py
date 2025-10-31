import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

# Test with negative window size
indexer = FixedForwardWindowIndexer(window_size=-2)
start, end = indexer.get_window_bounds(num_values=5)

print(f"window_size=-2, num_values=5")
print(f"start: {start}")
print(f"end: {end}")
print()

# Check for invalid bounds
print("Invalid window bounds (where start > end):")
invalid_found = False
for i in range(len(start)):
    if start[i] > end[i]:
        print(f"  Index {i}: start[{i}]={start[i]} > end[{i}]={end[i]}")
        invalid_found = True

if not invalid_found:
    print("  None found")
print()

# Also test the exact failing case from hypothesis
print("Testing hypothesis failing case: window_size=-1, num_values=2")
indexer2 = FixedForwardWindowIndexer(window_size=-1)
start2, end2 = indexer2.get_window_bounds(num_values=2, step=1)
print(f"start: {start2}")
print(f"end: {end2}")

print("\nChecking bounds:")
for i in range(len(start2)):
    valid = "✓" if start2[i] <= end2[i] else "✗ INVALID"
    print(f"  Index {i}: start={start2[i]}, end={end2[i]} {valid}")