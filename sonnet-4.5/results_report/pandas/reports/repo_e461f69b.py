import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

# Create an indexer with negative window size
indexer = FixedForwardWindowIndexer(window_size=-1)

# Get window bounds for 2 values
start, end = indexer.get_window_bounds(num_values=2)

print(f"start: {start}")
print(f"end: {end}")

# Check each index
for i in range(len(start)):
    print(f"Index {i}: start[{i}] = {start[i]}, end[{i}] = {end[i]}")
    if start[i] > end[i]:
        print(f"ERROR: start[{i}] > end[{i}] (violates window bound invariant)")

# Verify the invariant
assert np.all(start <= end), f"Invariant violated: Found start > end at some indices"