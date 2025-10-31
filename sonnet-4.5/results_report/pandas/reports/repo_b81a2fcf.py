import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

# Create indexer with negative window size
indexer = FixedForwardWindowIndexer(window_size=-1)

# Get window bounds for 2 values
start, end = indexer.get_window_bounds(num_values=2)

print(f"start: {start}")
print(f"end: {end}")
print(f"start[1] = {start[1]}, end[1] = {end[1]}")

# Check the invariant
assert start[1] <= end[1], f"Invariant violated: start[1] ({start[1]}) > end[1] ({end[1]})"