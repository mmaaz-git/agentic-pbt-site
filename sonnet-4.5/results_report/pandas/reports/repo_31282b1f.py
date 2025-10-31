import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

# Create indexer with negative window_size
indexer = FixedForwardWindowIndexer(window_size=-1)

# Get window bounds for 2 values
start, end = indexer.get_window_bounds(num_values=2)

print(f"start: {start}")
print(f"end: {end}")
print()

# Check if bounds are valid
for i in range(len(start)):
    if start[i] > end[i]:
        print(f"Invalid bounds at index {i}: start[{i}]={start[i]} > end[{i}]={end[i]}")
    else:
        print(f"Valid bounds at index {i}: start[{i}]={start[i]} <= end[{i}]={end[i]}")