import numpy as np
from pandas.core.indexers.objects import FixedWindowIndexer

# Create the indexer with window_size=0
indexer = FixedWindowIndexer(window_size=0)

# Get window bounds with the failing parameters
start, end = indexer.get_window_bounds(num_values=2, closed='neither')

print(f"start: {start}")
print(f"end: {end}")
print(f"start[1] > end[1]: {start[1]} > {end[1]}")
print(f"Violates invariant: {start[1] > end[1]}")