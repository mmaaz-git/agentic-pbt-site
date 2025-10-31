import pandas as pd
import numpy as np
from pandas.api.indexers import FixedWindowIndexer, FixedForwardWindowIndexer

# Test what happens with FixedWindowIndexer with negative window
indexer = FixedWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2)

print(f"FixedWindowIndexer with window_size=-1:")
print(f"start: {start}")
print(f"end: {end}")
print(f"start[1] = {start[1]}, end[1] = {end[1]}")

# Also check that normal FixedWindowIndexer can produce invalid bounds
if start[1] > end[1]:
    print(f"FixedWindowIndexer also violates invariant: start[1] ({start[1]}) > end[1] ({end[1]})")
else:
    print(f"FixedWindowIndexer maintains invariant: start[1] ({start[1]}) <= end[1] ({end[1]})")