import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-2)
start, end = indexer.get_window_bounds(num_values=5)

print(f"start: {start}")
print(f"end: {end}")

for i in range(len(start)):
    if start[i] > end[i]:
        print(f"Invalid: start[{i}]={start[i]} > end[{i}]={end[i]}")