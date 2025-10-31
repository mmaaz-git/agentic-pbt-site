from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2, step=1)

print(f"start: {start}")
print(f"end: {end}")
print(f"\nInvariant violated: start[1]={start[1]} > end[1]={end[1]}")
print(f"Is start[1] > end[1]? {start[1] > end[1]}")