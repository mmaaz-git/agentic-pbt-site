from pandas.api.indexers import FixedForwardWindowIndexer

# Reproduce the bug with negative window_size
indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2)

print(f"start: {start}")
print(f"end: {end}")
print(f"\nWindow bounds invariant check:")
for i in range(len(start)):
    if start[i] > end[i]:
        print(f"  VIOLATION at index {i}: start[{i}]={start[i]} > end[{i}]={end[i]}")
    else:
        print(f"  OK at index {i}: start[{i}]={start[i]} <= end[{i}]={end[i]}")