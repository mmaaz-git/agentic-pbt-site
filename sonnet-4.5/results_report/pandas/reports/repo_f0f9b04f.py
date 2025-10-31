from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2, step=1)

print(f"start: {start}")
print(f"end: {end}")
print(f"\nInvariant violated: start[1]={start[1]} > end[1]={end[1]}")

# Check all windows
for i in range(len(start)):
    if start[i] > end[i]:
        print(f"Window {i}: start[{i}]={start[i]} > end[{i}]={end[i]} - INVALID")
    else:
        print(f"Window {i}: start[{i}]={start[i]} <= end[{i}]={end[i]} - OK")