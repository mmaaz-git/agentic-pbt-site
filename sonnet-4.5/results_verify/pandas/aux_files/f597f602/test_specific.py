from pandas.api.indexers import FixedForwardWindowIndexer

# Test the specific failing input from the bug report
indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2, step=1)

print(f"Testing: num_values=2, window_size=-1, step=1")
print(f"start: {start}")
print(f"end: {end}")

for i in range(len(start)):
    print(f"  Index {i}: start={start[i]}, end={end[i]}, valid={start[i] <= end[i]}")