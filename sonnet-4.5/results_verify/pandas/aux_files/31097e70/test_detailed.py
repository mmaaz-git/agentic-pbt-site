import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

# Test with window_size=-1 and num_values=5 to see the pattern
indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=5)

print(f"window_size=-1, num_values=5:")
print(f"start: {start}")
print(f"end: {end}")
print()

# Check violations
for i in range(len(start)):
    if start[i] > end[i]:
        print(f"Violation at index {i}: start[{i}]={start[i]} > end[{i}]={end[i]}")

# Let's trace through the code logic:
# Line 340: start = np.arange(0, num_values, step, dtype="int64")
#           start = [0, 1, 2, 3, 4]
# Line 341: end = start + self.window_size
#           end = [0, 1, 2, 3, 4] + (-1) = [-1, 0, 1, 2, 3]
# Line 342-343: if self.window_size:  # -1 is truthy
#               end = np.clip(end, 0, num_values)
#               end = [0, 0, 1, 2, 3]
print("\nTraced logic:")
print("start = np.arange(0, 5, 1) = [0, 1, 2, 3, 4]")
print("end = start + (-1) = [-1, 0, 1, 2, 3]")
print("end = np.clip(end, 0, 5) = [0, 0, 1, 2, 3]")
print("Result: start[i] > end[i] for i >= 1")