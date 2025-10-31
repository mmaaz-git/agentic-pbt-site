import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

# Create an indexer with negative window size
indexer = FixedForwardWindowIndexer(window_size=-1)

# Get window bounds for 3 values
start, end = indexer.get_window_bounds(num_values=3)

print("Window bounds for num_values=3, window_size=-1:")
print(f"start array: {start}")
print(f"end array: {end}")
print()

# Check the invariant violation
for i in range(len(start)):
    print(f"Window {i}: start[{i}]={start[i]}, end[{i}]={end[i]}")
    if start[i] > end[i]:
        print(f"  *** INVARIANT VIOLATED: start[{i}] > end[{i}] ***")
    else:
        print(f"  OK: start[{i}] <= end[{i}]")

# Demonstrate the effect on rolling operations
import numpy as np
df = pd.DataFrame({'A': [1, 2, 3]})
print("\nDataFrame:")
print(df)

# Custom rolling with the broken indexer
print("\nRolling with FixedForwardWindowIndexer(window_size=-1):")
rolling = df.rolling(indexer)
result = rolling.sum()
print(result)