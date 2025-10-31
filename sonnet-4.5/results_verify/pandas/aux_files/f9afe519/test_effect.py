import numpy as np
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

# Test actual effect on rolling operations
print("=== Testing effect on rolling operations ===")

# Create a simple DataFrame
df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
print("DataFrame:")
print(df)
print()

# Test with positive window (normal case)
print("Normal case with window_size=2:")
indexer_pos = FixedForwardWindowIndexer(window_size=2)
result_pos = df.rolling(window=indexer_pos, min_periods=1).sum()
print(result_pos)
print()

# Test with negative window
print("Negative window case with window_size=-1:")
try:
    indexer_neg = FixedForwardWindowIndexer(window_size=-1)
    result_neg = df.rolling(window=indexer_neg, min_periods=1).sum()
    print(result_neg)
    print("\nThis worked but the values may be unexpected")
except Exception as e:
    print(f"Error occurred: {e}")

# Let's see what window bounds are actually used
print("\nActual window bounds for negative window size:")
indexer_neg = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer_neg.get_window_bounds(5)
print(f"start: {start}")
print(f"end: {end}")

# Manually verify what each window would contain
print("\nSlices that would be created:")
values = np.array([1, 2, 3, 4, 5])
for i in range(len(start)):
    slice_values = values[start[i]:end[i]]
    print(f"  Window {i}: values[{start[i]}:{end[i]}] = {slice_values} (sum={np.sum(slice_values)})")