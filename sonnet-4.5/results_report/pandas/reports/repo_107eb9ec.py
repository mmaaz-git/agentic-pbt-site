import numpy as np
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

# Test case with negative window_size
indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2, step=1)

print(f"start: {start}")
print(f"end: {end}")
print(f"start[1] > end[1]: {start[1]} > {end[1]}")

# Test with rolling operation
df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
print(f"\nOriginal DataFrame:")
print(df)

result = df.rolling(window=indexer, min_periods=1).sum()
print(f"\nRolling result with window_size=-1:")
print(result)