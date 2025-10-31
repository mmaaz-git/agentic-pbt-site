#!/usr/bin/env python3
"""Test what happens when using resample with the bug"""

import pandas as pd
import dask.dataframe as dd
import numpy as np

# Create test data that triggers the bug
dates = pd.date_range('2000-12-17 00:00:00', periods=2, freq='30min')
df = pd.DataFrame({'value': [1, 2]}, index=dates)

print("Original DataFrame:")
print(df)

# Convert to Dask DataFrame
ddf = dd.from_pandas(df, npartitions=1)
print(f"\nOriginal Dask divisions: {ddf.divisions}")

# Try to resample - this should trigger the bug
try:
    print("\nAttempting to resample with '1W', closed='right', label='right'...")
    resampled = ddf.resample('1W', closed='right', label='right')

    # Try to perform an aggregation
    result = resampled.sum()

    print(f"Resample result divisions: {result.divisions}")
    print(f"Are result divisions monotonic? {all(result.divisions[i] <= result.divisions[i+1] for i in range(len(result.divisions)-1))}")

    # Try to compute the result
    computed = result.compute()
    print(f"\nComputed result:\n{computed}")

except Exception as e:
    print(f"Error during resample operation: {e}")
    import traceback
    traceback.print_exc()