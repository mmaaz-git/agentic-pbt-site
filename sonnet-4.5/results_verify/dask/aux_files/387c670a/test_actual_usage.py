#!/usr/bin/env python3
"""Test if the bug causes actual failures in Dask usage"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from datetime import datetime

# Create a test dataframe with the problematic date range
start = pd.Timestamp('2000-01-01')
end = start + pd.Timedelta(days=30)

# Create a pandas DataFrame first
dates = pd.date_range(start, end, periods=100)
df = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(100)
})
df = df.set_index('date')

print("Testing actual Dask resample usage with the problematic parameters")
print("=" * 60)

# Convert to Dask DataFrame with 12 partitions (matching our bug case)
ddf = dd.from_pandas(df, npartitions=12)

print(f"Dask DataFrame created with {ddf.npartitions} partitions")
print(f"Divisions: {ddf.divisions}")

try:
    # Try to resample with the problematic parameters
    resampled = ddf.resample('3D', closed='right', label='right').sum()
    print("\nResampling created successfully")

    # Try to compute the result
    result = resampled.compute()
    print(f"\nComputation successful!")
    print(f"Result shape: {result.shape}")
    print(f"First few rows:\n{result.head()}")

except Exception as e:
    print(f"\nERROR during resample/compute: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Also test with mean
print("\n" + "=" * 60)
print("Testing with mean aggregation:")
try:
    resampled = ddf.resample('3D', closed='right', label='right').mean()
    result = resampled.compute()
    print(f"Mean computation successful!")
    print(f"Result shape: {result.shape}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")