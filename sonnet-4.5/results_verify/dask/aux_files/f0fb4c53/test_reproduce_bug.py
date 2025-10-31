#!/usr/bin/env python3
"""Test to reproduce the bug reported in dask.dataframe.nlargest"""

import pandas as pd
import dask.dataframe as dd
import traceback

print("Testing dask.dataframe.nlargest bug reproduction...")
print("=" * 60)

# Create test data as described in the bug report
df = pd.DataFrame({'x': [0, 0, 0, 0, 1]})
print("Created pandas DataFrame:")
print(df)
print()

# Convert to dask dataframe
ddf = dd.from_pandas(df, npartitions=2)
print("Converted to dask DataFrame with 2 partitions")
print()

# Try the operation that should fail
print("Attempting: ddf.nlargest(1, 'x')['x'].compute()")
try:
    result = ddf.nlargest(1, 'x')['x']
    computed = result.compute()
    print(f"Result: {computed}")
    print("ERROR: Expected a TypeError but the operation succeeded!")
except TypeError as e:
    print(f"Got expected TypeError: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
except Exception as e:
    print(f"Got unexpected exception: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Testing the pandas equivalent to verify expected behavior:")
print()

# Test with pandas to show what the expected behavior should be
print("pandas: df.nlargest(1, 'x')['x']")
pandas_result = df.nlargest(1, 'x')['x']
print(f"Result: {pandas_result.tolist()}")