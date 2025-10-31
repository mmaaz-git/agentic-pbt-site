#!/usr/bin/env python3
"""Test how dask handles index with columns parameter in read_parquet"""

import pandas as pd
import dask.dataframe as dd
import tempfile
import shutil

# Create test data
df_pandas = pd.DataFrame({
    "a": [1, 2, 3, 4, 5],
    "b": [10, 20, 30, 40, 50],
    "c": [100, 200, 300, 400, 500],
})

# Test with Parquet
tmp = tempfile.mkdtemp()
try:
    print("=== Testing with Parquet ===")
    df_dask = dd.from_pandas(df_pandas, npartitions=2)
    df_dask.to_parquet(tmp)

    print("Original data written to Parquet")

    # Try reading with columns=['a', 'b'] and index='c'
    try:
        print("\nAttempting to read Parquet with columns=['a', 'b'], index='c'...")
        df_read = dd.read_parquet(tmp, columns=["a", "b"], index="c")
        result = df_read.compute()
        print("Success! Result:")
        print(result)
        print(f"Columns: {list(result.columns)}")
        print(f"Index name: {result.index.name}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

finally:
    shutil.rmtree(tmp, ignore_errors=True)