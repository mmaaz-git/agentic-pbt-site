#!/usr/bin/env python3
"""Test the reported dask.dataframe multiplication bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import dask.dataframe as dd
import pandas as pd
import numpy as np

print("Testing specific failing example:")
print("="*50 + "\n")

# Test the specific example from the bug report
df1 = pd.DataFrame({'x': [2, 2], 'y': [0, 0]})
df2 = pd.DataFrame({'x': [4611686018427387904, 4611686018427387904, 4611686018427387904], 'y': [0, 0, 0]})

print("df1:")
print(df1)
print("\ndf2:")
print(df2)

# Calculate pandas result
pandas_result = df1 * df2
print("\nPandas result:")
print(pandas_result)
print(f"Pandas result dtypes: {pandas_result.dtypes.to_dict()}")
print(f"Pandas x[0] value: {pandas_result['x'].iloc[0]}")
print(f"Pandas x[1] value: {pandas_result['x'].iloc[1]}")
if len(pandas_result) > 2:
    print(f"Pandas x[2] value: {pandas_result['x'].iloc[2]}")

# Calculate dask result with explicit same npartitions for both
print("\n" + "="*50)
print("Testing with Dask (2 partitions):")
print("="*50)
try:
    ddf1 = dd.from_pandas(df1, npartitions=2)
    ddf2 = dd.from_pandas(df2, npartitions=2)
    dask_result = (ddf1 * ddf2).compute()
    print("\nDask result:")
    print(dask_result)
    print(f"Dask result dtypes: {dask_result.dtypes.to_dict()}")
    print(f"Dask x[0] value: {dask_result['x'].iloc[0]}")
    print(f"Dask x[1] value: {dask_result['x'].iloc[1]}")
    if len(dask_result) > 2:
        print(f"Dask x[2] value: {dask_result['x'].iloc[2]}")

    print("\n" + "="*50)
    print("Comparison:")
    print("="*50)
    print(f"Expected x[0]: {pandas_result['x'].iloc[0]}")
    print(f"Actual x[0]:   {dask_result['x'].iloc[0]}")
    print(f"Are they equal? {pandas_result['x'].iloc[0] == dask_result['x'].iloc[0]}")

    print(f"\nExpected x[1]: {pandas_result['x'].iloc[1]}")
    print(f"Actual x[1]:   {dask_result['x'].iloc[1]}")
    print(f"Are they equal? {pandas_result['x'].iloc[1] == dask_result['x'].iloc[1]}")
except Exception as e:
    print(f"Error with 2 partitions: {e}")

# Let's also check what happens with the multiplication itself
print("\n" + "="*50)
print("Manual calculation check:")
print("="*50)
val1 = 2
val2 = 4611686018427387904
print(f"Value 1: {val1}")
print(f"Value 2: {val2}")
print(f"Python int multiplication: {val1 * val2}")

# Check numpy behavior
result_np_int64 = np.int64(val1) * np.int64(val2)
print(f"NumPy int64 multiplication (with overflow): {result_np_int64}")

# Check if there's an overflow
print(f"\nMax int64: {np.iinfo(np.int64).max}")
print(f"Product as Python int: {val1 * val2}")
print(f"Product > max int64? {val1 * val2 > np.iinfo(np.int64).max}")

# Show what pandas does
print("\n" + "="*50)
print("Understanding pandas behavior:")
print("="*50)
s1 = pd.Series([2, 2], dtype='int64')
s2 = pd.Series([4611686018427387904, 4611686018427387904, 4611686018427387904], dtype='int64')
result_pandas = s1 * s2
print(f"Pandas Series multiplication result:")
print(result_pandas)
print(f"Result dtype: {result_pandas.dtype}")

# Test with single partition to see if it's a partitioning issue
print("\n" + "="*50)
print("Testing with Dask (1 partition):")
print("="*50)
try:
    ddf1_single = dd.from_pandas(df1, npartitions=1)
    ddf2_single = dd.from_pandas(df2, npartitions=1)
    dask_result_single = (ddf1_single * ddf2_single).compute()
    print("Dask result (1 partition):")
    print(dask_result_single)
    print(f"x[0] value: {dask_result_single['x'].iloc[0]}")
    print(f"Matches pandas? {dask_result_single['x'].iloc[0] == pandas_result['x'].iloc[0]}")
except Exception as e:
    print(f"Error with 1 partition: {e}")

# Let's try with aligned indices first
print("\n" + "="*50)
print("Testing with pre-aligned DataFrames (same index):")
print("="*50)
df1_aligned = pd.DataFrame({'x': [2, 2, np.nan], 'y': [0, 0, np.nan]}, index=[0, 1, 2])
df2_aligned = pd.DataFrame({'x': [4611686018427387904, 4611686018427387904, 4611686018427387904], 'y': [0, 0, 0]}, index=[0, 1, 2])
pandas_aligned = df1_aligned * df2_aligned
print("Pandas result (pre-aligned):")
print(pandas_aligned)

try:
    ddf1_aligned = dd.from_pandas(df1_aligned, npartitions=2)
    ddf2_aligned = dd.from_pandas(df2_aligned, npartitions=2)
    dask_aligned = (ddf1_aligned * ddf2_aligned).compute()
    print("\nDask result (pre-aligned):")
    print(dask_aligned)
    print(f"Do aligned results match? {pandas_aligned.equals(dask_aligned)}")
except Exception as e:
    print(f"Error with aligned DataFrames: {e}")