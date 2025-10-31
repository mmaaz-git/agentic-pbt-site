#!/usr/bin/env python3
"""Test the reported dask.dataframe multiplication bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import data_frames, column, range_indexes
import dask.dataframe as dd
import pandas as pd
import numpy as np

# First, run the hypothesis test
@settings(max_examples=100)
@given(
    df1=data_frames(
        columns=[
            column('x', dtype=int),
            column('y', dtype=int),
        ],
        index=range_indexes(min_size=1, max_size=30),
    ),
    df2=data_frames(
        columns=[
            column('x', dtype=int),
            column('y', dtype=int),
        ],
        index=range_indexes(min_size=1, max_size=30),
    ),
)
def test_multiply_dataframe_matches_pandas(df1, df2):
    ddf1 = dd.from_pandas(df1, npartitions=2)
    ddf2 = dd.from_pandas(df2, npartitions=2)

    dask_result = (ddf1 * ddf2).compute()
    pandas_result = df1 * df2

    pd.testing.assert_frame_equal(dask_result, pandas_result)

print("Running hypothesis test...")
try:
    test_multiply_dataframe_matches_pandas()
    print("Hypothesis test passed")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

print("\n" + "="*50)
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
print(f"Pandas x[0] type: {type(pandas_result['x'].iloc[0])}")

# Calculate dask result
ddf1 = dd.from_pandas(df1, npartitions=2)
ddf2 = dd.from_pandas(df2, npartitions=2)
dask_result = (ddf1 * ddf2).compute()
print("\nDask result:")
print(dask_result)
print(f"Dask result dtypes: {dask_result.dtypes.to_dict()}")
print(f"Dask x[0] value: {dask_result['x'].iloc[0]}")
print(f"Dask x[0] type: {type(dask_result['x'].iloc[0])}")

print("\n" + "="*50)
print("Comparison:")
print("="*50)
print(f"Expected x[0]: {pandas_result['x'].iloc[0]}")
print(f"Actual x[0]:   {dask_result['x'].iloc[0]}")
print(f"Are they equal? {pandas_result['x'].iloc[0] == dask_result['x'].iloc[0]}")

# Let's also check what happens with the multiplication itself
print("\n" + "="*50)
print("Manual calculation check:")
print("="*50)
val1 = 2
val2 = 4611686018427387904
print(f"Value 1: {val1}")
print(f"Value 2: {val2}")
print(f"Python int multiplication: {val1 * val2}")
print(f"NumPy int64 multiplication: {np.int64(val1) * np.int64(val2)}")

# Check if there's an overflow
import sys
print(f"\nMax int64: {np.iinfo(np.int64).max}")
print(f"Product: {val1 * val2}")
print(f"Product > max int64? {val1 * val2 > np.iinfo(np.int64).max}")

# Test with different partitioning
print("\n" + "="*50)
print("Testing with different partitioning:")
print("="*50)

for n_partitions in [1, 2, 3]:
    ddf1 = dd.from_pandas(df1, npartitions=n_partitions)
    ddf2 = dd.from_pandas(df2, npartitions=n_partitions)
    dask_result = (ddf1 * ddf2).compute()
    print(f"\nWith {n_partitions} partition(s):")
    print(f"  x[0]: {dask_result['x'].iloc[0]}")
    print(f"  Matches pandas? {dask_result['x'].iloc[0] == pandas_result['x'].iloc[0]}")