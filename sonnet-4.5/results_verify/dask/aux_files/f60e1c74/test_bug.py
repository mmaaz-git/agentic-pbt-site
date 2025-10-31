#!/usr/bin/env python3
"""Test the reported bug in dask.dataframe.DataFrame.repartition"""

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import data_frames, column
import dask.dataframe as dd
import traceback

print("Testing the property-based test from the bug report...")

# First run the exact failing case
print("\n1. Running the exact failing case from the bug report:")
df = pd.DataFrame({'a': [0], 'b': [0.0]})
npartitions1 = 1
npartitions2 = 2

print(f"   DataFrame: {df.to_dict()}")
print(f"   npartitions1: {npartitions1}, npartitions2: {npartitions2}")

try:
    ddf = dd.from_pandas(df, npartitions=npartitions1, sort=False)
    print(f"   Created dask dataframe with {ddf.npartitions} partitions")
    print(f"   Divisions: {ddf.divisions}")

    repartitioned = ddf.repartition(npartitions=npartitions2)
    result = repartitioned.compute()
    print(f"   SUCCESS: Repartitioning worked!")
    print(f"   Result shape: {result.shape}")
except AssertionError as e:
    print(f"   ASSERTION ERROR: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"   OTHER ERROR ({type(e).__name__}): {e}")
    traceback.print_exc()

# Now test variations
print("\n2. Testing with sort=True instead of sort=False:")
try:
    ddf = dd.from_pandas(df, npartitions=npartitions1, sort=True)
    print(f"   Created dask dataframe with {ddf.npartitions} partitions")
    print(f"   Divisions: {ddf.divisions}")

    repartitioned = ddf.repartition(npartitions=npartitions2)
    result = repartitioned.compute()
    print(f"   SUCCESS: Repartitioning worked with sort=True!")
    print(f"   Result shape: {result.shape}")
except AssertionError as e:
    print(f"   ASSERTION ERROR: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"   OTHER ERROR ({type(e).__name__}): {e}")
    traceback.print_exc()

# Test with more data
print("\n3. Testing with more data (10 rows):")
df_bigger = pd.DataFrame({'a': list(range(10)), 'b': [float(i) for i in range(10)]})
try:
    ddf = dd.from_pandas(df_bigger, npartitions=1, sort=False)
    print(f"   Created dask dataframe with {ddf.npartitions} partitions")
    print(f"   Divisions: {ddf.divisions}")

    repartitioned = ddf.repartition(npartitions=2)
    result = repartitioned.compute()
    print(f"   SUCCESS: Repartitioning worked with more data!")
    print(f"   Result shape: {result.shape}")
except AssertionError as e:
    print(f"   ASSERTION ERROR: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"   OTHER ERROR ({type(e).__name__}): {e}")
    traceback.print_exc()

# Test repartitioning from 2 to 1 (opposite direction)
print("\n4. Testing repartitioning from 2 to 1 (reducing partitions):")
df_test = pd.DataFrame({'a': [0, 1], 'b': [0.0, 1.0]})
try:
    ddf = dd.from_pandas(df_test, npartitions=2, sort=False)
    print(f"   Created dask dataframe with {ddf.npartitions} partitions")
    print(f"   Divisions: {ddf.divisions}")

    repartitioned = ddf.repartition(npartitions=1)
    result = repartitioned.compute()
    print(f"   SUCCESS: Repartitioning from 2 to 1 worked!")
    print(f"   Result shape: {result.shape}")
except AssertionError as e:
    print(f"   ASSERTION ERROR: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"   OTHER ERROR ({type(e).__name__}): {e}")
    traceback.print_exc()

# Now run the property test
print("\n5. Running the property-based test (limited examples):")
@settings(max_examples=10)
@given(
    df=data_frames(
        columns=[
            column("a", elements=st.integers(-100, 100)),
            column("b", elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)),
        ],
    ),
    npartitions1=st.integers(min_value=1, max_value=5),
    npartitions2=st.integers(min_value=1, max_value=5),
)
def test_repartition_preserves_data(df, npartitions1, npartitions2):
    if len(df) == 0:
        npartitions1 = 1
        npartitions2 = 1

    try:
        ddf = dd.from_pandas(df, npartitions=npartitions1, sort=False)
        repartitioned = ddf.repartition(npartitions=npartitions2)
        result = repartitioned.compute()
        pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True), check_dtype=False)
        return True
    except AssertionError:
        print(f"   FAILED with df shape {df.shape}, npartitions1={npartitions1}, npartitions2={npartitions2}")
        return False

# Run a few examples
try:
    test_repartition_preserves_data()
    print("   Property test passed!")
except Exception as e:
    print(f"   Property test failed: {e}")