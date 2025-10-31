#!/usr/bin/env python3
"""Test to reproduce the dask.dataframe.from_pandas surrogate character bug"""

import pandas as pd
import dask.dataframe as dd
import traceback
import sys

print("=" * 60)
print("Testing dask.dataframe.from_pandas with surrogate characters")
print("=" * 60)

# Test 1: Simple reproduction with surrogate character
print("\nTest 1: Simple surrogate character test")
print("-" * 40)
try:
    df = pd.DataFrame({'text': ['\ud800']})
    print(f"Created pandas DataFrame with shape: {df.shape}")
    print(f"DataFrame dtypes: {df.dtypes}")
    print(f"First value repr: {repr(df['text'].iloc[0])}")

    print("\nAttempting to convert to Dask DataFrame...")
    ddf = dd.from_pandas(df, npartitions=1)
    print("Successfully converted to Dask DataFrame!")

    print("Attempting to compute...")
    result = ddf.compute()
    print(f"Computed result shape: {result.shape}")
    print("Success! Round-trip completed.")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

# Test 2: Test with the exact failing input from the report
print("\n\nTest 2: Exact failing input from bug report")
print("-" * 40)
try:
    df = pd.DataFrame({'a': [0], 'b': [0.0], 'c': ['\ud800']})
    print(f"Created pandas DataFrame with shape: {df.shape}")
    print(f"DataFrame dtypes:\n{df.dtypes}")
    print(f"Column 'c' first value repr: {repr(df['c'].iloc[0])}")

    print("\nAttempting to convert to Dask DataFrame...")
    ddf = dd.from_pandas(df, npartitions=3)
    print("Successfully converted to Dask DataFrame!")

    print("Attempting to compute...")
    result = ddf.compute()
    print(f"Computed result shape: {result.shape}")

    # Try to verify round-trip
    pd.testing.assert_frame_equal(result, df, check_index_type=False)
    print("Round-trip test passed!")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

# Test 3: Verify that pandas itself handles surrogate characters
print("\n\nTest 3: Verify pandas handles surrogate characters")
print("-" * 40)
try:
    df = pd.DataFrame({'text': ['\ud800', 'normal text', '\udc00']})
    print(f"Created pandas DataFrame with shape: {df.shape}")
    print(f"DataFrame dtypes: {df.dtypes}")

    # Test some pandas operations
    print(f"\nString length of first element: {len(df['text'].iloc[0])}")
    print(f"String repr of first element: {repr(df['text'].iloc[0])}")
    print(f"DataFrame shape: {df.shape}")

    # Test some operations that don't require printing the value
    print(f"Can compute string lengths: {df['text'].str.len().tolist()}")
    print(f"Can check if values exist: {df['text'].notna().all()}")
    print("Pandas handles surrogate characters without issues!")
except Exception as e:
    print(f"ERROR in pandas: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

# Test 4: Check if disabling PyArrow conversion helps
print("\n\nTest 4: Test with PyArrow string conversion disabled")
print("-" * 40)
try:
    import dask

    # Try to disable PyArrow string conversion
    print("Setting dask config to disable PyArrow string conversion...")
    with dask.config.set({'dataframe.convert-string': False}):
        df = pd.DataFrame({'text': ['\ud800']})
        print(f"Created pandas DataFrame with shape: {df.shape}")

        print("\nAttempting to convert to Dask DataFrame...")
        ddf = dd.from_pandas(df, npartitions=1)
        print("Successfully converted to Dask DataFrame with PyArrow disabled!")

        print("Attempting to compute...")
        result = ddf.compute()
        print(f"Computed result shape: {result.shape}")
        print("Success with PyArrow disabled!")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

# Test 5: Run the Hypothesis test
print("\n\nTest 5: Running Hypothesis property test")
print("-" * 40)
try:
    from hypothesis import given, settings, example
    from hypothesis.extra.pandas import column, data_frames, range_indexes

    @given(
        data_frames([
            column('a', dtype=int),
            column('b', dtype=float),
            column('c', dtype=str)
        ], index=range_indexes(min_size=1, max_size=50))
    )
    @example(pd.DataFrame({'a': [0], 'b': [0.0], 'c': ['\ud800']}))
    @settings(max_examples=10, deadline=2000)
    def test_from_pandas_roundtrip_dataframe(df):
        ddf = dd.from_pandas(df, npartitions=3)
        result = ddf.compute()
        pd.testing.assert_frame_equal(result, df, check_index_type=False)

    print("Running property test with 10 examples...")
    test_from_pandas_roundtrip_dataframe()
    print("Property test passed!")
except Exception as e:
    print(f"ERROR in property test: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test completed")
print("=" * 60)