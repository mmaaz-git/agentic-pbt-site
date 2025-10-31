#!/usr/bin/env python3
"""Test to reproduce the dask.dataframe.from_pandas surrogate character bug"""

import pandas as pd
import dask.dataframe as dd
import traceback

print("=" * 60)
print("Testing dask.dataframe.from_pandas with surrogate characters")
print("=" * 60)

# Test 1: Simple reproduction with surrogate character
print("\nTest 1: Simple surrogate character test")
print("-" * 40)
try:
    df = pd.DataFrame({'text': ['\ud800']})
    print(f"Created pandas DataFrame: {df}")
    print(f"DataFrame dtypes: {df.dtypes}")

    ddf = dd.from_pandas(df, npartitions=1)
    print("Successfully converted to Dask DataFrame!")
    result = ddf.compute()
    print(f"Computed result: {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 2: Test with the exact failing input from the report
print("\n\nTest 2: Exact failing input from bug report")
print("-" * 40)
try:
    df = pd.DataFrame({'a': [0], 'b': [0.0], 'c': ['\ud800']})
    print(f"Created pandas DataFrame:\n{df}")
    print(f"DataFrame dtypes:\n{df.dtypes}")

    ddf = dd.from_pandas(df, npartitions=3)
    print("Successfully converted to Dask DataFrame!")
    result = ddf.compute()
    print(f"Computed result:\n{result}")

    # Try to verify round-trip
    pd.testing.assert_frame_equal(result, df, check_index_type=False)
    print("Round-trip test passed!")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 3: Verify that pandas itself handles surrogate characters
print("\n\nTest 3: Verify pandas handles surrogate characters")
print("-" * 40)
try:
    df = pd.DataFrame({'text': ['\ud800', 'normal text', '\udc00']})
    print(f"Created pandas DataFrame with multiple surrogates:\n{df}")

    # Test some pandas operations
    print(f"\nString length of first element: {len(df['text'].iloc[0])}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Can access values: {df['text'].values}")
    print("Pandas handles surrogate characters without issues!")
except Exception as e:
    print(f"ERROR in pandas: {type(e).__name__}: {e}")

# Test 4: Check if disabling PyArrow conversion helps
print("\n\nTest 4: Test with PyArrow string conversion disabled")
print("-" * 40)
try:
    import dask

    # Try to disable PyArrow string conversion
    with dask.config.set({'dataframe.convert-string': False}):
        df = pd.DataFrame({'text': ['\ud800']})
        print(f"Created pandas DataFrame: {df}")

        ddf = dd.from_pandas(df, npartitions=1)
        print("Successfully converted to Dask DataFrame with PyArrow disabled!")
        result = ddf.compute()
        print(f"Computed result: {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test completed")
print("=" * 60)