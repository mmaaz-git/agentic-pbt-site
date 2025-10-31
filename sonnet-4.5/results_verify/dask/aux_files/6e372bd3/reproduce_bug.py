#!/usr/bin/env python3
"""Reproduce the bug reported for dask.dataframe.io.orc.read_orc"""

import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd
import traceback

def test_bug_reproduction():
    tmp = tempfile.mkdtemp()
    try:
        print("Creating test DataFrame...")
        df_pandas = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": [100, 200, 300, 400, 500],
        })

        print("Converting to Dask DataFrame and writing to ORC...")
        df_dask = dd.from_pandas(df_pandas, npartitions=2)
        df_dask.to_orc(tmp, write_index=False)

        print("Attempting to read ORC with columns=['a', 'b'] and index='c'...")
        df_read = dd.read_orc(tmp, columns=["a", "b"], index="c")
        result = df_read.compute()

        print("Success! Result:")
        print(result)
        print(f"Columns: {list(result.columns)}")
        print(f"Index name: {result.index.name}")
        return True

    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    print("=== Testing Bug Reproduction ===")
    success = test_bug_reproduction()
    if not success:
        print("\nBug confirmed: read_orc fails when index column is not in columns list")
    else:
        print("\nNo bug found: read_orc works correctly")