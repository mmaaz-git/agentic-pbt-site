#!/usr/bin/env python3
"""Test script to reproduce the dask ORC read bug"""

import tempfile
import pandas as pd
import dask.dataframe as dd
import traceback

def test_read_orc_with_index_not_in_columns():
    """Test reading ORC with index column not included in columns list"""
    print("Testing read_orc with index not in columns list...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })

        # Save as ORC using dask
        df = dd.from_pandas(data, npartitions=1)
        df.to_orc(tmpdir, write_index=False)

        print(f"Created ORC file in {tmpdir}")
        print(f"Original data columns: {data.columns.tolist()}")

        # Test 1: Reading without index (should work)
        try:
            result1 = dd.read_orc(tmpdir, columns=['b', 'c'])
            result1_df = result1.compute()
            print(f"\nTest 1 - No index specified: SUCCESS")
            print(f"Columns: {result1_df.columns.tolist()}")
            print(f"Index name: {result1_df.index.name}")
        except Exception as e:
            print(f"\nTest 1 - No index specified: FAILED")
            print(f"Error: {e}")

        # Test 2: Reading with index in columns list (should work)
        try:
            result2 = dd.read_orc(tmpdir, columns=['a', 'b', 'c'], index='a')
            result2_df = result2.compute()
            print(f"\nTest 2 - Index 'a' in columns list: SUCCESS")
            print(f"Columns: {result2_df.columns.tolist()}")
            print(f"Index name: {result2_df.index.name}")
        except Exception as e:
            print(f"\nTest 2 - Index 'a' in columns list: FAILED")
            print(f"Error: {e}")
            traceback.print_exc()

        # Test 3: Reading with index NOT in columns list (this is the bug)
        try:
            result3 = dd.read_orc(tmpdir, columns=['b', 'c'], index='a')
            result3_df = result3.compute()
            print(f"\nTest 3 - Index 'a' NOT in columns list: SUCCESS")
            print(f"Columns: {result3_df.columns.tolist()}")
            print(f"Index name: {result3_df.index.name}")
            print(f"Index values: {result3_df.index.tolist()}")
        except Exception as e:
            print(f"\nTest 3 - Index 'a' NOT in columns list: FAILED")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    test_read_orc_with_index_not_in_columns()