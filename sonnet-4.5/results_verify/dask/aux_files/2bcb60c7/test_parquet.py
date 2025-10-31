#!/usr/bin/env python3
"""Test if read_parquet has similar behavior"""

import tempfile
import pandas as pd
import dask.dataframe as dd

def test_read_parquet_with_index_not_in_columns():
    """Test reading Parquet with index column not included in columns list"""
    print("Testing read_parquet with index not in columns list...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })

        # Save as Parquet using dask
        df = dd.from_pandas(data, npartitions=1)
        df.to_parquet(tmpdir, write_index=False)

        print(f"Created Parquet file in {tmpdir}")
        print(f"Original data columns: {data.columns.tolist()}")

        # Test: Reading with index NOT in columns list
        try:
            result = dd.read_parquet(tmpdir, columns=['b', 'c'], index='a')
            result_df = result.compute()
            print(f"\nParquet test - Index 'a' NOT in columns list: SUCCESS")
            print(f"Columns: {result_df.columns.tolist()}")
            print(f"Index name: {result_df.index.name}")
            print(f"Index values: {result_df.index.tolist()}")
        except Exception as e:
            print(f"\nParquet test - Index 'a' NOT in columns list: FAILED")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")

if __name__ == "__main__":
    test_read_parquet_with_index_not_in_columns()