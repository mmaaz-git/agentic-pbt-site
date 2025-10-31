#!/usr/bin/env python3
"""Test if the bug affects public API users"""

import tempfile
import pandas as pd
import numpy as np
import dask.dataframe as dd

# Create a test ORC file
data = pd.DataFrame({
    'col1': np.arange(10),
    'col2': np.arange(10, 20),
    'col3': np.arange(20, 30)
})

with tempfile.TemporaryDirectory() as tmpdir:
    # Write test data
    df = dd.from_pandas(data, npartitions=2)
    df.to_orc(tmpdir)

    # Test with columns list being reused
    columns_list = ['col1', 'col2']
    original_columns = columns_list.copy()

    # Read ORC with index
    result = dd.read_orc(tmpdir, columns=columns_list, index='col1')

    print(f"Original columns: {original_columns}")
    print(f"Columns after read_orc: {columns_list}")
    print(f"Columns mutated? {columns_list != original_columns}")
    print(f"Result columns: {list(result.columns)}")

    # Test another read with the same list
    if columns_list != original_columns:
        print("\nAttempting second read with mutated list:")
        result2 = dd.read_orc(tmpdir, columns=columns_list, index='col1')
        print(f"Result2 columns: {list(result2.columns)}")