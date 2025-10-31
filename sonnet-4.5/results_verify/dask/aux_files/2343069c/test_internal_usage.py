#!/usr/bin/env python3
"""Test internal usage of _read_orc to see if the mutation is problematic"""

import tempfile
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.dataframe.io.orc.core import _read_orc

# Simulate what dd.from_map does - it passes the same columns to each partition
def test_from_map_scenario():
    # This simulates what happens in dd.from_map when columns is passed
    columns = ['col1', 'col2']
    original = columns.copy()

    # Simulate multiple partitions being read with the same columns reference
    parts = [("dummy_path", [0]), ("dummy_path", [1]), ("dummy_path", [2])]

    print(f"Original columns: {original}")
    print(f"Testing multiple partition reads with same columns list:")

    for i, part in enumerate(parts):
        print(f"\n  Partition {i}:")
        print(f"    Before _read_orc: {columns}")

        # Simulate what would happen in _read_orc
        # (we can't actually call it without valid ORC files)
        if 'col1' not in columns:  # index='col1'
            columns.append('col1')

        print(f"    After _read_orc:  {columns}")

    print(f"\nFinal state: {columns}")
    print(f"List grew from {len(original)} to {len(columns)} items!")
    print(f"Final list has duplicates? {len(columns) != len(set(columns))}")

test_from_map_scenario()