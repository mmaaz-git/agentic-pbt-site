#!/usr/bin/env python3
"""Test to show difference between public read_orc and private _read_orc"""

import dask.dataframe as dd
from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine

# Test 1: How dd.from_map might use _read_orc
print("=" * 60)
print("Test 1: Simulating dd.from_map usage")
print("=" * 60)

# This simulates what happens in read_orc at lines 97-108
columns = ['col1', 'col2', 'col3']
index = 'col1'

# Line 97-98: Create a new list if index is in columns
if columns is not None and index in columns:
    columns_for_map = [col for col in columns if col != index]
else:
    columns_for_map = columns

print(f"Original columns: {columns}")
print(f"Columns passed to from_map: {columns_for_map}")
print(f"Are they the same object? {columns is columns_for_map}")

# Now simulate what happens inside dd.from_map when it calls _read_orc
# Note that dd.from_map would pass columns_for_map to _read_orc
print("\nSimulating _read_orc call inside dd.from_map:")
print(f"Before _read_orc: {columns_for_map}")

try:
    _read_orc(
        parts=[],
        engine=ArrowORCEngine,
        fs=None,
        schema={},
        index=index,
        columns=columns_for_map
    )
except Exception as e:
    pass

print(f"After _read_orc: {columns_for_map}")
print(f"Original columns unchanged: {columns}")
print()

# Test 2: Direct call where columns doesn't contain index
print("=" * 60)
print("Test 2: Direct call where index not in columns")
print("=" * 60)

columns2 = ['col2', 'col3']
index2 = 'col1'
print(f"Before _read_orc: {columns2}")

try:
    _read_orc(
        parts=[],
        engine=ArrowORCEngine,
        fs=None,
        schema={},
        index=index2,
        columns=columns2
    )
except:
    pass

print(f"After _read_orc: {columns2}")
print("Index was added to the list!")
print()

# Test 3: What if dd.from_map is called multiple times with same columns?
print("=" * 60)
print("Test 3: Multiple dd.from_map calls (hypothetical)")
print("=" * 60)

# If someone passes the same columns list to multiple calls
shared_columns = ['col1', 'col2']
print(f"Shared columns list: {shared_columns}")

for i in range(3):
    # If dd.from_map doesn't make a copy and passes shared_columns directly
    try:
        _read_orc(
            parts=[],
            engine=ArrowORCEngine,
            fs=None,
            schema={},
            index=f'idx{i}',
            columns=shared_columns
        )
    except:
        pass
    print(f"After call {i+1}: {shared_columns}")

print("\nThis shows the accumulation problem!")