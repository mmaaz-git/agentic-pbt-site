#!/usr/bin/env python3
"""Test script to reproduce the _read_orc mutation bug"""

from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine

# Test 1: Simple reproduction
print("Test 1: Simple reproduction")
columns_list = ['col1', 'col2']
original = columns_list.copy()

parts = [("dummy_path", [0])]
try:
    _read_orc(parts, engine=ArrowORCEngine, fs=None, schema={}, index='col1', columns=columns_list)
except:
    pass

print(f"Before: {original}")
print(f"After:  {columns_list}")
print(f"Mutated: {columns_list != original}")
print()

# Test 2: With example from bug report
print("Test 2: Example from bug report (column_names=['a'], index_name='a')")
columns_list = ['a']
original = columns_list.copy()

parts = [("dummy_path", [0])]
try:
    _read_orc(parts, engine=ArrowORCEngine, fs=None, schema={}, index='a', columns=columns_list)
except:
    pass

print(f"Before: {original}")
print(f"After:  {columns_list}")
print(f"Mutated: {columns_list != original}")