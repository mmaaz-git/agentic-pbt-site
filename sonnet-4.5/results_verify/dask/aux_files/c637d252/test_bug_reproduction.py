#!/usr/bin/env python3
"""Test script to reproduce the reported bug in dask._read_orc"""

from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine

# Test 1: Simple reproduction
print("=" * 60)
print("Test 1: Simple reproduction of the mutation")
print("=" * 60)

columns = ['col1', 'col2']
print(f"Before: {columns}")
print(f"columns id: {id(columns)}")

try:
    _read_orc(
        parts=[],
        engine=ArrowORCEngine,
        fs=None,
        schema={},
        index='col1',
        columns=columns
    )
except Exception as e:
    print(f"Exception occurred (expected): {e}")

print(f"After: {columns}")
print(f"columns id: {id(columns)} (same object)")
print()

# Test 2: The specific failing case from hypothesis
print("=" * 60)
print("Test 2: Failing hypothesis test case")
print("=" * 60)

columns2 = ['0']
columns2_before = list(columns2)
print(f"Before: {columns2}")

try:
    _read_orc(
        parts=[],
        engine=ArrowORCEngine,
        fs=None,
        schema={},
        index='0',
        columns=columns2
    )
except Exception as e:
    print(f"Exception occurred (expected): {e}")

print(f"After: {columns2}")
print(f"Was mutated: {columns2 != columns2_before}")
print(f"Expected: {columns2_before}, Got: {columns2}")
print()

# Test 3: Multiple calls showing accumulation
print("=" * 60)
print("Test 3: Multiple calls with same list")
print("=" * 60)

columns3 = ['col1', 'col2']
print(f"Initial: {columns3}")

for i in range(3):
    try:
        _read_orc(
            parts=[],
            engine=ArrowORCEngine,
            fs=None,
            schema={},
            index=f'idx{i}',
            columns=columns3
        )
    except:
        pass
    print(f"After call {i+1}: {columns3}")

print()

# Test 4: Check with None columns
print("=" * 60)
print("Test 4: With columns=None")
print("=" * 60)

try:
    result = _read_orc(
        parts=[],
        engine=ArrowORCEngine,
        fs=None,
        schema={},
        index='idx',
        columns=None
    )
    print("No mutation when columns=None (expected)")
except Exception as e:
    print(f"Exception occurred: {e}")