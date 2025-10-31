#!/usr/bin/env python3
"""Demonstration of the dask ORC columns mutation bug"""

from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine

# Test case 1: Basic mutation demonstration
print("=== Test Case 1: Basic Mutation ===")
columns = ['col1', 'col2']
print(f"Before calling _read_orc: columns = {columns}")
print(f"Object ID before: {id(columns)}")

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
    pass  # Expected to fail due to empty parts

print(f"After calling _read_orc: columns = {columns}")
print(f"Object ID after: {id(columns)}")
print(f"Mutation occurred: {columns == ['col1', 'col2', 'col1']}")
print()

# Test case 2: The specific failing example from Hypothesis
print("=== Test Case 2: Hypothesis Failing Input ===")
columns = ['0']
print(f"Before calling _read_orc: columns = {columns}")

try:
    _read_orc(
        parts=[],
        engine=ArrowORCEngine,
        fs=None,
        schema={},
        index='0',
        columns=columns
    )
except Exception as e:
    pass

print(f"After calling _read_orc: columns = {columns}")
print(f"Expected: ['0'], Got: {columns}")
print()

# Test case 3: Multiple calls accumulate mutations
print("=== Test Case 3: Accumulation of Mutations ===")
columns = ['col1', 'col2']
print(f"Initial columns: {columns}")

for i in range(3):
    try:
        _read_orc(
            parts=[],
            engine=ArrowORCEngine,
            fs=None,
            schema={},
            index=f'idx{i}',
            columns=columns
        )
    except Exception as e:
        pass
    print(f"After call {i+1}: columns = {columns}")

print()
print("=== Summary ===")
print("The _read_orc function mutates its input columns list by appending")
print("the index parameter to it. This violates Python's principle that")
print("functions should not mutate their inputs unless explicitly documented.")