#!/usr/bin/env python3
"""Test to reproduce the sorted_columns bug with None values"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.dataframe.io.parquet.core import sorted_columns

# Test case 1: The reported failing case
print("Test 1: Reproducing the reported bug")
print("=" * 50)
statistics = [{'columns': [{'name': '0', 'min': 0, 'max': None}]}]
print(f"Input: statistics={statistics}")

try:
    result = sorted_columns(statistics)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n")

# Test case 2: Normal case with valid values
print("Test 2: Normal case with valid min/max")
print("=" * 50)
statistics = [{'columns': [{'name': '0', 'min': 0, 'max': 10}]}]
print(f"Input: statistics={statistics}")

try:
    result = sorted_columns(statistics)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n")

# Test case 3: Both min and max are None
print("Test 3: Both min and max are None")
print("=" * 50)
statistics = [{'columns': [{'name': '0', 'min': None, 'max': None}]}]
print(f"Input: statistics={statistics}")

try:
    result = sorted_columns(statistics)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n")

# Test case 4: min is None, max is valid
print("Test 4: min is None, max is valid")
print("=" * 50)
statistics = [{'columns': [{'name': '0', 'min': None, 'max': 10}]}]
print(f"Input: statistics={statistics}")

try:
    result = sorted_columns(statistics)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n")

# Test case 5: Multiple row groups with None in later statistics
print("Test 5: Multiple row groups - second has None max")
print("=" * 50)
statistics = [
    {'columns': [{'name': '0', 'min': 0, 'max': 5}]},
    {'columns': [{'name': '0', 'min': 6, 'max': None}]}
]
print(f"Input: statistics={statistics}")

try:
    result = sorted_columns(statistics)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")