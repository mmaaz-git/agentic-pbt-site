#!/usr/bin/env python3
"""Test to reproduce the sorted_columns TypeError with None values bug."""

import sys
import traceback
from hypothesis import given, strategies as st, settings
import pytest

# Import the function from dask
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')
from dask.dataframe.io.parquet.core import sorted_columns

print("=" * 60)
print("TESTING SORTED_COLUMNS BUG REPRODUCTION")
print("=" * 60)

# First, test the specific failing cases mentioned in the bug report
print("\n1. Testing first failing case from bug report:")
print("   Input: [{'columns': [{'name': 'test_col', 'min': None, 'max': None}]},")
print("          {'columns': [{'name': 'test_col', 'min': 0, 'max': None}]}]")

statistics1 = [
    {'columns': [{'name': 'test_col', 'min': None, 'max': None}]},
    {'columns': [{'name': 'test_col', 'min': 0, 'max': None}]}
]

try:
    result = sorted_columns(statistics1, columns=['test_col'])
    print(f"   Result: {result}")
    print("   ERROR: Expected TypeError but function succeeded!")
except TypeError as e:
    print(f"   Got expected TypeError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"   Got unexpected error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n2. Testing alternative failing case from bug report:")
print("   Input: [{'columns': [{'name': 'test_col', 'min': 0, 'max': None}]}]")

statistics2 = [
    {'columns': [{'name': 'test_col', 'min': 0, 'max': None}]}
]

try:
    result = sorted_columns(statistics2, columns=['test_col'])
    print(f"   Result: {result}")
    print("   ERROR: Expected TypeError but function succeeded!")
except TypeError as e:
    print(f"   Got expected TypeError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"   Got unexpected error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n3. Testing a case with only max=None in second row group:")
statistics3 = [
    {'columns': [{'name': 'test_col', 'min': 0, 'max': 10}]},
    {'columns': [{'name': 'test_col', 'min': 5, 'max': None}]}
]

try:
    result = sorted_columns(statistics3, columns=['test_col'])
    print(f"   Result: {result}")
except Exception as e:
    print(f"   Got error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n4. Testing valid case (should work):")
statistics4 = [
    {'columns': [{'name': 'test_col', 'min': 0, 'max': 10}]},
    {'columns': [{'name': 'test_col', 'min': 15, 'max': 20}]}
]

try:
    result = sorted_columns(statistics4, columns=['test_col'])
    print(f"   Result: {result}")
except Exception as e:
    print(f"   Got unexpected error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n5. Running property-based test from bug report:")

@st.composite
def statistics_with_nones(draw):
    num_row_groups = draw(st.integers(min_value=1, max_value=10))
    col_name = "test_col"

    statistics = []
    for i in range(num_row_groups):
        has_min = draw(st.booleans())
        has_max = draw(st.booleans())

        min_val = draw(st.integers(min_value=-100, max_value=100)) if has_min else None
        max_val = draw(st.integers(min_value=-100, max_value=100)) if has_max else None

        if min_val is not None and max_val is not None and min_val > max_val:
            min_val, max_val = max_val, min_val

        statistics.append({
            "columns": [{
                "name": col_name,
                "min": min_val,
                "max": max_val
            }]
        })

    return statistics, col_name

@given(statistics_with_nones())
@settings(max_examples=50)  # Reduced for faster testing
def test_sorted_columns_none_handling(data):
    statistics, col_name = data
    try:
        result = sorted_columns(statistics, columns=[col_name])

        for item in result:
            divisions = item["divisions"]
            assert None not in divisions, f"Found None in divisions: {divisions}"
            assert divisions == sorted(divisions), f"Divisions not sorted: {divisions}"
    except TypeError as e:
        # If we get a TypeError, this confirms the bug
        print(f"\n   Found bug with input: {statistics}")
        print(f"   Error: {e}")
        raise

try:
    test_sorted_columns_none_handling()
    print("   Property-based test completed without finding issues")
except Exception as e:
    print(f"   Property-based test found issues (as expected)")

print("\n" + "=" * 60)
print("BUG REPRODUCTION COMPLETE")
print("=" * 60)