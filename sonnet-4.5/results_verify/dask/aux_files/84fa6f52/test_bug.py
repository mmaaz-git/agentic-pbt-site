#!/usr/bin/env python3
"""Test the reported bug in FromArray._column_indices"""

import numpy as np
from hypothesis import given, strategies as st, settings

# First, let's run the Hypothesis test
print("=" * 60)
print("Running Hypothesis Test")
print("=" * 60)

from dask.dataframe.dask_expr.io.io import FromArray

@given(
    num_columns=st.integers(min_value=3, max_value=10)
)
@settings(max_examples=10)
def test_column_order_preservation(num_columns):
    arr = np.arange(num_columns * 2).reshape(2, num_columns)
    original_columns = [f'col_{i}' for i in range(num_columns)]
    requested_columns = [original_columns[-1], original_columns[0]]

    from_array = FromArray(
        frame=arr,
        chunksize=10,
        original_columns=original_columns,
        meta=None,
        columns=requested_columns
    )

    column_indices = from_array._column_indices
    expected_indices = [num_columns - 1, 0]

    print(f"Test with {num_columns} columns:")
    print(f"  Requested columns: {requested_columns}")
    print(f"  Got indices: {column_indices}")
    print(f"  Expected indices: {expected_indices}")
    print(f"  Match: {column_indices == expected_indices}")

    assert column_indices == expected_indices, f"Expected {expected_indices}, got {column_indices}"

try:
    test_column_order_preservation()
except AssertionError as e:
    print(f"\nHypothesis test FAILED: {e}")
except Exception as e:
    print(f"\nUnexpected error in Hypothesis test: {e}")

# Now run the manual reproduction
print("\n" + "=" * 60)
print("Manual Reproduction")
print("=" * 60)

arr = np.array([[10, 20, 30], [40, 50, 60]])
original_columns = ['a', 'b', 'c']
requested_columns = ['c', 'a']

from_array = FromArray(
    frame=arr,
    chunksize=10,
    original_columns=original_columns,
    meta=None,
    columns=requested_columns
)

column_indices = from_array._column_indices
print(f"Original array:\n{arr}")
print(f"Original columns: {original_columns}")
print(f"Requested columns: {requested_columns}")
print(f"Column indices returned: {column_indices}")
print(f"Expected indices: [2, 0]")

data_slice = arr[:, column_indices]
print(f"\nActual data slice (using returned indices): \n{data_slice}")
print(f"Expected data slice (columns c, a in that order): \n{arr[:, [2, 0]]}")

# Check if the data matches what we expect
if list(column_indices) == [2, 0]:
    print("\n✓ Column indices are CORRECT (preserve request order)")
else:
    print("\n✗ Column indices are WRONG (do not preserve request order)")

# Let's also check what pandas does
print("\n" + "=" * 60)
print("Pandas Comparison")
print("=" * 60)

import pandas as pd

df = pd.DataFrame(arr, columns=original_columns)
print(f"Original DataFrame:\n{df}")
print(f"\nSelecting columns ['c', 'a']:")
print(df[['c', 'a']])
print("\nNote: Pandas preserves the order of requested columns.")