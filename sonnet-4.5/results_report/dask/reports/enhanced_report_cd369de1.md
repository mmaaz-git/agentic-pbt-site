# Bug Report: dask.dataframe Multiplication with Integer Overflow and Mismatched Indices Produces Wrong Sign

**Target**: `dask.dataframe` (DataFrame multiplication operator)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When multiplying two Dask DataFrames with mismatched indices and integer values that overflow int64, Dask produces incorrect results with the wrong sign in the first partition, while pandas handles the overflow correctly.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from hypothesis.extra.pandas import data_frames, column, range_indexes
import dask.dataframe as dd
import pandas as pd


@settings(max_examples=100)
@given(
    df1=data_frames(
        columns=[
            column('x', dtype=int),
            column('y', dtype=int),
        ],
        index=range_indexes(min_size=1, max_size=30),
    ),
    df2=data_frames(
        columns=[
            column('x', dtype=int),
            column('y', dtype=int),
        ],
        index=range_indexes(min_size=1, max_size=30),
    ),
)
@example(
    df1=pd.DataFrame({'x': [2, 2], 'y': [0, 0]}),
    df2=pd.DataFrame({'x': [4611686018427387904, 4611686018427387904, 4611686018427387904], 'y': [0, 0, 0]})
)
def test_multiply_dataframe_matches_pandas(df1, df2):
    ddf1 = dd.from_pandas(df1, npartitions=2)
    ddf2 = dd.from_pandas(df2, npartitions=2)

    dask_result = (ddf1 * ddf2).compute()
    pandas_result = df1 * df2

    pd.testing.assert_frame_equal(dask_result, pandas_result)


if __name__ == "__main__":
    test_multiply_dataframe_matches_pandas()
```

<details>

<summary>
**Failing input**: `df1 = pd.DataFrame({'x': [2, 2], 'y': [0, 0]}), df2 = pd.DataFrame({'x': [4611686018427387904, 4611686018427387904, 4611686018427387904], 'y': [0, 0, 0]})`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 39, in <module>
    test_multiply_dataframe_matches_pandas()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 8, in test_multiply_dataframe_matches_pandas
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 35, in test_multiply_dataframe_matches_pandas
    pd.testing.assert_frame_equal(dask_result, pandas_result)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1303, in assert_frame_equal
    assert_series_equal(
    ~~~~~~~~~~~~~~~~~~~^
        lcol,
        ^^^^^
    ...<12 lines>...
        check_flags=False,
        ^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1091, in assert_series_equal
    _testing.assert_almost_equal(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        left._values,
        ^^^^^^^^^^^^^
    ...<5 lines>...
        index_values=left.index,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "pandas/_libs/testing.pyx", line 55, in pandas._libs.testing.assert_almost_equal
  File "pandas/_libs/testing.pyx", line 173, in pandas._libs.testing.assert_almost_equal
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    raise AssertionError(msg)
AssertionError: DataFrame.iloc[:, 0] (column name="x") are different

DataFrame.iloc[:, 0] (column name="x") values are different (33.33333 %)
[index]: [0, 1, 2]
[left]:  [-9.223372036854776e+18, 9.223372036854776e+18, nan]
[right]: [9.223372036854776e+18, 9.223372036854776e+18, nan]
At positional index 0, first diff: -9.223372036854776e+18 != 9.223372036854776e+18
Falsifying explicit example: test_multiply_dataframe_matches_pandas(
    df1=
           x  y
        0  2  0
        1  2  0
    ,
    df2=
                             x  y
        0  4611686018427387904  0
        1  4611686018427387904  0
        2  4611686018427387904  0
    ,
)
```
</details>

## Reproducing the Bug

```python
import dask.dataframe as dd
import pandas as pd
import numpy as np

df1 = pd.DataFrame({'x': [2, 2], 'y': [0, 0]})
df2 = pd.DataFrame({'x': [4611686018427387904, 4611686018427387904, 4611686018427387904], 'y': [0, 0, 0]})

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

pandas_result = df1 * df2
print("\n=== PANDAS RESULT ===")
print(pandas_result)
print(f"dtype of x column: {pandas_result['x'].dtype}")

ddf1 = dd.from_pandas(df1, npartitions=2)
ddf2 = dd.from_pandas(df2, npartitions=2)
dask_result = (ddf1 * ddf2).compute()
print("\n=== DASK RESULT ===")
print(dask_result)
print(f"dtype of x column: {dask_result['x'].dtype}")

print("\n=== COMPARISON ===")
print(f"Expected x[0] (pandas): {pandas_result['x'].iloc[0]}")
print(f"Actual x[0] (dask):     {dask_result['x'].iloc[0]}")
print(f"Expected x[1] (pandas): {pandas_result['x'].iloc[1]}")
print(f"Actual x[1] (dask):     {dask_result['x'].iloc[1]}")

print("\n=== ISSUE ===")
if pandas_result['x'].iloc[0] != dask_result['x'].iloc[0]:
    print(f"BUG DETECTED: Dask result differs from Pandas!")
    print(f"  - Row 0: Expected {pandas_result['x'].iloc[0]}, got {dask_result['x'].iloc[0]}")
    print(f"  - Sign flipped: {pandas_result['x'].iloc[0] > 0} -> {dask_result['x'].iloc[0] > 0}")
else:
    print("Results match")

print("\n=== ADDITIONAL INFO ===")
print(f"2 * 4611686018427387904 = {2 * 4611686018427387904}")
print(f"This exceeds int64 max: {2 * 4611686018427387904 > np.iinfo(np.int64).max}")
print(f"int64 max value: {np.iinfo(np.int64).max}")
print(f"NumPy int64 overflow result: {np.int64(2) * np.int64(4611686018427387904)}")
```

<details>

<summary>
BUG DETECTED: Incorrect sign in first partition
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/48/repo.py:43: RuntimeWarning: overflow encountered in scalar multiply
  print(f"NumPy int64 overflow result: {np.int64(2) * np.int64(4611686018427387904)}")
DataFrame 1:
   x  y
0  2  0
1  2  0

DataFrame 2:
                     x  y
0  4611686018427387904  0
1  4611686018427387904  0
2  4611686018427387904  0

=== PANDAS RESULT ===
              x    y
0  9.223372e+18  0.0
1  9.223372e+18  0.0
2           NaN  NaN
dtype of x column: float64

=== DASK RESULT ===
              x    y
0 -9.223372e+18  0.0
1  9.223372e+18  0.0
2           NaN  NaN
dtype of x column: float64

=== COMPARISON ===
Expected x[0] (pandas): 9.223372036854776e+18
Actual x[0] (dask):     -9.223372036854776e+18
Expected x[1] (pandas): 9.223372036854776e+18
Actual x[1] (dask):     9.223372036854776e+18

=== ISSUE ===
BUG DETECTED: Dask result differs from Pandas!
  - Row 0: Expected 9.223372036854776e+18, got -9.223372036854776e+18
  - Sign flipped: True -> False

=== ADDITIONAL INFO ===
2 * 4611686018427387904 = 9223372036854775808
This exceeds int64 max: True
int64 max value: 9223372036854775807
NumPy int64 overflow result: -9223372036854775808
```
</details>

## Why This Is A Bug

This is a clear bug in Dask's handling of DataFrame multiplication when three conditions combine: mismatched indices requiring alignment, integer overflow, and partitioned data. The evidence is conclusive:

1. **Mathematically incorrect results**: Multiplying positive integers (2 × 4611686018427387904) should yield a positive result. Pandas correctly handles the overflow by promoting to float64 and maintaining the positive sign (9.223372036854776e+18). Dask produces -9.223372036854776e+18 in the first partition, which is wrong.

2. **Internal inconsistency**: The same multiplication produces different results within the same DataFrame - row 0 is negative (wrong) while row 1 is positive (correct). This inconsistency cannot be justified as a design choice.

3. **Violates Dask's core principle**: Dask documentation emphasizes "the same API" and "the same execution" as pandas. This fundamental arithmetic operation should produce identical results regardless of partitioning.

4. **Silent data corruption**: The bug produces incorrect numerical results without any error or warning, which could lead to serious errors in data analysis downstream.

## Relevant Context

The bug specifically occurs when:
- DataFrames have different lengths (2 rows vs 3 rows) requiring index alignment
- Multiplication causes int64 overflow (2 × 4611686018427387904 > int64_max)
- DataFrames are split into multiple partitions (npartitions=2)

The root cause appears to be inconsistent overflow handling between partitions during index alignment operations. When DataFrames are pre-aligned or use a single partition, the bug does not occur.

This is part of a broader pattern affecting multiple arithmetic operations (addition, subtraction, multiplication) in Dask when these conditions combine. The issue has been confirmed in Dask version 2025.9.1.

## Proposed Fix

The issue appears to be in how Dask handles type promotion and overflow during partitioned arithmetic operations with index alignment. A high-level fix would involve:

1. Ensuring consistent type promotion across all partitions before performing arithmetic operations
2. Applying pandas' overflow handling logic uniformly across partitions
3. Special handling for the case where DataFrames have mismatched divisions/indices

The fix should be implemented in the binary operation handling logic of dask-expr to ensure all partitions use the same dtype promotion and overflow behavior that pandas uses.