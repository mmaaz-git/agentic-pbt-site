# Bug Report: pandas DataFrame.T.T Loses Integer Dtype for Empty DataFrames

**Target**: `pandas.DataFrame.T` (transpose operation)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Double transposing an empty DataFrame with integer columns converts int64 dtypes to float64, violating the mathematical property that transpose is an involution and contradicting pandas documentation that states homogeneous dtypes should be preserved.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, settings
from hypothesis.extra import pandas as pdst


@given(pdst.data_frames(columns=[
    pdst.column('A', dtype=int),
    pdst.column('B', dtype=int)
]))
@settings(max_examples=500)
def test_transpose_involution_preserves_dtype(df):
    result = df.T.T
    for col in df.columns:
        assert df[col].dtype == result[col].dtype, f"Column {col} dtype changed from {df[col].dtype} to {result[col].dtype}"

# Run the test
if __name__ == "__main__":
    test_transpose_involution_preserves_dtype()
```

<details>

<summary>
**Failing input**: `Empty DataFrame with columns A and B of dtype int64`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 18, in <module>
    test_transpose_involution_preserves_dtype()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 7, in test_transpose_involution_preserves_dtype
    pdst.column('A', dtype=int),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 14, in test_transpose_involution_preserves_dtype
    assert df[col].dtype == result[col].dtype, f"Column {col} dtype changed from {df[col].dtype} to {result[col].dtype}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Column A dtype changed from int64 to float64
Falsifying example: test_transpose_involution_preserves_dtype(
    df=
        Empty DataFrame
        Columns: [A, B]
        Index: []
    ,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd

# Create an empty DataFrame with integer columns
df = pd.DataFrame({'A': pd.Series([], dtype='int64'), 'B': pd.Series([], dtype='int64')})

print("Original DataFrame:")
print(f"df.shape: {df.shape}")
print(f"df.dtypes:\n{df.dtypes}")
print(f"df:\n{df}")

print("\n" + "="*50)
print("After single transpose (df.T):")
transposed_once = df.T
print(f"df.T.shape: {transposed_once.shape}")
print(f"df.T.dtypes:\n{transposed_once.dtypes}")
print(f"df.T:\n{transposed_once}")

print("\n" + "="*50)
print("After double transpose (df.T.T):")
result = df.T.T
print(f"df.T.T.shape: {result.shape}")
print(f"df.T.T.dtypes:\n{result.dtypes}")
print(f"df.T.T:\n{result}")

print("\n" + "="*50)
print("Comparison:")
print(f"Original dtypes: {dict(df.dtypes)}")
print(f"After df.T.T dtypes: {dict(result.dtypes)}")
print(f"Dtypes preserved? {df.dtypes.equals(result.dtypes)}")
```

<details>

<summary>
Output showing dtype change from int64 to float64
</summary>
```
Original DataFrame:
df.shape: (0, 2)
df.dtypes:
A    int64
B    int64
dtype: object
df:
Empty DataFrame
Columns: [A, B]
Index: []

==================================================
After single transpose (df.T):
df.T.shape: (2, 0)
df.T.dtypes:
Series([], dtype: object)
df.T:
Empty DataFrame
Columns: []
Index: [A, B]

==================================================
After double transpose (df.T.T):
df.T.T.shape: (0, 2)
df.T.T.dtypes:
A    float64
B    float64
dtype: object
df.T.T:
Empty DataFrame
Columns: [A, B]
Index: []

==================================================
Comparison:
Original dtypes: {'A': dtype('int64'), 'B': dtype('int64')}
After df.T.T dtypes: {'A': dtype('float64'), 'B': dtype('float64')}
Dtypes preserved? False
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Mathematical Property Violation**: The transpose operation should be an involution, meaning T(T(df)) = df for all DataFrames. This includes preserving data types, not just values.

2. **Documentation Contradiction**: The pandas documentation for DataFrame.transpose() explicitly states: "When the dtype is homogeneous in the original DataFrame, we get a transposed DataFrame with the same dtype." An empty DataFrame with all int64 columns IS homogeneous, so dtypes should be preserved.

3. **Inconsistent Behavior**: Non-empty DataFrames correctly preserve int64 dtype through double transpose. Only empty DataFrames exhibit this bug, creating an inconsistency.

4. **Root Cause**: The bug occurs because when pandas creates a DataFrame from an empty numpy array with shape (n, 0) where n > 0, it defaults to float64 regardless of the input array's dtype. However, when the shape is (0, n), it correctly preserves the dtype. This asymmetric behavior causes the transpose to be non-involutive.

## Relevant Context

- The bug specifically affects the fast transpose path in pandas/core/frame.py when `_can_fast_transpose` returns True
- During the first transpose (df.T), an empty int64 array of shape (0, 2) becomes shape (2, 0) - still int64
- When constructing a new DataFrame from this (2, 0) shaped int64 array, pandas incorrectly converts to float64
- Similar issues have been reported and fixed before (e.g., GitHub issue #22858 about empty DataFrame dtype preservation)
- The issue is in DataFrame construction logic, not in the transpose operation itself

Code location: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/frame.py:3886-3898`

## Proposed Fix

The fix requires modifying the DataFrame constructor to preserve integer dtype when creating from empty arrays with 0 columns. The issue occurs in the internal DataFrame initialization when the shape is (n, 0) with n > 0.

A high-level approach to fix this:

1. In the DataFrame constructor or the underlying array management code, detect when creating a DataFrame from an empty array with 0 columns
2. Preserve the original dtype instead of defaulting to float64
3. Ensure this applies consistently to both (0, n) and (n, 0) shaped arrays

The specific fix would need to be in the pandas internals that handle empty DataFrame creation, likely in the BlockManager or array construction logic that currently has an asymmetry between handling (0, n) vs (n, 0) shaped arrays.