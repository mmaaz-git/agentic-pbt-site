# Bug Report: pandas.api.interchange Silent Data Corruption in Boolean Columns with NA Values

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `from_dataframe` function silently converts NA (missing) values to False in nullable boolean columns, causing data corruption without any warning or error.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe
from hypothesis import given, strategies as st, settings


@given(st.lists(st.one_of(st.booleans(), st.none()), min_size=1, max_size=20))
@settings(max_examples=50)
def test_nullable_bool_dtype(values):
    """Test that nullable boolean columns preserve NA values through interchange protocol."""
    df = pd.DataFrame({'col': pd.array(values, dtype='boolean')})

    # Convert through interchange protocol
    interchange_obj = df.__dataframe__()
    result = from_dataframe(interchange_obj)

    # Check that the dataframes are equal
    pd.testing.assert_frame_equal(result, df)


if __name__ == "__main__":
    # Run the test
    test_nullable_bool_dtype()
```

<details>

<summary>
**Failing input**: `values=[None]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 25, in <module>
    test_nullable_bool_dtype()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 10, in test_nullable_bool_dtype
    @settings(max_examples=50)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 20, in test_nullable_bool_dtype
    pd.testing.assert_frame_equal(result, df)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
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
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 999, in assert_series_equal
    assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 421, in assert_attr_equal
    raise_assert_detail(obj, msg, left_attr, right_attr)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    raise AssertionError(msg)
AssertionError: Attributes of DataFrame.iloc[:, 0] (column name="col") are different

Attribute "dtype" are different
[left]:  bool
[right]: boolean
Falsifying example: test_nullable_bool_dtype(
    values=[None],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe

# Create a DataFrame with nullable boolean dtype containing NA values
df = pd.DataFrame({'col': pd.array([True, False, None], dtype='boolean')})
print("Original DataFrame:")
print(df)
print(f"DataFrame dtype: {df['col'].dtype}")
print(f"Has NA values? {df['col'].isna().any()}")
print(f"NA count: {df['col'].isna().sum()}")

# Convert to interchange format and back
print("\n--- Converting through interchange protocol ---")
interchange_obj = df.__dataframe__()
result = from_dataframe(interchange_obj)

print("\nAfter round-trip through interchange:")
print(result)
print(f"DataFrame dtype: {result['col'].dtype}")
print(f"Has NA values? {result['col'].isna().any()}")
print(f"NA count: {result['col'].isna().sum()}")

# Show individual values
print("\n--- Comparing individual values ---")
print("Original values:")
for i, val in enumerate(df['col']):
    print(f"  Index {i}: {val}")

print("\nRound-trip values:")
for i, val in enumerate(result['col']):
    print(f"  Index {i}: {val}")
```

<details>

<summary>
Silent data corruption: NA values become False
</summary>
```
Original DataFrame:
     col
0   True
1  False
2   <NA>
DataFrame dtype: boolean
Has NA values? True
NA count: 1

--- Converting through interchange protocol ---

After round-trip through interchange:
     col
0   True
1  False
2  False
DataFrame dtype: bool
Has NA values? False
NA count: 0

--- Comparing individual values ---
Original values:
  Index 0: True
  Index 1: False
  Index 2: <NA>

Round-trip values:
  Index 0: True
  Index 1: False
  Index 2: False
```
</details>

## Why This Is A Bug

This is a **critical data corruption bug** that violates fundamental data integrity principles:

1. **Silent Data Corruption**: NA (missing/unknown) values are converted to False (a definite negative value) without any warning or error. This changes the semantic meaning of the data entirely - missing data should never be silently converted to valid data.

2. **Violates Round-Trip Property**: The interchange protocol should preserve data integrity. Converting `from_dataframe(df.__dataframe__())` should return an equivalent DataFrame, but instead it corrupts the data.

3. **Loss of Data Type Information**: The nullable 'boolean' dtype is converted to non-nullable 'bool' dtype, losing the ability to represent missing values.

4. **Dangerous in Production**: Since no error or warning is raised, users may not realize their data has been corrupted. This could lead to incorrect business decisions, faulty analyses, or wrong model predictions based on corrupted data.

5. **Inconsistent with Protocol Specification**: The interchange protocol has explicit support for null representation through validity buffers (as confirmed by the column having a USE_BYTEMASK null type), but the implementation fails to properly handle these nulls for boolean columns.

## Relevant Context

The root cause is in the `primitive_column_to_ndarray` function in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py`. When processing boolean columns:

1. The function correctly receives the validity buffer indicating which values are NA
2. It calls `set_nulls` to mark null positions in the array
3. However, numpy's bool dtype cannot represent NA values
4. The `set_nulls` function (lines 544-551) tries to handle this by casting to float when it encounters a TypeError, but for boolean columns this doesn't happen - the NA values are simply lost during the initial array creation

The interchange protocol column correctly reports:
- dtype: `(<DtypeKind.BOOL: 20>, 8, 'b', '|')`
- null info: `(<ColumnNullType.USE_BYTEMASK: 4>, 1)`
- Has a validity buffer to track NA positions

But the implementation doesn't preserve these NA values when converting back to pandas.

## Proposed Fix

The fix requires properly handling nullable boolean columns by using pandas' nullable boolean dtype instead of numpy's non-nullable bool dtype:

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -188,6 +188,25 @@ def primitive_column_to_ndarray(col: Column) -> tuple[np.ndarray, Any]:
     """
     Convert a column holding one of the primitive dtypes to a NumPy array.

+    A primitive type is one of: int, uint, float, bool.
+
+    Parameters
+    ----------
+    col : Column
+
+    Returns
+    -------
+    tuple
+        Tuple of np.ndarray holding the data and the memory owner object
+        that keeps the memory alive.
+    """
+    buffers = col.get_buffers()
+
+    # Special handling for boolean columns with nulls
+    null_kind, _ = col.describe_null
+    if col.dtype[0] == DtypeKind.BOOL and null_kind != ColumnNullType.NON_NULLABLE:
+        return _nullable_boolean_column_to_array(col, buffers)
+
     data_buff, data_dtype = buffers["data"]
     data = buffer_to_ndarray(
         data_buff, data_dtype, offset=col.offset, length=col.size()
@@ -196,6 +215,36 @@ def primitive_column_to_ndarray(col: Column) -> tuple[np.ndarray, Any]:
     data = set_nulls(data, col, buffers["validity"])
     return data, buffers

+def _nullable_boolean_column_to_array(col: Column, buffers) -> tuple[pd.array, Any]:
+    """
+    Convert a nullable boolean column to pandas BooleanArray.
+
+    Parameters
+    ----------
+    col : Column
+    buffers : dict
+        Buffers from col.get_buffers()
+
+    Returns
+    -------
+    tuple
+        Tuple of pd.array with boolean dtype and the buffers
+    """
+    data_buff, data_dtype = buffers["data"]
+    # Get the boolean data as numpy array
+    data = buffer_to_ndarray(
+        data_buff, data_dtype, offset=col.offset, length=col.size()
+    )
+
+    # Create a mask for NA values from validity buffer
+    validity = buffers.get("validity")
+    if validity is not None:
+        valid_buff, valid_dtype = validity
+        mask = ~buffer_to_ndarray(
+            valid_buff, valid_dtype, offset=col.offset, length=col.size()
+        ).astype(bool)
+    else:
+        mask = np.zeros(len(data), dtype=bool)
+
+    # Create pandas BooleanArray with mask
+    return pd.array(data, dtype="boolean").where(~mask), buffers
```