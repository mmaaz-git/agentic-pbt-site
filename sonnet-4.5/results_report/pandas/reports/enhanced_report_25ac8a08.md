# Bug Report: pandas.api.interchange Boolean Null Values Silently Converted to False

**Target**: `pandas.core.interchange.from_dataframe.set_nulls`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When using the DataFrame interchange protocol, null values in nullable boolean columns are silently converted to `False` instead of being preserved as null/NA, causing data corruption without warning.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas.api.interchange import from_dataframe


@given(st.lists(st.one_of(st.booleans(), st.none()), min_size=1, max_size=100))
def test_round_trip_nullable_bool(bool_list):
    df = pd.DataFrame({"col": pd.array(bool_list, dtype="boolean")})
    result = from_dataframe(df.__dataframe__())
    pd.testing.assert_frame_equal(result, df)

if __name__ == "__main__":
    test_round_trip_nullable_bool()
```

<details>

<summary>
**Failing input**: `bool_list=[None]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 13, in <module>
    test_round_trip_nullable_bool()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 7, in test_round_trip_nullable_bool
    def test_round_trip_nullable_bool(bool_list):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 10, in test_round_trip_nullable_bool
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
Falsifying example: test_round_trip_nullable_bool(
    bool_list=[None],
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({"col": pd.array([True, False, None], dtype="boolean")})
print("Original:", df["col"].tolist())

result = from_dataframe(df.__dataframe__())
print("After round-trip:", result["col"].tolist())
```

<details>

<summary>
Null values are converted to False
</summary>
```
Original: [True, False, <NA>]
After round-trip: [True, False, False]
```
</details>

## Why This Is A Bug

The DataFrame interchange protocol is designed to enable data exchange between different DataFrame libraries while preserving data integrity. According to the interchange protocol specification, null values should be preserved during round-trip conversions.

The bug occurs in the `set_nulls` function in `pandas/core/interchange/from_dataframe.py` (lines 494-557). This function attempts to handle null values by assigning `None` to positions marked as null:

```python
try:
    data[null_pos] = None
except TypeError:
    # If TypeError, convert to float to support nulls
    data = data.astype(float)
    data[null_pos] = None
```

The logic assumes that non-nullable dtypes will raise a `TypeError` when attempting to assign `None`. However, NumPy's boolean dtype has unexpected behavior - it silently converts `None` to `False` instead of raising an error. This causes:

1. **Silent data corruption**: Null values become `False` with no warning
2. **Loss of data type information**: The nullable `boolean` dtype is converted to non-nullable `bool`
3. **Violation of interchange protocol contract**: The protocol guarantees that nulls are preserved

## Relevant Context

This bug specifically affects:
- Users of the DataFrame interchange protocol (`__dataframe__()` and `from_dataframe()`)
- DataFrames with nullable boolean columns (dtype `"boolean"`)
- Data exchange between pandas and other DataFrame libraries (Polars, Vaex, etc.)

The interchange protocol documentation states that implementations should preserve null values: https://data-apis.org/dataframe-protocol/latest/API.html

The root cause is NumPy's boolean array behavior:
```python
>>> import numpy as np
>>> arr = np.array([True, False], dtype=bool)
>>> arr[1] = None  # Expected: TypeError, Actual: silently converts to False
>>> arr
array([True, False])
```

## Proposed Fix

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -541,6 +541,11 @@ def set_nulls(
     if null_pos is not None and np.any(null_pos):
         if not allow_modify_inplace:
             data = data.copy()
+
+        # NumPy bool dtype silently converts None to False instead of raising TypeError
+        # Check for bool dtype explicitly and convert to float to support nulls
+        if isinstance(data, np.ndarray) and data.dtype == np.bool_:
+            data = data.astype(float)
+
         try:
             data[null_pos] = None
         except TypeError:
```