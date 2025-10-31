# Bug Report: pandas.api.interchange Categorical Missing Values Corrupted

**Target**: `pandas.core.interchange.from_dataframe.categorical_column_to_series`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Missing values in categorical columns are silently converted to actual category values during interchange protocol conversion, resulting in data corruption.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas.api.interchange import from_dataframe


@given(
    st.lists(st.sampled_from(['a', 'b', 'c']), min_size=5, max_size=20),
    st.integers(min_value=0, max_value=4)
)
def test_categorical_preserves_missing(categories, null_idx):
    codes = [0, 1, 2, -1, 0]
    cat = pd.Categorical.from_codes(codes, categories=['a', 'b', 'c'])
    df = pd.DataFrame({'cat': cat})

    result = from_dataframe(df.__dataframe__())

    assert result.isna().sum().sum() == df.isna().sum().sum()
```

**Failing input**: Any categorical with -1 codes (missing values)

## Reproducing the Bug

```python
import pandas as pd
import numpy as np
from pandas.api.interchange import from_dataframe

categories = ['a', 'b', 'c']
codes = np.array([0, 1, 2, -1, 0, 1], dtype='int8')

cat = pd.Categorical.from_codes(codes, categories=categories)
df = pd.DataFrame({'cat': cat})

print("Original:")
print(df)

result = from_dataframe(df.__dataframe__())

print("\nAfter interchange:")
print(result)
```

**Output:**
```
Original:
   cat
0    a
1    b
2    c
3  NaN    <- Missing value
4    a
5    b

After interchange:
  cat
0   a
1   b
2   c
3   c    <- Corrupted to 'c' instead of NaN!
4   a
5   b
```

## Why This Is A Bug

1. **Data corruption**: Missing values are silently converted to actual category values
2. **Silent failure**: No error or warning is raised
3. **Violates documented behavior**: The interchange protocol should preserve the data structure

The root cause is in `set_nulls` (line 521-522 of from_dataframe.py):

```python
if validity is None:
    return data  # Early return prevents USE_SENTINEL handling!
```

For categorical columns with `USE_SENTINEL` null type:
1. The validity buffer is None (sentinels don't need a separate validity buffer)
2. `set_nulls` returns early without checking for sentinel values
3. The -1 codes were already converted via modulo: `-1 % 3 = 2` → category 'c'
4. The sentinel handling (line 526-527) is never reached

## Fix

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -518,12 +518,12 @@ def set_nulls(
     np.ndarray or pd.Series
         Data with the nulls being set.
     """
-    if validity is None:
-        return data
     null_kind, sentinel_val = col.describe_null
     null_pos = None

     if null_kind == ColumnNullType.USE_SENTINEL:
+        if validity is not None:
+            return data  # Unexpected: USE_SENTINEL with validity buffer
         null_pos = pd.Series(data) == sentinel_val
     elif null_kind in (ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
         assert validity, "Expected to have a validity buffer for the mask"
@@ -536,6 +536,8 @@ def set_nulls(
     elif null_kind in (ColumnNullType.NON_NULLABLE, ColumnNullType.USE_NAN):
         pass
     else:
+        if validity is None:
+            return data
         raise NotImplementedError(f"Null kind {null_kind} is not yet supported.")

     if null_pos is not None and np.any(null_pos):
```

The fix removes the early return and handles `USE_SENTINEL` correctly by checking the data for sentinel values.