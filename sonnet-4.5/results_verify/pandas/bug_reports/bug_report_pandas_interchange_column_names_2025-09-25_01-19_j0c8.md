# Bug Report: pandas.core.interchange Column Name Type Conversion

**Target**: `pandas.core.interchange`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The DataFrame interchange protocol converts non-string column names to strings during `__dataframe__()` creation, but doesn't restore the original column name types when converting back via `from_dataframe()`. This violates the roundtrip property: `from_dataframe(df.__dataframe__()) ≠ df` when column names are not strings.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas.core.interchange.from_dataframe import from_dataframe

@given(
    st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=10),
    st.integers(min_value=0, max_value=100)
)
def test_roundtrip_preserves_numeric_column_names(data, col_name):
    df = pd.DataFrame({col_name: data})
    result = from_dataframe(df.__dataframe__())
    pd.testing.assert_frame_equal(result, df)
```

**Failing input**: `col_name=0, data=[1, 2, 3]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.interchange.from_dataframe import from_dataframe

df = pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6]})
print("Original column names:", df.columns.tolist())
print("Original column types:", [type(c).__name__ for c in df.columns])

result = from_dataframe(df.__dataframe__())
print("Result column names:", result.columns.tolist())
print("Result column types:", [type(c).__name__ for c in result.columns])
```

Output:
```
Original column names: [0, 1]
Original column types: ['int', 'int']
Result column names: ['0', '1']
Result column types: ['str', 'str']
```

## Why This Is A Bug

1. **Violates roundtrip property**: Users expect `from_dataframe(df.__dataframe__())` to return an equivalent DataFrame
2. **Silent data mutation**: Column name types change without warning
3. **Breaks downstream code**: Code that relies on column name types (e.g., `isinstance(df.columns[0], int)`) will break
4. **Inconsistent with index handling**: The protocol already preserves the pandas index in metadata but not column names

## Fix

The root cause is in `pandas/core/interchange/dataframe.py:36`:
```python
self._df = df.rename(columns=str, copy=False)
```

This converts column names to strings (required by the interchange protocol) but doesn't preserve the original column names in metadata.

**Proposed fix**:

```diff
--- a/pandas/core/interchange/dataframe.py
+++ b/pandas/core/interchange/dataframe.py
@@ -31,7 +31,8 @@ class PandasDataFrameXchg(DataFrameXchg):
     def __init__(self, df: DataFrame, allow_copy: bool = True) -> None:
         Constructor - an instance of this (private) class is returned from
         `pd.DataFrame.__dataframe__`.
+        self._original_columns = df.columns
         self._df = df.rename(columns=str, copy=False)
         self._allow_copy = allow_copy
         for i, _col in enumerate(self._df.columns):
@@ -50,7 +51,10 @@ class PandasDataFrameXchg(DataFrameXchg):
     def metadata(self) -> dict[str, Index]:
         # `index` isn't a regular column, and the protocol doesn't support row
         # labels - so we export it as Pandas-specific metadata here.
-        return {"pandas.index": self._df.index}
+        return {
+            "pandas.index": self._df.index,
+            "pandas.columns": self._original_columns
+        }
```

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -137,6 +137,10 @@ def _from_dataframe(df: DataFrameXchg, allow_copy: bool = True):
     index_obj = df.metadata.get("pandas.index", None)
     if index_obj is not None:
         pandas_df.index = index_obj
+
+    columns_obj = df.metadata.get("pandas.columns", None)
+    if columns_obj is not None:
+        pandas_df.columns = columns_obj

     return pandas_df
```
