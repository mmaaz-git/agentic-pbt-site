# Bug Report: pandas.api.interchange Large Integer Round-Trip Failure

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The interchange protocol fails to round-trip pandas DataFrames containing integers that exceed int64 range, raising `NotImplementedError: Non-string object dtypes are not supported yet`. This violates the expected behavior that a pandas DataFrame should be convertible through the interchange protocol and back.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas.api.interchange import from_dataframe

@given(
    st.lists(st.integers(), min_size=0, max_size=100),
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=0, max_size=100),
)
def test_from_dataframe_roundtrip_preserves_data(int_col, float_col):
    assume(len(int_col) == len(float_col))

    original_df = pd.DataFrame({
        'int_col': int_col,
        'float_col': float_col
    })

    interchange_obj = original_df.__dataframe__()
    result_df = from_dataframe(interchange_obj)

    pd.testing.assert_frame_equal(original_df, result_df)
```

**Failing input**: `int_col=[0, -9_223_372_036_854_775_809], float_col=[0.0, 0.0]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({'int_col': [0, -9_223_372_036_854_775_809]})
interchange_obj = df.__dataframe__()
result_df = from_dataframe(interchange_obj)
```

**Output:**
```
NotImplementedError: Non-string object dtypes are not supported yet
```

## Why This Is A Bug

The value `-9_223_372_036_854_775_809` is one less than the minimum int64 value (`-2**63 = -9_223_372_036_854_775_808`), causing pandas to silently convert the column to object dtype to accommodate the Python arbitrary-precision integer. The interchange protocol then fails because it doesn't support non-string object dtypes.

This violates the round-trip property that should hold for pandas DataFrames: `from_dataframe(df.__dataframe__()) == df`. The `__dataframe__` docstring states it should work for "any dataframe" with no mention of limitations for large integers.

## Fix

The issue is in `/pandas/core/interchange/column.py` at line 136-144:

```python
elif is_string_dtype(dtype):
    if infer_dtype(self._col) in ("string", "empty"):
        return (
            DtypeKind.STRING,
            8,
            dtype_to_arrow_c_fmt(dtype),
            Endianness.NATIVE,
        )
    raise NotImplementedError("Non-string object dtypes are not supported yet")
```

A fix would be to detect integer-typed object columns and provide a better error message or handle them specially:

```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -136,7 +136,12 @@ class PandasColumn(Column):
         elif is_string_dtype(dtype):
             if infer_dtype(self._col) in ("string", "empty"):
                 return (
                     DtypeKind.STRING,
                     8,
                     dtype_to_arrow_c_fmt(dtype),
                     Endianness.NATIVE,
                 )
-            raise NotImplementedError("Non-string object dtypes are not supported yet")
+            inferred_type = infer_dtype(self._col)
+            if inferred_type == "integer":
+                raise ValueError(
+                    "Column contains integers that exceed int64/uint64 range. "
+                    "The interchange protocol does not support arbitrary-precision integers. "
+                    "Consider converting to float64 or using a different data type."
+                )
+            raise NotImplementedError(f"Object dtype with inferred type '{inferred_type}' is not supported yet")
         else:
             return self._dtype_from_pandasdtype(dtype)
```

This provides a clearer error message explaining the limitation, rather than the generic "Non-string object dtypes are not supported yet" message.