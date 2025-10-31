# Bug Report: pandas.core.interchange Loses Nullable Integer Dtype

**Target**: `pandas.core.interchange.from_dataframe.set_nulls`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The pandas interchange protocol loses nullable integer dtypes (Int64, Int32, etc.) during round-trip conversion, incorrectly converting them to float64 when null values are present. This causes data type information loss.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd


@given(
    st.lists(
        st.one_of(
            st.integers(min_value=-2**31, max_value=2**31-1),
            st.none()
        ),
        min_size=1,
        max_size=100
    )
)
def test_nullable_integer_roundtrip(int_list):
    df = pd.DataFrame({"int_col": pd.array(int_list, dtype=pd.Int64Dtype())})

    interchange_obj = df.__dataframe__()
    result = pd.api.interchange.from_dataframe(interchange_obj)

    pd.testing.assert_frame_equal(result, df)
```

**Failing input**: `[None]` (a list with a single None value)

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({"int_col": pd.array([None], dtype=pd.Int64Dtype())})
print(f"Original dtype: {df['int_col'].dtype}")

interchange_obj = df.__dataframe__()
result = pd.api.interchange.from_dataframe(interchange_obj)

print(f"Result dtype: {result['int_col'].dtype}")
```

Output:
```
Original dtype: Int64
Result dtype: float64
```

Expected: `Int64`
Actual: `float64`

## Why This Is A Bug

1. **Data type information loss**: Nullable integer dtypes are a core pandas feature and should be preserved during interchange operations
2. **Violates round-trip property**: Converting DataFrame → interchange → DataFrame should preserve dtypes
3. **Incorrect fallback logic**: The code in `from_dataframe.py:550` catches `TypeError` when setting nulls and blindly converts to float, but pandas has nullable integer dtypes that can handle None values natively

The bug occurs in the `set_nulls` function at lines 544-551:

```python
try:
    data[null_pos] = None
except TypeError:
    # TypeError happens if the `data` dtype appears to be non-nullable
    # in numpy notation (bool, int, uint). If this happens,
    # cast the `data` to nullable float dtype.
    data = data.astype(float)  # <-- BUG: Should preserve nullable int dtypes
    data[null_pos] = None
```

## Fix

The fix should check if we're dealing with a nullable dtype and preserve it:

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -544,11 +544,21 @@ def set_nulls(
         try:
             data[null_pos] = None
         except TypeError:
-            # TypeError happens if the `data` dtype appears to be non-nullable
-            # in numpy notation (bool, int, uint). If this happens,
-            # cast the `data` to nullable float dtype.
-            data = data.astype(float)
-            data[null_pos] = None
+            # TypeError happens if the `data` dtype appears to be non-nullable
+            # in numpy notation (bool, int, uint).
+
+            # For integer types, use nullable integer dtype if available
+            if data.dtype.kind in ('i', 'u'):
+                # Map numpy int/uint dtypes to pandas nullable integer dtypes
+                dtype_map = {
+                    'int64': pd.Int64Dtype(), 'int32': pd.Int32Dtype(),
+                    'int16': pd.Int16Dtype(), 'int8': pd.Int8Dtype(),
+                    'uint64': pd.UInt64Dtype(), 'uint32': pd.UInt32Dtype(),
+                    'uint16': pd.UInt16Dtype(), 'uint8': pd.UInt8Dtype(),
+                }
+                data = pd.array(data, dtype=dtype_map.get(str(data.dtype), pd.Int64Dtype()))
+            else:
+                data = data.astype(float)
+            data[null_pos] = None
         except SettingWithCopyError:
             # `SettingWithCopyError` may happen for datetime-like with missing values.
             data = data.copy()
```

This fix preserves nullable integer dtypes when setting null values, maintaining type fidelity during interchange operations.