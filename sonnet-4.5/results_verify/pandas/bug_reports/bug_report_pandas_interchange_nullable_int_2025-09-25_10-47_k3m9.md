# Bug Report: pandas.core.interchange Nullable Integer Dtype Loss

**Target**: `pandas.core.interchange.from_dataframe.set_nulls`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Nullable integer dtypes (`Int64`, `Int32`, etc.) are incorrectly converted to `float64` during round-trip operations through the pandas interchange protocol, violating the expectation that data interchange should preserve dtype information.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.api.interchange import from_dataframe

@given(
    st.lists(
        st.one_of(st.integers(min_value=0, max_value=100), st.none()),
        min_size=1,
        max_size=100
    )
)
@settings(max_examples=1000)
def test_roundtrip_nullable_integers(data):
    df = pd.DataFrame({"col": pd.array(data, dtype="Int64")})

    interchange_obj = df.__dataframe__()
    result_df = from_dataframe(interchange_obj)

    pd.testing.assert_frame_equal(result_df, df)
```

**Failing input**: `[None]` (or any list containing at least one `None` value mixed with integers)

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({"col": pd.array([1, None, 3], dtype="Int64")})
print("Original dtype:", df["col"].dtype)

interchange_obj = df.__dataframe__()
result_df = from_dataframe(interchange_obj)
print("Result dtype:", result_df["col"].dtype)

assert df["col"].dtype == result_df["col"].dtype
```

**Output:**
```
Original dtype: Int64
Result dtype: float64
AssertionError: Dtype changed from Int64 to float64
```

## Why This Is A Bug

The pandas interchange protocol is designed to enable zero-copy data exchange between dataframe libraries while preserving data integrity and metadata. The conversion of `Int64` to `float64` violates this contract because:

1. **Semantic difference**: `Int64` is a nullable integer type, while `float64` is a floating-point type. Code that relies on integer semantics will break.
2. **API contract violation**: The interchange protocol should preserve dtype information for proper round-trip conversion.
3. **Loss of type information**: Nullable integer types have specific properties (exact integer representation, explicit null support) that are lost when converted to float.

The root cause is in `/pandas/core/interchange/from_dataframe.py:546-551`:

```python
except TypeError:
    # TypeError happens if the `data` dtype appears to be non-nullable
    # in numpy notation (bool, int, uint). If this happens,
    # cast the `data` to nullable float dtype.
    data = data.astype(float)
    data[null_pos] = None
```

When the code tries to set `None` values in a numpy integer array (which `Int64` gets converted to in the interchange format), it catches the `TypeError` and blindly converts to float instead of restoring the original pandas nullable integer dtype.

## Fix

The fix requires tracking the original pandas dtype through the interchange protocol and restoring it during conversion back. Here's a conceptual patch:

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -543,8 +543,14 @@ def set_nulls(
             data = data.copy()
         try:
             data[null_pos] = None
         except TypeError:
-            # TypeError happens if the `data` dtype appears to be non-nullable
-            # in numpy notation (bool, int, uint). If this happens,
-            # cast the `data` to nullable float dtype.
-            data = data.astype(float)
-            data[null_pos] = None
+            # For integer dtypes, use pandas nullable integer array instead of float
+            if data.dtype.kind in ('i', 'u'):
+                # Determine appropriate nullable integer dtype
+                dtype_map = {8: 'Int8', 16: 'Int16', 32: 'Int32', 64: 'Int64'}
+                if data.dtype.kind == 'u':
+                    dtype_map = {8: 'UInt8', 16: 'UInt16', 32: 'UInt32', 64: 'UInt64'}
+                nullable_dtype = dtype_map.get(data.dtype.itemsize * 8)
+                data = pd.array(data, dtype=nullable_dtype)
+                data[null_pos] = None
+            else:
+                data = data.astype(float)
+                data[null_pos] = None
```

This fix preserves integer semantics by using pandas' nullable integer arrays instead of converting to float.