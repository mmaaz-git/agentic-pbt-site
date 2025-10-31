# Bug Report: pandas.core.interchange Object Dtype Integers Not Supported

**Target**: `pandas.core.interchange`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The DataFrame interchange protocol crashes when attempting to convert DataFrames containing integers outside the int64 range (-2^63 to 2^63-1). These values are stored with object dtype by pandas, which causes `__dataframe__()` to succeed but `from_dataframe()` to fail with NotImplementedError.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.api.interchange import from_dataframe

@given(
    data=st.lists(st.integers(), min_size=0, max_size=100),
    col_name=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
)
@settings(max_examples=1000)
def test_round_trip_integer_column(data, col_name):
    df = pd.DataFrame({col_name: data})

    interchange_df = df.__dataframe__()
    df_roundtrip = from_dataframe(interchange_df)

    assert df.equals(df_roundtrip), f"Round-trip failed: {df} != {df_roundtrip}"
```

**Failing input**: `data=[-9_223_372_036_854_775_809], col_name='A'`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({'a': [-9_223_372_036_854_775_809]})

print(f"DataFrame dtype: {df['a'].dtype}")
print(f"Value is below int64 min: {-9_223_372_036_854_775_809 < np.iinfo(np.int64).min}")

interchange_df = df.__dataframe__()
print("Created interchange object successfully")

df_roundtrip = from_dataframe(interchange_df)
```

## Why This Is A Bug

The interchange protocol's `__dataframe__()` method succeeds without validation, creating a PandasDataFrameXchg object. However, when attempting to use this object (via `from_dataframe()` or accessing column dtypes), it fails with "NotImplementedError: Non-string object dtypes are not supported yet".

This violates the principle of fail-fast: if a DataFrame contains unsupported data types, the error should occur at the point of calling `__dataframe__()`, not later when attempting to use the interchange object. The current behavior creates a partially-constructed object that cannot be used, leading to confusing error messages.

Python integers can be arbitrarily large, and pandas correctly handles them using object dtype. The interchange protocol should either:
1. Validate and reject such DataFrames early in `__dataframe__()`
2. Support object dtype integers by converting them appropriately

## Fix

The issue is in `pandas/core/interchange/column.py` in the `dtype` property. When `is_string_dtype(dtype)` is True for object dtype, it checks `infer_dtype(self._col)` and only accepts "string" or "empty", rejecting other types like "integer".

The fix should add early validation in `DataFrame.__dataframe__()`:

```diff
diff --git a/pandas/core/frame.py b/pandas/core/frame.py
index 1234567..abcdefg 100644
--- a/pandas/core/frame.py
+++ b/pandas/core/frame.py
@@ -1234,6 +1234,15 @@ class DataFrame(NDFrame, OpsMixin):
         """

         from pandas.core.interchange.dataframe import PandasDataFrameXchg
+        from pandas.api.types import infer_dtype
+
+        for col in self.columns:
+            dtype = self[col].dtype
+            if dtype == object:
+                inferred = infer_dtype(self[col])
+                if inferred not in ("string", "empty"):
+                    raise ValueError(
+                        f"Column '{col}' has unsupported object dtype (inferred: {inferred}). "
+                        "The interchange protocol only supports object columns containing strings."
+                    )

         return PandasDataFrameXchg(self, allow_copy=allow_copy)
```