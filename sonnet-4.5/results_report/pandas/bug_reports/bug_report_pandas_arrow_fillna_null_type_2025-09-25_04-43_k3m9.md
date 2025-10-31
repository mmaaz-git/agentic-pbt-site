# Bug Report: pandas.core.arrays.arrow ArrowExtensionArray.fillna Crashes on Null-Typed Arrays

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.fillna`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

ArrowExtensionArray.fillna() crashes with an unhelpful error message when called on arrays with PyArrow's `null` type, which occurs when arrays are created from all-None values without an explicit type.

## Property-Based Test

```python
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray
from hypothesis import given, strategies as st


@given(st.lists(st.one_of(st.integers(), st.none()), min_size=1))
def test_fillna_preserves_non_na(values):
    arr = ArrowExtensionArray(pa.array(values))
    filled = arr.fillna(value=999)

    for i in range(len(arr)):
        if not pd.isna(arr[i]):
            assert arr[i] == filled[i]
```

**Failing input**: `values=[None]`

## Reproducing the Bug

```python
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

arr = ArrowExtensionArray(pa.array([None]))
arr.fillna(value=999)
```

**Error**:
```
pyarrow.lib.ArrowInvalid: Invalid null value
```

**Why this happens**:
When `pa.array([None])` is created without an explicit type, PyArrow infers the type as `null` - a type that can only hold null values. The fillna method attempts to create a PyArrow scalar with this null type and a non-null value (999), which is invalid.

**Contrast with working case**:
```python
arr_with_type = ArrowExtensionArray(pa.array([None], type=pa.int64()))
filled = arr_with_type.fillna(value=999)
```

This works because the array has a concrete type (int64) that can hold the fill value.

## Why This Is A Bug

1. **Inconsistent behavior**: ArrowExtensionArray accepts null-typed arrays during construction without error, but fails when fillna() is called.

2. **Poor user experience**: The error message "Invalid null value" doesn't explain the actual issue (that the array has a null type and can't be filled).

3. **Documented construction method**: Both `ArrowExtensionArray(pa.array([...]))` and `ArrowExtensionArray._from_sequence([...])` are documented ways to create arrays, and both can create null-typed arrays.

4. **Asymmetric API**: pandas Series handles this gracefully by requiring explicit dtypes, but the lower-level ArrowExtensionArray doesn't.

## Fix

The fillna method should detect null-typed arrays and either:

**Option 1**: Raise an informative error during fillna:
```diff
diff --git a/pandas/core/arrays/arrow/array.py b/pandas/core/arrays/arrow/array.py
index 1234567..abcdefg 100644
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1155,6 +1155,11 @@ class ArrowExtensionArray(
         )
         mask = self.isna()

+        if pa.types.is_null(self._pa_array.type):
+            raise ValueError(
+                "Cannot fillna on an array with null type. "
+                "Please create the array with an explicit type."
+            )
+
         if method is not None:
             raise NotImplementedError(
                 f"ArrowExtensionArray does not support fillna with method {method}"
```

**Option 2**: Infer type from the fill value (more complex, may have unexpected behavior)

Option 1 is preferable as it provides clear guidance to users while avoiding potentially surprising type conversions.