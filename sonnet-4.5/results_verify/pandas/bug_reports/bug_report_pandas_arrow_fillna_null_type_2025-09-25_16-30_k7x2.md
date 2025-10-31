# Bug Report: ArrowExtensionArray fillna fails on null-typed arrays

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.fillna`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

ArrowExtensionArray.fillna() crashes with "Invalid null value" when called on an array with PyArrow null type (inferred from all-None data).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

@given(
    st.lists(st.one_of(st.integers(min_value=-100, max_value=100), st.none()), min_size=1),
    st.integers(min_value=-100, max_value=100)
)
def test_arrow_array_fillna_replaces_nulls(values, fill_value):
    arr = ArrowExtensionArray(pa.array(values))
    result = arr.fillna(fill_value)

    for i, val in enumerate(values):
        if val is None:
            assert result[i] == fill_value
        else:
            assert result[i] == val
```

**Failing input**: `values=[None]`, `fill_value=0`

## Reproducing the Bug

```python
import pandas as pd
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

arr = ArrowExtensionArray(pa.array([None]))
arr.fillna(0)
```

**Output:**
```
pyarrow.lib.ArrowInvalid: Invalid null value
```

## Why This Is A Bug

1. **Inconsistent with other pandas dtypes**: Regular pandas dtypes (Int64, object) handle fillna on all-NA arrays without error.

2. **Reasonable use case**: Users may have all-NA data that they want to fill with a default value.

3. **Cryptic error message**: "Invalid null value" doesn't help users understand the issue or how to fix it.

**Comparison with other dtypes:**

```python
# Int64 dtype - works fine
pd.Series([None], dtype="Int64").fillna(0)  # → Series([0], dtype: Int64)

# Arrow with explicit dtype - works fine
pd.array([None], dtype="int64[pyarrow]").fillna(0)  # → [0]

# Arrow with inferred null type - FAILS
ArrowExtensionArray(pa.array([None])).fillna(0)  # → ArrowInvalid!
```

## Fix

The issue occurs at line 1160 in `array.py` where `_box_pa` tries to create a scalar of type `null` with a non-null value.

**Solution**: When the array has type `null`, infer the appropriate type from the fill value before boxing:

```diff
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1157,7 +1157,13 @@ class ArrowExtensionArray(
                 )

         try:
-            fill_value = self._box_pa(value, pa_type=self._pa_array.type)
+            # Handle null type specially - infer type from fill value
+            if pa.types.is_null(self._pa_array.type):
+                fill_scalar = self._box_pa_scalar(value, pa_type=None)
+                fill_value = fill_scalar
+            else:
+                fill_value = self._box_pa(value, pa_type=self._pa_array.type)
+
         except pa.ArrowTypeError as err:
             msg = f"Invalid value '{value!s}' for dtype '{self.dtype}'"
             raise TypeError(msg) from err