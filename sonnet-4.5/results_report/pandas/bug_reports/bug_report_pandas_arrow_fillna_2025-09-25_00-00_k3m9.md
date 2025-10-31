# Bug Report: ArrowExtensionArray.fillna Fails on Null Type Arrays

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.fillna`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`ArrowExtensionArray.fillna()` crashes with `ArrowInvalid` when called on arrays with PyArrow `null` type, which occurs when creating arrays from all-null data without an explicit type.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
import pyarrow as pa
from hypothesis import given, strategies as st
from pandas.core.arrays.arrow import ArrowExtensionArray

@given(
    st.lists(st.integers(min_value=-1000, max_value=1000) | st.none(), min_size=1, max_size=50),
    st.integers(min_value=-1000, max_value=1000)
)
def test_arrow_array_fillna_removes_all_nulls(data, fill_value):
    arr = ArrowExtensionArray(pa.array(data))
    filled = arr.fillna(fill_value)

    filled_list = filled.tolist()
    for val in filled_list:
        assert val is not None and not pd.isna(val)
```

**Failing input**: `data=[None], fill_value=0`

## Reproducing the Bug

```python
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

arr = ArrowExtensionArray(pa.array([None]))
print(f"Array type: {arr._pa_array.type}")

result = arr.fillna(0)
```

**Output:**
```
Array type: null
Traceback (most recent call last):
  ...
  File "pandas/core/arrays/arrow/array.py", line 1160, in fillna
    fill_value = self._box_pa(value, pa_type=self._pa_array.type)
  File "pandas/core/arrays/arrow/array.py", line 407, in _box_pa
    return cls._box_pa_scalar(value, pa_type)
  File "pandas/core/arrays/arrow/array.py", line 443, in _box_pa_scalar
    pa_scalar = pa.scalar(value, type=pa_type, from_pandas=True)
pyarrow.lib.ArrowInvalid: Invalid null value
```

## Why This Is A Bug

1. **Realistic Use Case**: Users may have columns with all-null data (e.g., unpopulated fields) and want to fill them with default values.

2. **Automatic Type Inference**: PyArrow automatically infers `null` type for all-null data, so users encounter this without explicitly requesting null types.

3. **Inconsistent Behavior**: Other methods like `to_numpy()` have special handling for null types (see line 1455-1458 in array.py), but `fillna()` does not.

4. **Poor Error Message**: The error "Invalid null value" doesn't help users understand the problem or how to fix it.

5. **Expected Behavior**: Filling null values with a concrete value should infer the appropriate type and perform the operation, similar to how pandas handles this in other contexts.

## Fix

The `fillna` method should detect null type arrays and handle them specially by inferring the type from the fill value:

```diff
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1157,6 +1157,17 @@ class ArrowExtensionArray(
                 )

         try:
+            # Handle null type arrays: infer type from fill_value
+            if pa.types.is_null(self._pa_array.type):
+                if not is_scalar(value):
+                    # Let the super() method handle array-like fill values
+                    return super().fillna(value=value, method=method, limit=limit, copy=copy)
+                # Infer type from the fill value
+                inferred_type = pa.infer_type([value])
+                # Cast the null array to the inferred type
+                casted_array = self._pa_array.cast(inferred_type)
+                return type(self)(pc.fill_null(casted_array, fill_value=value))
+
             fill_value = self._box_pa(value, pa_type=self._pa_array.type)
         except pa.ArrowTypeError as err:
             msg = f"Invalid value '{value!s}' for dtype '{self.dtype}'"
```

Alternative simpler fix that catches the error and provides a better message:

```diff
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1157,8 +1157,15 @@ class ArrowExtensionArray(
                 )

         try:
             fill_value = self._box_pa(value, pa_type=self._pa_array.type)
-        except pa.ArrowTypeError as err:
+        except (pa.ArrowTypeError, pa.ArrowInvalid) as err:
+            if pa.types.is_null(self._pa_array.type):
+                raise TypeError(
+                    f"Cannot fill array with null type. Please specify a dtype "
+                    f"when creating the array, e.g., pa.array(data, type=pa.int64())"
+                ) from err
             msg = f"Invalid value '{value!s}' for dtype '{self.dtype}'"
             raise TypeError(msg) from err
```