# Bug Report: pandas.core.arrays.arrow ArrowExtensionArray.fillna Crashes on Null-Typed Arrays

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.fillna`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

ArrowExtensionArray.fillna() crashes with an uninformative error when called on arrays with PyArrow's `null` type, which occurs when arrays are created from all-None values without specifying an explicit type.

## Property-Based Test

```python
import pandas as pd
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


if __name__ == "__main__":
    test_fillna_preserves_non_na()
```

<details>

<summary>
**Failing input**: `values=[None]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 18, in <module>
    test_fillna_preserves_non_na()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 8, in test_fillna_preserves_non_na
    def test_fillna_preserves_non_na(values):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 10, in test_fillna_preserves_non_na
    filled = arr.fillna(value=999)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 1160, in fillna
    fill_value = self._box_pa(value, pa_type=self._pa_array.type)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 407, in _box_pa
    return cls._box_pa_scalar(value, pa_type)
           ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 443, in _box_pa_scalar
    pa_scalar = pa.scalar(value, type=pa_type, from_pandas=True)
  File "pyarrow/scalar.pxi", line 1599, in pyarrow.lib.scalar
  File "pyarrow/error.pxi", line 155, in pyarrow.lib.pyarrow_internal_check_status
  File "pyarrow/error.pxi", line 92, in pyarrow.lib.check_status
pyarrow.lib.ArrowInvalid: Invalid null value
Falsifying example: test_fillna_preserves_non_na(
    values=[None],
)
```
</details>

## Reproducing the Bug

```python
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

# Create an array with only None values
# PyArrow will infer this as null type
arr = ArrowExtensionArray(pa.array([None]))

print(f"Array type: {arr._pa_array.type}")
print(f"Array values: {arr}")

# Try to fill NA values with 999
# This will crash because null type can't hold non-null values
try:
    filled = arr.fillna(value=999)
    print(f"Filled array: {filled}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
Error: ArrowInvalid: Invalid null value
</summary>
```
Array type: null
Array values: <ArrowExtensionArray>
[<NA>]
Length: 1, dtype: null[pyarrow]
Error: ArrowInvalid: Invalid null value
```
</details>

## Why This Is A Bug

This violates expected behavior because ArrowExtensionArray accepts null-typed arrays during construction but fails when common operations like fillna() are called. The error message "Invalid null value" provides no guidance about the actual issue - that PyArrow's null type can only hold null values and cannot be filled with non-null values. Users following documented construction patterns like `ArrowExtensionArray(pa.array([None]))` or `ArrowExtensionArray._from_sequence([None])` will encounter this crash without understanding why. The API is inconsistent: if null-typed arrays cannot support fillna operations, they should either be rejected at construction time or fillna should provide a clear error message explaining the limitation.

## Relevant Context

PyArrow's `null` type is a special data type that can only contain null values - it cannot hold any non-null data by design. When `pa.array([None])` is called without an explicit type parameter, PyArrow infers the type as `null`. The fillna method in ArrowExtensionArray attempts to create a PyArrow scalar with the fill value and the array's type (line 1160 in array.py), which calls `pa.scalar(999, type=pa.null(), from_pandas=True)` at line 443. This is fundamentally invalid in PyArrow since the null type cannot hold the value 999.

In contrast, when an explicit type is provided like `pa.array([None], type=pa.int64())`, the array has a concrete type that can hold both null and non-null values, so fillna works correctly.

Documentation references:
- ArrowExtensionArray: https://pandas.pydata.org/docs/reference/api/pandas.core.arrays.ArrowExtensionArray.html
- PyArrow null type: https://arrow.apache.org/docs/python/generated/pyarrow.null.html

## Proposed Fix

```diff
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1157,6 +1157,12 @@ class ArrowExtensionArray(
                 )

         try:
+            if pa.types.is_null(self._pa_array.type):
+                raise ValueError(
+                    "Cannot fillna on an array with null type. "
+                    "Arrays with null type can only contain null values. "
+                    "Please create the array with an explicit type, e.g., pa.array([None], type=pa.int64())"
+                )
             fill_value = self._box_pa(value, pa_type=self._pa_array.type)
         except pa.ArrowTypeError as err:
             msg = f"Invalid value '{value!s}' for dtype '{self.dtype}'"
```