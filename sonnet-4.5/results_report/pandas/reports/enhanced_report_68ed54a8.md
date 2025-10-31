# Bug Report: ArrowExtensionArray.fillna Crashes on Null Type Arrays

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.fillna`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `fillna()` method on `ArrowExtensionArray` crashes with an `ArrowInvalid` exception when called on arrays with PyArrow's `null` type, which PyArrow automatically infers when creating arrays from all-null data.

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

if __name__ == "__main__":
    test_arrow_array_fillna_removes_all_nulls()
```

<details>

<summary>
**Failing input**: `data=[None], fill_value=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 22, in <module>
    test_arrow_array_fillna_removes_all_nulls()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 10, in test_arrow_array_fillna_removes_all_nulls
    st.lists(st.integers(min_value=-1000, max_value=1000) | st.none(), min_size=1, max_size=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 15, in test_arrow_array_fillna_removes_all_nulls
    filled = arr.fillna(fill_value)
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
Falsifying example: test_arrow_array_fillna_removes_all_nulls(
    data=[None],
    fill_value=0,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py:1161
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

# Create an array with null type (happens when all values are None)
arr = ArrowExtensionArray(pa.array([None]))
print(f"Array type: {arr._pa_array.type}")
print(f"Array contents: {arr._pa_array}")

# Try to fill with a value
result = arr.fillna(0)
print(f"Result: {result}")
```

<details>

<summary>
ArrowInvalid: Invalid null value
</summary>
```
Array type: null
Array contents: [
1 nulls
]
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/repo.py", line 13, in <module>
    result = arr.fillna(0)
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
```
</details>

## Why This Is A Bug

The `fillna()` method is designed to replace null values with a specified fill value. When users have data consisting entirely of null values, PyArrow automatically infers the data type as `null`. This is a common scenario in real-world applications where database columns may be empty or unpopulated fields exist in datasets.

The current implementation attempts to box the fill value into a PyArrow scalar with type `null` (line 1160: `self._box_pa(value, pa_type=self._pa_array.type)`), which fails because PyArrow cannot create a non-null scalar value with null type. This violates the expected contract of `fillna()` - it should be able to fill null values regardless of how the array was constructed.

The documentation for `fillna()` states it should "Fill NA/null values using the specified method" but doesn't mention any limitation for null-type arrays. Users reasonably expect to be able to fill null values with concrete values, especially since this operation would work with regular pandas arrays or numpy arrays containing only NaNs.

## Relevant Context

PyArrow's type inference system automatically assigns the `null` type when all elements in an array are null/None. This is documented behavior in PyArrow: https://arrow.apache.org/docs/python/generated/pyarrow.array.html

The pandas codebase already has special handling for null types in other methods. For example, the `to_numpy()` method in the same file (around line 1455-1458) includes special handling for null type arrays.

The error occurs specifically at line 443 in `_box_pa_scalar` when calling `pa.scalar(value, type=pa_type, from_pandas=True)` with `pa_type=pa.null()`. PyArrow raises `ArrowInvalid` because you cannot create a non-null scalar with null type - the null type by definition can only contain null values.

## Proposed Fix

```diff
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1157,6 +1157,12 @@ class ArrowExtensionArray(
                 )

         try:
+            # Handle null type arrays by inferring type from fill value
+            if pa.types.is_null(self._pa_array.type):
+                inferred_type = pa.infer_type([value])
+                casted_array = self._pa_array.cast(inferred_type)
+                return type(self)(pc.fill_null(casted_array, value))
+
             fill_value = self._box_pa(value, pa_type=self._pa_array.type)
         except pa.ArrowTypeError as err:
             msg = f"Invalid value '{value!s}' for dtype '{self.dtype}'"
```