# Bug Report: ArrowExtensionArray.take() Empty List TypeError

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.take()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Calling `ArrowExtensionArray.take([])` with an empty Python list crashes with `ArrowNotImplementedError` because numpy converts the empty list to a float64 array instead of an integer array, which PyArrow's `array_take` function doesn't accept.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.arrays.arrow import ArrowExtensionArray
import pyarrow as pa

@st.composite
def arrow_arrays(draw):
    dtype_choice = draw(st.sampled_from([pa.int64(), pa.float64(), pa.string(), pa.bool_()]))
    size = draw(st.integers(min_value=0, max_value=100))

    if dtype_choice == pa.int64():
        values = draw(st.lists(
            st.one_of(st.integers(min_value=-10000, max_value=10000), st.none()),
            min_size=size, max_size=size
        ))
    elif dtype_choice == pa.float64():
        values = draw(st.lists(
            st.one_of(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), st.none()),
            min_size=size, max_size=size
        ))
    elif dtype_choice == pa.string():
        values = draw(st.lists(st.one_of(st.text(max_size=20), st.none()), min_size=size, max_size=size))
    else:
        values = draw(st.lists(st.one_of(st.booleans(), st.none()), min_size=size, max_size=size))

    pa_array = pa.array(values, type=dtype_choice)
    return ArrowExtensionArray(pa_array)

@given(arrow_arrays())
def test_take_with_empty_indices(arr):
    result = arr.take([])
    assert len(result) == 0
```

**Failing input**: `arr=<ArrowExtensionArray>[]; Length: 0, dtype: int64[pyarrow]`

## Reproducing the Bug

```python
import pandas as pd
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

arr = ArrowExtensionArray(pa.array([1, 2, 3], type=pa.int64()))
result = arr.take([])
```

**Output**:
```
pyarrow.lib.ArrowNotImplementedError: Function 'array_take' has no kernel matching input types (int64, double)
```

The issue is that `np.asanyarray([])` creates a float64 array by default:
```python
import numpy as np
np.asanyarray([]).dtype  # dtype('float64')
```

When this float64 array is passed to PyArrow's `array_take` function, it fails because PyArrow expects integer indices.

## Why This Is A Bug

The `take()` method should accept any sequence-like object as indices, including empty lists. The docstring states "indices : sequence of int or one-dimensional np.ndarray of int" but doesn't mention that empty lists will fail. This violates the reasonable user expectation that `take([])` should return an empty array, which is a common edge case in data processing.

## Fix

```diff
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1350,7 +1350,10 @@ class ArrowExtensionArray(
         it's called by :meth:`Series.reindex`, or any other method
         that causes realignment, with a `fill_value`.
         """
-        indices_array = np.asanyarray(indices)
+        indices_array = np.asanyarray(indices, dtype=np.intp)
+        if indices_array.size == 0:
+            # Ensure empty arrays have integer dtype
+            indices_array = np.array([], dtype=np.intp)

         if len(self._pa_array) == 0 and (indices_array >= 0).any():
             raise IndexError("cannot do a non-empty take")
```

Or more simply:

```diff
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1350,7 +1350,7 @@ class ArrowExtensionArray(
         it's called by :meth:`Series.reindex`, or any other method
         that causes realignment, with a `fill_value`.
         """
-        indices_array = np.asanyarray(indices)
+        indices_array = np.asanyarray(indices, dtype=np.intp if len(indices) == 0 else None) if len(indices) == 0 else np.asanyarray(indices).astype(np.intp, copy=False)
```

Actually, the cleanest fix is:

```diff
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1350,7 +1350,11 @@ class ArrowExtensionArray(
         it's called by :meth:`Series.reindex`, or any other method
         that causes realignment, with a `fill_value`.
         """
-        indices_array = np.asanyarray(indices)
+        indices_array = np.asanyarray(indices)
+        if indices_array.size == 0:
+            # np.asanyarray([]) defaults to float64, but we need integer indices
+            indices_array = np.array([], dtype=np.intp)
+        elif not np.issubdtype(indices_array.dtype, np.integer):
+            indices_array = indices_array.astype(np.intp)

         if len(self._pa_array) == 0 and (indices_array >= 0).any():
             raise IndexError("cannot do a non-empty take")
```