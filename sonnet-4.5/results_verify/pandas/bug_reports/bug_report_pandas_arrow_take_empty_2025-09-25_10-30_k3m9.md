# Bug Report: ArrowExtensionArray.take() Crashes with Empty Indices

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.take()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`ArrowExtensionArray.take([])` crashes with `ArrowNotImplementedError` when passed an empty list of indices, while other pandas array types handle this correctly.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st


@st.composite
def int_arrow_arrays(draw, min_size=0, max_size=30):
    data = draw(st.lists(
        st.one_of(st.integers(min_value=-1000, max_value=1000), st.none()),
        min_size=min_size,
        max_size=max_size
    ))
    return pd.array(data, dtype='int64[pyarrow]')


@given(arr=int_arrow_arrays(min_size=1, max_size=20))
def test_take_empty_indices(arr):
    """Taking with empty indices should return empty array."""
    result = arr.take([])
    assert len(result) == 0
    assert result.dtype == arr.dtype
```

**Failing input**: Any ArrowExtensionArray (e.g., `pd.array([1], dtype='int64[pyarrow]')`)

## Reproducing the Bug

```python
import pandas as pd

arr = pd.array([1, 2, 3], dtype='int64[pyarrow]')
result = arr.take([])
```

Output:
```
ArrowNotImplementedError: Function 'array_take' has no kernel matching input types (int64, double)
```

Expected: Should return an empty ArrowExtensionArray with the same dtype.

Comparison with regular pandas arrays (which work correctly):
```python
regular_arr = pd.array([1, 2, 3], dtype='int64')
result = regular_arr.take([])
```

## Why This Is A Bug

1. **API Inconsistency**: Other pandas array types (NumpyExtensionArray, etc.) handle `take([])` correctly
2. **Violates User Expectations**: Taking zero elements is a valid operation that should return an empty array
3. **Common Use Case**: Empty indices can occur in normal data processing workflows (e.g., filtering)

## Root Cause

In `array.py` line ~1354, the `take()` method converts indices using:
```python
indices_array = np.asanyarray(indices)
```

When `indices` is an empty list `[]`, NumPy defaults to creating a `float64` array. PyArrow then fails because it cannot match the `array_take` kernel for mismatched types (e.g., `int64` array with `float64` indices).

## Fix

```diff
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1351,7 +1351,10 @@ class ArrowExtensionArray(
         api.extensions.take
         """
-        indices_array = np.asanyarray(indices)
+        indices_array = np.asanyarray(indices, dtype=np.intp)
+        if indices_array.dtype.kind != 'i':
+            # Ensure integer dtype for indices
+            indices_array = indices_array.astype(np.intp, copy=False)

         if len(self._pa_array) == 0 and (indices_array >= 0).any():
             raise IndexError("cannot do a non-empty take")
```

Alternative simpler fix:
```diff
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1351,7 +1351,7 @@ class ArrowExtensionArray(
         api.extensions.take
         """
-        indices_array = np.asanyarray(indices)
+        indices_array = np.asanyarray(indices, dtype=np.intp)

         if len(self._pa_array) == 0 and (indices_array >= 0).any():
             raise IndexError("cannot do a non-empty take")
```