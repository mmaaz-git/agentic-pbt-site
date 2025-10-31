# Bug Report: pandas.core.arrays.arrow.ArrowExtensionArray.take() Empty List TypeError

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.take()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ArrowExtensionArray.take()` method crashes with `ArrowNotImplementedError` when passed an empty Python list as indices, because numpy converts the empty list to a float64 array instead of an integer array, which PyArrow cannot process.

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

if __name__ == "__main__":
    test_take_with_empty_indices()
```

<details>

<summary>
**Failing input**: `arr=<ArrowExtensionArray>[]  Length: 0, dtype: int64[pyarrow]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 34, in <module>
    test_take_with_empty_indices()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 29, in test_take_with_empty_indices
    def test_take_with_empty_indices(arr):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 30, in test_take_with_empty_indices
    result = arr.take([])
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 1384, in take
    return type(self)(self._pa_array.take(indices_array))
                      ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "pyarrow/table.pxi", line 1079, in pyarrow.lib.ChunkedArray.take
  File "/home/npc/.local/lib/python3.13/site-packages/pyarrow/compute.py", line 504, in take
    return call_function('take', [data, indices], options, memory_pool)
  File "pyarrow/_compute.pyx", line 612, in pyarrow._compute.call_function
  File "pyarrow/_compute.pyx", line 407, in pyarrow._compute.Function.call
  File "pyarrow/error.pxi", line 155, in pyarrow.lib.pyarrow_internal_check_status
  File "pyarrow/error.pxi", line 92, in pyarrow.lib.check_status
pyarrow.lib.ArrowNotImplementedError: Function 'array_take' has no kernel matching input types (int64, double)
Falsifying example: test_take_with_empty_indices(
    arr=<ArrowExtensionArray>
    []
    Length: 0, dtype: int64[pyarrow],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

# Create an ArrowExtensionArray with some sample data
arr = ArrowExtensionArray(pa.array([1, 2, 3], type=pa.int64()))

# Try to take an empty list of indices
try:
    result = arr.take([])
    print("Success: Got result with length", len(result))
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
ArrowNotImplementedError: Function 'array_take' has no kernel matching input types (int64, double)
</summary>
```
Error: ArrowNotImplementedError: Function 'array_take' has no kernel matching input types (int64, double)
```
</details>

## Why This Is A Bug

This violates expected behavior and contradicts the documented interface in several ways:

1. **Documentation Contract Violation**: The `take()` method documentation states it accepts "sequence of int or one-dimensional np.ndarray of int" as indices. An empty list `[]` is a valid sequence in Python and should be accepted according to this specification.

2. **Inconsistent Type Conversion**: The bug occurs because `np.asanyarray([])` creates a float64 array by default rather than an integer array. This is a known NumPy behavior where empty arrays default to float64, but the ArrowExtensionArray implementation doesn't account for this edge case.

3. **Cryptic Error Message**: Users receive an unintuitive error about type mismatch ("int64, double") rather than proper handling of the empty indices case. This makes debugging difficult.

4. **Common Use Case**: Empty selections are routine in data processing pipelines (e.g., filtering that results in no matches). Other similar methods in pandas handle empty indices correctly.

5. **Inconsistent with Related APIs**: When properly typed integer arrays are used, both NumPy's `take()` and PyArrow's native `take()` handle empty indices correctly, returning empty arrays.

## Relevant Context

The issue is in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py` at line 1353 where `indices_array = np.asanyarray(indices)` converts the empty list to float64.

This can be verified with:
- `np.asanyarray([]).dtype` returns `dtype('float64')`
- `np.array([], dtype=np.intp)` correctly creates an empty integer array

PyArrow's `array_take` function requires integer indices and cannot process float64 indices, hence the error. The fix needs to ensure empty arrays are created with integer dtype before being passed to PyArrow.

Documentation reference: The take() method is extensively used by pandas operations like `Series.__getitem__`, `.loc`, `.iloc`, and `Series.reindex`, making this bug potentially impactful for data processing workflows.

## Proposed Fix

```diff
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1350,7 +1350,11 @@ class ArrowExtensionArray(
         it's called by :meth:`Series.reindex`, or any other method
         that causes realignment, with a `fill_value`.
         """
-        indices_array = np.asanyarray(indices)
+        indices_array = np.asanyarray(indices)
+        # Handle empty indices - np.asanyarray([]) defaults to float64
+        if indices_array.size == 0 and indices_array.dtype == np.float64:
+            indices_array = np.array([], dtype=np.intp)

         if len(self._pa_array) == 0 and (indices_array >= 0).any():
             raise IndexError("cannot do a non-empty take")
```