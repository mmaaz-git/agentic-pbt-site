# Bug Report: pandas.core.arrays.arrow.ArrowExtensionArray.take Crashes with Empty Index List

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.take`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ArrowExtensionArray.take()` method crashes with an `ArrowNotImplementedError` when provided an empty list of indices, instead of returning an empty array as expected.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import pandas as pd
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

@settings(max_examples=500)
@given(
    st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    st.lists(st.integers(min_value=0, max_value=99), min_size=0, max_size=20)
)
def test_take_multiple_indices(data, indices):
    assume(all(idx < len(data) for idx in indices))
    arr = ArrowExtensionArray._from_sequence(data, dtype=pd.ArrowDtype(pa.int64()))
    result = arr.take(indices)
    expected = [data[idx] for idx in indices]
    assert result.tolist() == expected

if __name__ == "__main__":
    # Run the test
    test_take_multiple_indices()
```

<details>

<summary>
**Failing input**: `data=[0], indices=[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 20, in <module>
    test_take_multiple_indices()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 7, in test_take_multiple_indices
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 14, in test_take_multiple_indices
    result = arr.take(indices)
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
Falsifying example: test_take_multiple_indices(
    data=[0],
    indices=[],
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

# Create an ArrowExtensionArray with some data
arr = ArrowExtensionArray._from_sequence([1, 2, 3], dtype=pd.ArrowDtype(pa.int64()))

# Try to take with an empty list of indices - this should return an empty array
print("Attempting to take with empty indices list...")
result = arr.take([])
print(f"Result: {result}")
print(f"Result type: {type(result)}")
print(f"Result length: {len(result)}")
```

<details>

<summary>
ArrowNotImplementedError: Function 'array_take' has no kernel matching input types (int64, double)
</summary>
```
Attempting to take with empty indices list...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/repo.py", line 10, in <module>
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
```
</details>

## Why This Is A Bug

The `ArrowExtensionArray.take()` method should handle empty index sequences correctly, returning an empty array of the same dtype. This violates expected behavior in several ways:

1. **Documentation contradiction**: The method's docstring references `numpy.take` (line 1343 in array.py), which successfully handles empty indices by returning an empty array.

2. **Invalid type mismatch**: The crash occurs because `np.asanyarray([])` creates a float64 array by default (NumPy's behavior for empty arrays without explicit dtype), not an integer array. This causes PyArrow's `take()` function to fail with a type mismatch between the data array (int64) and indices array (float64).

3. **Inconsistent with NumPy**: NumPy's `take()` handles empty indices gracefully: `np.take([1,2,3], [])` returns an empty array of the same dtype.

4. **Confusing error message**: The error "Function 'array_take' has no kernel matching input types (int64, double)" exposes internal PyArrow implementation details rather than providing a clear indication of what went wrong.

5. **Valid input rejected**: An empty list is a valid sequence type according to the method signature which accepts "sequence of int". The method should not crash on valid input.

## Relevant Context

The bug occurs at line 1353 of `/pandas/core/arrays/arrow/array.py`:
```python
indices_array = np.asanyarray(indices)
```

When `indices` is an empty list, NumPy's `asanyarray` function creates an array with the default dtype of `float64` for empty arrays. This can be verified:
- `np.asanyarray([]).dtype` returns `dtype('float64')`
- `np.asanyarray([], dtype=np.intp).dtype` returns `dtype('int64')`

PyArrow's `take()` function requires matching types between data and indices arrays, hence the error when it receives a float64 indices array for int64 data.

The `take()` method is fundamental to pandas indexing operations and is called by various high-level methods including `Series.__getitem__`, `.loc`, `.iloc`, and `Series.reindex` as noted in the method's documentation.

## Proposed Fix

```diff
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1350,7 +1350,7 @@ class ArrowExtensionArray(
         that causes realignment, with a `fill_value`.
         """

-        indices_array = np.asanyarray(indices)
+        indices_array = np.asanyarray(indices, dtype=np.intp)

         if len(self._pa_array) == 0 and (indices_array >= 0).any():
             raise IndexError("cannot do a non-empty take")
```