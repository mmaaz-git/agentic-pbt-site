# Bug Report: ArrowExtensionArray.take() Crashes with Empty Indices Due to NumPy Float64 Default

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.take()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

ArrowExtensionArray.take() crashes with ArrowNotImplementedError when passed an empty list of indices because np.asanyarray([]) defaults to float64 dtype, which PyArrow's take function cannot handle with integer arrays.

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


if __name__ == "__main__":
    # Run the test
    test_take_empty_indices()
```

<details>

<summary>
**Failing input**: `pd.array([<NA>], dtype='int64[pyarrow]')` (or any ArrowExtensionArray)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 25, in <module>
    test_take_empty_indices()
    ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 16, in test_take_empty_indices
    def test_take_empty_indices(arr):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 18, in test_take_empty_indices
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
Falsifying example: test_take_empty_indices(
    arr=<ArrowExtensionArray>
    [<NA>]
    Length: 1, dtype: int64[pyarrow],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

print("Demonstrating ArrowExtensionArray.take([]) bug\n")
print("=" * 60)

# Create an ArrowExtensionArray with integer data
print("Creating ArrowExtensionArray with int64[pyarrow] dtype:")
arr = pd.array([1, 2, 3], dtype='int64[pyarrow]')
print(f"arr = pd.array([1, 2, 3], dtype='int64[pyarrow]')")
print(f"arr = {arr}")
print(f"arr.dtype = {arr.dtype}")
print()

# Try to take with empty indices - this should crash
print("Attempting to take with empty indices:")
print("result = arr.take([])")
print()

try:
    result = arr.take([])
    print(f"Result: {result}")
    print(f"Result length: {len(result)}")
    print(f"Result dtype: {result.dtype}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    print()

print("=" * 60)
print("\nComparison with regular NumPy-backed array (which works correctly):\n")

# Show that regular arrays work fine
print("Creating regular int64 array:")
regular_arr = pd.array([1, 2, 3], dtype='int64')
print(f"regular_arr = pd.array([1, 2, 3], dtype='int64')")
print(f"regular_arr = {regular_arr}")
print(f"regular_arr.dtype = {regular_arr.dtype}")
print()

print("Attempting to take with empty indices:")
print("result = regular_arr.take([])")
try:
    result = regular_arr.take([])
    print(f"Result: {result}")
    print(f"Result length: {len(result)}")
    print(f"Result dtype: {result.dtype}")
    print("\nSUCCESS: Regular array handles empty indices correctly")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print()
print("=" * 60)
print("\nDiagnosis of the issue:")
print("When indices=[], np.asanyarray([]) creates a float64 array by default:")
indices = []
indices_array = np.asanyarray(indices)
print(f"indices = {indices}")
print(f"np.asanyarray(indices) = {indices_array}")
print(f"np.asanyarray(indices).dtype = {indices_array.dtype}")
print()
print("PyArrow cannot handle float64 indices with integer arrays,")
print("causing the 'no kernel matching input types' error.")
```

<details>

<summary>
ArrowNotImplementedError when taking empty indices from ArrowExtensionArray
</summary>
```
Demonstrating ArrowExtensionArray.take([]) bug

============================================================
Creating ArrowExtensionArray with int64[pyarrow] dtype:
arr = pd.array([1, 2, 3], dtype='int64[pyarrow]')
arr = <ArrowExtensionArray>
[1, 2, 3]
Length: 3, dtype: int64[pyarrow]
arr.dtype = int64[pyarrow]

Attempting to take with empty indices:
result = arr.take([])

ERROR: ArrowNotImplementedError: Function 'array_take' has no kernel matching input types (int64, double)

============================================================

Comparison with regular NumPy-backed array (which works correctly):

Creating regular int64 array:
regular_arr = pd.array([1, 2, 3], dtype='int64')
regular_arr = <NumpyExtensionArray>
[1, 2, 3]
Length: 3, dtype: int64
regular_arr.dtype = int64

Attempting to take with empty indices:
result = regular_arr.take([])
Result: <NumpyExtensionArray>
[]
Length: 0, dtype: int64
Result length: 0
Result dtype: int64

SUCCESS: Regular array handles empty indices correctly

============================================================

Diagnosis of the issue:
When indices=[], np.asanyarray([]) creates a float64 array by default:
indices = []
np.asanyarray(indices) = []
np.asanyarray(indices).dtype = float64

PyArrow cannot handle float64 indices with integer arrays,
causing the 'no kernel matching input types' error.
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **API Contract Violation**: The ExtensionArray.take() documentation states it accepts "sequence of int" without minimum length restrictions. An empty list is a valid sequence and should be handled correctly.

2. **Inconsistency Across Pandas**: All other ExtensionArray implementations (NumpyExtensionArray, StringArray, PeriodArray, IntervalArray, BooleanArray, Categorical) correctly handle empty indices by returning an empty array of the same dtype. ArrowExtensionArray is the sole outlier.

3. **PyArrow Capability Mismatch**: PyArrow itself supports empty integer indices perfectly well (`pa.array([1,2,3]).take(pa.array([], type=pa.int64()))` works). The crash occurs only because pandas incorrectly converts empty lists to float64 arrays instead of integer arrays.

4. **Real-World Impact**: Empty indices naturally occur in data processing pipelines when filtering or selecting data results in no matches. Users expect `arr.take([])` to return an empty array, not crash.

## Relevant Context

The bug occurs at line 1353 in `/pandas/core/arrays/arrow/array.py`:
```python
indices_array = np.asanyarray(indices)
```

When `indices` is an empty list `[]`, NumPy's `asanyarray()` defaults to creating a float64 array. This happens because NumPy cannot infer the dtype from an empty list and uses its default float64.

PyArrow's take function then receives mismatched types:
- Data array: int64
- Indices array: float64 (should be integer)

This causes the kernel matching error since PyArrow requires integer indices for the take operation.

Key observations:
- This affects all Arrow-backed arrays (int64[pyarrow], string[pyarrow], float64[pyarrow], etc.)
- The fix is straightforward: explicitly specify integer dtype when converting indices
- No data corruption risk - the operation fails cleanly with an error

## Proposed Fix

```diff
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1350,7 +1350,7 @@ class ArrowExtensionArray(
         it's called by :meth:`Series.reindex`, or any other method
         that causes realignment, with a `fill_value`.
         """
-        indices_array = np.asanyarray(indices)
+        indices_array = np.asanyarray(indices, dtype=np.intp)

         if len(self._pa_array) == 0 and (indices_array >= 0).any():
             raise IndexError("cannot do a non-empty take")
```