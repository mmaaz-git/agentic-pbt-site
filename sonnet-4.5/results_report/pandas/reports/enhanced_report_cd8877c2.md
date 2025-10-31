# Bug Report: pandas.core.arrays.arrow.ArrowExtensionArray.all() and .any() Fail on Null-Type Arrays

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.all` and `pandas.core.arrays.arrow.ArrowExtensionArray.any`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `all()` and `any()` methods on ArrowExtensionArray raise a TypeError when called on arrays with null dtype (containing only None values or empty arrays), violating their documented behavior of returning True and False respectively when `skipna=True`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

arrow_int_array = st.lists(
    st.one_of(st.integers(min_value=-1000, max_value=1000), st.none()),
    min_size=0,
    max_size=100
).map(lambda x: ArrowExtensionArray(pa.array(x)))

@given(arrow_int_array)
@settings(max_examples=200)
def test_all_true_implies_any_true(arr):
    assume(len(arr) > 0)

    if arr.all(skipna=True):
        assert arr.any(skipna=True), "If all() is True, any() should also be True"

if __name__ == "__main__":
    test_all_true_implies_any_true()
```

<details>

<summary>
**Failing input**: `arr=(lambda x: ArrowExtensionArray(pa.array(x)))([None])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 1844, in _reduce_pyarrow
    result = pyarrow_meth(data_to_reduce, skip_nulls=skipna, **kwargs)
  File "/home/npc/.local/lib/python3.13/site-packages/pyarrow/compute.py", line 269, in wrapper
    return func.call(args, options, memory_pool)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pyarrow/_compute.pyx", line 407, in pyarrow._compute.Function.call
  File "pyarrow/error.pxi", line 155, in pyarrow.lib.pyarrow_internal_check_status
  File "pyarrow/error.pxi", line 92, in pyarrow.lib.check_status
pyarrow.lib.ArrowNotImplementedError: Function 'all' has no kernel matching input types (null)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 20, in <module>
    test_all_true_implies_any_true()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 12, in test_all_true_implies_any_true
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 16, in test_all_true_implies_any_true
    if arr.all(skipna=True):
       ~~~~~~~^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 1033, in all
    return self._reduce("all", skipna=skipna, **kwargs)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 1911, in _reduce
    result = self._reduce_calc(name, skipna=skipna, keepdims=keepdims, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 1920, in _reduce_calc
    pa_result = self._reduce_pyarrow(name, skipna=skipna, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py", line 1852, in _reduce_pyarrow
    raise TypeError(msg) from err
TypeError: 'ArrowExtensionArray' with dtype null[pyarrow] does not support reduction 'all' with pyarrow version 20.0.0. 'all' may be supported by upgrading pyarrow.
Falsifying example: test_all_true_implies_any_true(
    arr=(lambda x: ArrowExtensionArray(pa.array(x)))([None]),
)
```
</details>

## Reproducing the Bug

```python
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

# Create an ArrowExtensionArray with only None values
arr = ArrowExtensionArray(pa.array([None]))

print("Created ArrowExtensionArray with pa.array([None])")
print(f"Array dtype: {arr.dtype}")
print(f"Array values: {arr}")
print()

# Try calling all() with skipna=True (should return True according to docstring)
print("Calling arr.all(skipna=True)...")
try:
    result_all = arr.all(skipna=True)
    print(f"Result: {result_all}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print()

# Try calling any() with skipna=True (should return False according to docstring)
print("Calling arr.any(skipna=True)...")
try:
    result_any = arr.any(skipna=True)
    print(f"Result: {result_any}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print()

# Also test with an empty array
arr_empty = ArrowExtensionArray(pa.array([]))
print("Created ArrowExtensionArray with pa.array([])")
print(f"Empty array dtype: {arr_empty.dtype}")
print(f"Empty array values: {arr_empty}")
print()

print("Calling arr_empty.all(skipna=True)...")
try:
    result_all = arr_empty.all(skipna=True)
    print(f"Result: {result_all}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print()

print("Calling arr_empty.any(skipna=True)...")
try:
    result_any = arr_empty.any(skipna=True)
    print(f"Result: {result_any}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")
```

<details>

<summary>
TypeError raised for all() and any() on null-type arrays
</summary>
```
Created ArrowExtensionArray with pa.array([None])
Array dtype: null[pyarrow]
Array values: <ArrowExtensionArray>
[<NA>]
Length: 1, dtype: null[pyarrow]

Calling arr.all(skipna=True)...
Exception raised: TypeError: 'ArrowExtensionArray' with dtype null[pyarrow] does not support reduction 'all' with pyarrow version 20.0.0. 'all' may be supported by upgrading pyarrow.

Calling arr.any(skipna=True)...
Exception raised: TypeError: 'ArrowExtensionArray' with dtype null[pyarrow] does not support reduction 'any' with pyarrow version 20.0.0. 'any' may be supported by upgrading pyarrow.

Created ArrowExtensionArray with pa.array([])
Empty array dtype: null[pyarrow]
Empty array values: <ArrowExtensionArray>
[]
Length: 0, dtype: null[pyarrow]

Calling arr_empty.all(skipna=True)...
Exception raised: TypeError: 'ArrowExtensionArray' with dtype null[pyarrow] does not support reduction 'all' with pyarrow version 20.0.0. 'all' may be supported by upgrading pyarrow.

Calling arr_empty.any(skipna=True)...
Exception raised: TypeError: 'ArrowExtensionArray' with dtype null[pyarrow] does not support reduction 'any' with pyarrow version 20.0.0. 'any' may be supported by upgrading pyarrow.
```
</details>

## Why This Is A Bug

This violates the explicit API contract documented in the method docstrings. The `all()` method documentation states: "Exclude NA values. If the entire array is NA and `skipna` is True, then the result will be True, as for an empty array." Similarly, the `any()` method documentation states: "If the entire array is NA and `skipna` is True, then the result will be False, as for an empty array."

The current implementation raises a TypeError instead of returning these documented values. The error occurs because PyArrow doesn't provide `all` or `any` kernels for null-type arrays, but pandas should handle this special case to maintain API consistency. The error message is also misleading, suggesting that upgrading PyArrow might help, when in fact null-type arrays fundamentally lack these operations in PyArrow.

This breaks the expected pandas reduction API contract and prevents users from using basic reduction operations on arrays that happen to contain only NA values - a situation that can naturally occur during data processing workflows such as filtering operations or initialization phases.

## Relevant Context

The bug occurs in the `_reduce_pyarrow` method at `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py:1843-1852`. When PyArrow raises an `ArrowNotImplementedError` for null-type arrays, the exception is caught and re-raised with a misleading message.

Other pandas nullable array types (IntegerArray, BooleanArray) correctly handle this case. The docstrings even include examples showing the expected behavior:
- `pd.array([pd.NA], dtype='boolean[pyarrow]').all()` should return True
- `pd.array([pd.NA], dtype='boolean[pyarrow]').any()` should return False

PyArrow documentation: https://arrow.apache.org/docs/python/compute.html
Source code location: pandas/core/arrays/arrow/array.py

## Proposed Fix

```diff
--- a/pandas/core/arrays/arrow/array.py
+++ b/pandas/core/arrays/arrow/array.py
@@ -1749,6 +1749,17 @@ class ArrowExtensionArray(
         """
         pa_type = self._pa_array.type

+        # Handle null-type arrays specially
+        if pa.types.is_null(pa_type):
+            if skipna and name in ["all", "any"]:
+                # When skipna=True and all values are NA, follow documented behavior
+                if name == "all":
+                    return pa.scalar(True, type=pa.bool_())
+                elif name == "any":
+                    return pa.scalar(False, type=pa.bool_())
+            # For other cases with null type, return NA
+            return pa.scalar(None, type=pa.null())
+
         data_to_reduce = self._pa_array

         cast_kwargs = {} if pa_version_under13p0 else {"safe": False}
```