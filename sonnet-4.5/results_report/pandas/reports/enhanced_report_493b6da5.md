# Bug Report: pandas.core.arrays.sparse.SparseArray.argmin/argmax Crashes on Arrays Where All Values Equal Fill Value

**Target**: `pandas.core.arrays.sparse.SparseArray.argmin` and `pandas.core.arrays.sparse.SparseArray.argmax`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

SparseArray.argmin() and SparseArray.argmax() crash with ValueError when called on arrays where all values equal the fill_value, because the sparse representation optimizes storage by not storing any sparse values, resulting in an empty sp_values array that causes numpy's argmin/argmax functions to fail.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st
from pandas.arrays import SparseArray


@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50))
def test_argmin_argmax_match_dense(data):
    arr = SparseArray(data, fill_value=0)
    dense = arr.to_dense()

    assert arr.argmin() == dense.argmin()
    assert arr.argmax() == dense.argmax()


if __name__ == "__main__":
    test_argmin_argmax_match_dense()
```

<details>

<summary>
**Failing input**: `data=[0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 19, in <module>
    test_argmin_argmax_match_dense()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 10, in test_argmin_argmax_match_dense
    def test_argmin_argmax_match_dense(data):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 14, in test_argmin_argmax_match_dense
    assert arr.argmin() == dense.argmin()
           ~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1684, in argmin
    return self._argmin_argmax("argmin")
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1658, in _argmin_argmax
    _candidate = non_nan_idx[func(non_nans)]
                             ~~~~^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 1439, in argmin
    return _wrapfunc(a, 'argmin', axis=axis, out=out, **kwds)
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
ValueError: attempt to get argmin of an empty sequence
Falsifying example: test_argmin_argmax_match_dense(
    data=[0],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:58
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.arrays import SparseArray

# Create a SparseArray where all values equal the fill_value
arr = SparseArray([0], fill_value=0)

print("Testing SparseArray([0], fill_value=0)")
print(f"Array: {arr}")
print(f"Dense equivalent: {arr.to_dense()}")

# Test argmin
print("\nCalling arr.argmin()...")
try:
    result = arr.argmin()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test argmax
print("\nCalling arr.argmax()...")
try:
    result = arr.argmax()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Compare with dense array behavior
print("\n--- Comparison with dense array ---")
dense = arr.to_dense()
print(f"dense.argmin() = {dense.argmin()}")
print(f"dense.argmax() = {dense.argmax()}")

# Also test with regular numpy array for reference
import numpy as np
numpy_arr = np.array([0])
print(f"\nnumpy.array([0]).argmin() = {numpy_arr.argmin()}")
print(f"numpy.array([0]).argmax() = {numpy_arr.argmax()}")
```

<details>

<summary>
ValueError: attempt to get argmin of an empty sequence
</summary>
```
Testing SparseArray([0], fill_value=0)
Array: [0]
Fill: 0
IntIndex
Indices: array([], dtype=int32)

Dense equivalent: [0]

Calling arr.argmin()...
Error: ValueError: attempt to get argmin of an empty sequence

Calling arr.argmax()...
Error: ValueError: attempt to get argmax of an empty sequence

--- Comparison with dense array ---
dense.argmin() = 0
dense.argmax() = 0

numpy.array([0]).argmin() = 0
numpy.array([0]).argmax() = 0
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Breaks Array Interface Contract**: SparseArray is designed to be functionally equivalent to dense arrays, just with optimized storage. The methods argmin() and argmax() are fundamental array operations that should work on any non-empty array. NumPy arrays and pandas dense arrays correctly return index 0 when all values are equal.

2. **Inconsistent with Dense Representation**: The same SparseArray converted to dense via `.to_dense()` works correctly, returning index 0 for both argmin() and argmax(). This inconsistency breaks the principle that sparse and dense representations should behave identically.

3. **Implementation Oversight**: The crash occurs at line 1658 in `/pandas/core/arrays/sparse/array.py` where `np.argmin(non_nans)` is called on an empty array. When all values equal the fill_value, SparseArray optimizes storage by storing no sparse values (sp_values is empty). The code doesn't check for this edge case before calling NumPy's argmin/argmax functions.

4. **Undocumented Limitation**: There is no documentation indicating that argmin/argmax should fail when all values equal the fill_value. The methods have no special restrictions mentioned in their signatures or docstrings.

5. **Low-Level Error Message**: The error "attempt to get argmin of an empty sequence" is a low-level NumPy error rather than a deliberate, informative exception, indicating this is an unhandled edge case rather than intended behavior.

## Relevant Context

The bug affects any SparseArray where all elements equal the fill_value, regardless of array size or fill_value choice:
- `SparseArray([0], fill_value=0)` - crashes
- `SparseArray([0, 0, 0], fill_value=0)` - crashes
- `SparseArray([5, 5, 5], fill_value=5)` - crashes
- `SparseArray([0, 1, 0], fill_value=0)` - works (mixed values)
- `SparseArray([1, 1, 1], fill_value=0)` - works (all equal but not fill_value)

The root cause is in the `_argmin_argmax` method at line 1648-1672 of `pandas/core/arrays/sparse/array.py`. When all values equal fill_value:
- `self._sparse_values` returns an empty array `[]`
- `self._sparse_index.indices` returns an empty array `[]`
- Line 1658 calls `func(non_nans)` where `non_nans` is empty, triggering the ValueError

Workaround for users: Convert to dense first with `.to_dense().argmin()` or `.to_dense().argmax()`

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1654,8 +1654,17 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         idx = np.arange(values.shape[0])
         non_nans = values[~mask]
         non_nan_idx = idx[~mask]

-        _candidate = non_nan_idx[func(non_nans)]
+        # Handle case where all values equal fill_value (empty sparse values)
+        if len(non_nans) == 0:
+            # No sparse values exist - all values are fill_value
+            _loc = self._first_fill_value_loc()
+            if _loc == -1:
+                # Array is completely empty
+                raise ValueError(f"attempt to get {kind} of an empty sequence")
+            return _loc
+
+        _candidate = non_nan_idx[func(non_nans)]
         candidate = index[_candidate]

         if isna(self.fill_value):
```