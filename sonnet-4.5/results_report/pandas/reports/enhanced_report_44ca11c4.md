# Bug Report: pandas.core.arrays.sparse.SparseArray argmin/argmax ValueError on All-Fill-Value Arrays

**Target**: `pandas.core.arrays.sparse.SparseArray.argmin` and `pandas.core.arrays.sparse.SparseArray.argmax`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

SparseArray's argmin() and argmax() methods crash with a ValueError when called on arrays where all values equal the fill_value, instead of returning index 0 as numpy arrays and other pandas array types do.

## Property-Based Test

```python
import numpy as np
from pandas.core.arrays.sparse import SparseArray
from hypothesis import given, strategies as st, settings

@settings(max_examples=100)
@given(
    st.integers(min_value=-100, max_value=100),
    st.integers(min_value=1, max_value=20)
)
def test_argmin_argmax_all_fill_values(fill_value, size):
    data = [fill_value] * size
    arr = SparseArray(data, fill_value=fill_value)
    dense = np.array(data)

    assert arr.argmin() == dense.argmin()
    assert arr.argmax() == dense.argmax()

if __name__ == "__main__":
    test_argmin_argmax_all_fill_values()
```

<details>

<summary>
**Failing input**: `fill_value=0, size=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 19, in <module>
    test_argmin_argmax_all_fill_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 6, in test_argmin_argmax_all_fill_values
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 15, in test_argmin_argmax_all_fill_values
    assert arr.argmin() == dense.argmin()
           ~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1684, in argmin
    return self._argmin_argmax("argmin")
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1658, in _argmin_argmax
    _candidate = non_nan_idx[func(non_nans)]
                             ~~~~^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 1439, in argmin
    return _wrapfunc(a, 'argmin', axis=axis, out=out, **kwds)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
ValueError: attempt to get argmin of an empty sequence
Falsifying example: test_argmin_argmax_all_fill_values(
    fill_value=0,
    size=1,
)
```
</details>

## Reproducing the Bug

```python
from pandas.core.arrays.sparse import SparseArray
import numpy as np

# Test case 1: Single element array with fill_value=0
print("Test 1: Single element array [0]")
print("-" * 40)
try:
    arr = SparseArray([0])
    print(f"SparseArray([0]).argmin() = {arr.argmin()}")
except Exception as e:
    print(f"SparseArray([0]).argmin() raised: {type(e).__name__}: {e}")

dense = np.array([0])
print(f"numpy.array([0]).argmin() = {dense.argmin()}")
print()

# Test case 2: Multiple equal values with fill_value
print("Test 2: Array [5,5,5] with fill_value=5")
print("-" * 40)
try:
    arr = SparseArray([5, 5, 5], fill_value=5)
    print(f"SparseArray([5,5,5], fill_value=5).argmin() = {arr.argmin()}")
except Exception as e:
    print(f"SparseArray([5,5,5], fill_value=5).argmin() raised: {type(e).__name__}: {e}")

dense = np.array([5, 5, 5])
print(f"numpy.array([5,5,5]).argmin() = {dense.argmin()}")
print()

# Test case 3: argmax with same conditions
print("Test 3: argmax on [0,0,0] with fill_value=0")
print("-" * 40)
try:
    arr = SparseArray([0, 0, 0], fill_value=0)
    print(f"SparseArray([0,0,0], fill_value=0).argmax() = {arr.argmax()}")
except Exception as e:
    print(f"SparseArray([0,0,0], fill_value=0).argmax() raised: {type(e).__name__}: {e}")

dense = np.array([0, 0, 0])
print(f"numpy.array([0,0,0]).argmax() = {dense.argmax()}")
print()

# Test case 4: Working case - mixed values
print("Test 4: Working case - [0,1,0] with fill_value=0")
print("-" * 40)
arr = SparseArray([0, 1, 0], fill_value=0)
print(f"SparseArray([0,1,0], fill_value=0).argmin() = {arr.argmin()}")
print(f"SparseArray([0,1,0], fill_value=0).argmax() = {arr.argmax()}")
```

<details>

<summary>
ValueError: attempt to get argmin/argmax of an empty sequence
</summary>
```
Test 1: Single element array [0]
----------------------------------------
SparseArray([0]).argmin() raised: ValueError: attempt to get argmin of an empty sequence
numpy.array([0]).argmin() = 0

Test 2: Array [5,5,5] with fill_value=5
----------------------------------------
SparseArray([5,5,5], fill_value=5).argmin() raised: ValueError: attempt to get argmin of an empty sequence
numpy.array([5,5,5]).argmin() = 0

Test 3: argmax on [0,0,0] with fill_value=0
----------------------------------------
SparseArray([0,0,0], fill_value=0).argmax() raised: ValueError: attempt to get argmax of an empty sequence
numpy.array([0,0,0]).argmax() = 0

Test 4: Working case - [0,1,0] with fill_value=0
----------------------------------------
SparseArray([0,1,0], fill_value=0).argmin() = 0
SparseArray([0,1,0], fill_value=0).argmax() = 1
```
</details>

## Why This Is A Bug

This violates expected behavior for several reasons:

1. **NumPy Compatibility**: NumPy arrays return index 0 when all values are equal. Per NumPy documentation: "When there are multiple minimum/maximum values, the indices corresponding to the first occurrence are returned." This is standard behavior across all NumPy array types.

2. **Pandas Consistency**: Other pandas data structures follow NumPy's convention:
   - `pd.Series([5,5,5]).argmin()` returns 0
   - `pd.array([0,0,0]).argmin()` returns 0 (for IntegerArray)
   - All pandas ExtensionArrays should provide consistent behavior

3. **Mathematical Validity**: Finding the argmin/argmax of an array where all values are equal is mathematically valid - any index is a correct answer, and the convention is to return the first one.

4. **API Contract Violation**: The methods argmin() and argmax() should work on any non-empty array. The current implementation incorrectly assumes there will always be non-fill values to operate on.

5. **Sparse Array Abstraction Leak**: The crash exposes internal implementation details. Users shouldn't need to know that sparse arrays store values differently - the abstraction should handle all valid inputs seamlessly.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py` at line 1658 in the `_argmin_argmax` method. The problematic code attempts to call `func(non_nans)` where `non_nans` is the array of sparse values after removing NaN values. When all values equal the fill_value, the sparse array has no sparse values (they're all represented implicitly by the fill_value), making `non_nans` empty.

Key implementation details:
- SparseArray stores only values that differ from the fill_value in `_sparse_values`
- When all values equal fill_value, `_sparse_values` is empty
- The method already has logic to handle fill_values (lines 1661-1672) but only reaches it after the crash
- The method `_first_fill_value_loc()` exists and could be used for this case

Documentation references:
- [NumPy argmin documentation](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html)
- [Pandas SparseArray documentation](https://pandas.pydata.org/docs/reference/api/pandas.arrays.SparseArray.html)
- Code location: `pandas/core/arrays/sparse/array.py:1648-1684`

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1655,6 +1655,11 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         non_nans = values[~mask]
         non_nan_idx = idx[~mask]

+        # Handle case where all values are fill_value (empty sparse values)
+        if len(non_nans) == 0:
+            _loc = self._first_fill_value_loc()
+            return _loc if _loc != -1 else 0
+
         _candidate = non_nan_idx[func(non_nans)]
         candidate = index[_candidate]
```