# Bug Report: numpy.ma.compress_rowcols Family Returns Inconsistent Array Dimensions

**Target**: `numpy.ma.compress_rowcols`, `numpy.ma.compress_rows`, `numpy.ma.compress_cols`, `numpy.ma.compress_nd`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `compress_rowcols` family of functions returns 1D arrays when given fully masked 2D input, but returns 2D arrays when given partially masked 2D input that results in complete removal. This dimensionality inconsistency violates the expectation that 2D input should produce 2D output.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra import numpy as npst

@st.composite
def masked_2d_arrays(draw):
    shape = draw(npst.array_shapes(min_dims=2, max_dims=2, max_side=10))
    data = draw(npst.arrays(dtype=np.int64, shape=shape,
                           elements=st.integers(min_value=-100, max_value=100)))
    mask = draw(npst.arrays(dtype=bool, shape=shape))
    return ma.array(data, mask=mask)

@given(masked_2d_arrays())
@settings(max_examples=1000)
def test_compress_rowcols_maintains_2d(arr):
    result = ma.compress_rowcols(arr)
    assert result.ndim == 2

if __name__ == "__main__":
    test_compress_rowcols_maintains_2d()
```

<details>

<summary>
**Failing input**: `masked_array(data=[[--]], mask=[[True]])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 21, in <module>
    test_compress_rowcols_maintains_2d()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 15, in test_compress_rowcols_maintains_2d
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 18, in test_compress_rowcols_maintains_2d
    assert result.ndim == 2
           ^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_compress_rowcols_maintains_2d(
    arr=masked_array(data=[[--]],
                 mask=[[ True]],
           fill_value=999999,
                dtype=int64),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/extras.py:948
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

# Test 1: Fully masked array
arr1 = ma.array([[99]], mask=[[True]])
result1 = ma.compress_rowcols(arr1)
print(f"compress_rowcols fully masked: shape={arr1.shape} -> {result1.shape}, ndim={arr1.ndim} -> {result1.ndim}")

result2 = ma.compress_rows(arr1)
print(f"compress_rows fully masked: shape={arr1.shape} -> {result2.shape}, ndim={arr1.ndim} -> {result2.ndim}")

result3 = ma.compress_cols(arr1)
print(f"compress_cols fully masked: shape={arr1.shape} -> {result3.shape}, ndim={arr1.ndim} -> {result3.ndim}")

# Test 2: Partially masked 2D array where all rows/columns get removed
arr2 = ma.array([[1, 2], [3, 4]], mask=[[True, False], [False, True]])
result4 = ma.compress_rowcols(arr2)
print(f"\nPartially masked, all removed: shape={arr2.shape} -> {result4.shape}, ndim={arr2.ndim} -> {result4.ndim}")

# Test 3: Show the actual array results
print(f"\nActual results:")
print(f"Fully masked [[--]]: result = {result1}, type = {type(result1)}")
print(f"Partially masked with all removed: result = {result4}, type = {type(result4)}")
```

<details>

<summary>
Dimensionality inconsistency demonstrated
</summary>
```
compress_rowcols fully masked: shape=(1, 1) -> (0,), ndim=2 -> 1
compress_rows fully masked: shape=(1, 1) -> (0,), ndim=2 -> 1
compress_cols fully masked: shape=(1, 1) -> (0,), ndim=2 -> 1

Partially masked, all removed: shape=(2, 2) -> (0, 0), ndim=2 -> 2

Actual results:
Fully masked [[--]]: result = [], type = <class 'numpy.ndarray'>
Partially masked with all removed: result = [], type = <class 'numpy.ndarray'>
```
</details>

## Why This Is A Bug

This violates expected behavior in three critical ways:

1. **Dimensionality Inconsistency**: The same conceptual operation (removing all data) produces different dimensional results depending on whether the input is fully masked (returns 1D with shape `(0,)`) or partially masked with all elements removed (returns 2D with shape `(0, 0)`).

2. **Documentation Contract Violation**: The documentation explicitly states that the input "Must be a 2D array" and describes the functions as operating on rows and columns (inherently 2D concepts). The function names themselves (`compress_rows`, `compress_cols`, `compress_rowcols`) are specifically about 2D operations. When a function requires 2D input and operates on 2D concepts, users reasonably expect 2D output.

3. **Breaking Array Operation Chains**: This inconsistency breaks code that chains array operations. For example:
   - Broadcasting operations expect consistent dimensions
   - Subsequent matrix operations may fail unexpectedly
   - Shape-dependent logic will behave incorrectly

The bug occurs at line 948 in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/extras.py` where `compress_nd()` returns `nxarray([])` for fully masked arrays, creating a 1D empty array regardless of input dimensionality.

## Relevant Context

The `compress_rowcols` function is a wrapper around `compress_nd` (line 1010 in extras.py):
```python
def compress_rowcols(x, axis=None):
    if asarray(x).ndim != 2:
        raise NotImplementedError("compress_rowcols works for 2D arrays only.")
    return compress_nd(x, axis=axis)
```

The inconsistency stems from `compress_nd` at line 947-948:
```python
if m.all():
    return nxarray([])  # Always returns 1D empty array
```

While the partial masking case (lines 949-954) preserves dimensions through boolean indexing:
```python
data = x._data
for ax in axis:
    axes = tuple(list(range(ax)) + list(range(ax + 1, x.ndim)))
    data = data[(slice(None),) * ax + (~m.any(axis=axes),)]
return data  # Preserves original dimensionality
```

Documentation references:
- numpy.ma.compress_rowcols: https://numpy.org/doc/stable/reference/generated/numpy.ma.compress_rowcols.html
- Source code: https://github.com/numpy/numpy/blob/main/numpy/ma/extras.py

## Proposed Fix

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -945,7 +945,12 @@ def compress_nd(x, axis=None):
         return x._data
     # All is masked: return empty
     if m.all():
-        return nxarray([])
+        # Preserve dimensionality - return empty array with appropriate shape
+        result_shape = list(x.shape)
+        for ax in axis:
+            result_shape[ax] = 0
+        return nxarray([]).reshape(result_shape)
     # Filter elements through boolean indexing
     data = x._data
     for ax in axis:
```