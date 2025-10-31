# Bug Report: xarray.core.duck_array_ops cumprod/cumsum Incorrect Behavior with axis=None

**Target**: `xarray.core.duck_array_ops.cumprod` and `xarray.core.duck_array_ops.cumsum`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `axis=None` is specified, xarray's `cumprod()` and `cumsum()` functions return incorrect shapes and values compared to numpy. Instead of flattening the array and computing cumulative operations on the flattened result (as numpy does), xarray incorrectly applies the operation sequentially along each axis.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from xarray.core import duck_array_ops


@given(
    st.integers(min_value=2, max_value=8),
    st.integers(min_value=2, max_value=8)
)
@settings(max_examples=100)
def test_cumprod_axis_none_matches_numpy(rows, cols):
    values = np.random.randn(rows, cols)

    xr_result = duck_array_ops.cumprod(values, axis=None)
    np_result = np.cumprod(values, axis=None)

    assert xr_result.shape == np_result.shape, \
        f"Shape mismatch: {xr_result.shape} != {np_result.shape}"
    assert np.allclose(xr_result.flatten(), np_result), \
        f"Values mismatch"


@given(
    st.integers(min_value=2, max_value=8),
    st.integers(min_value=2, max_value=8)
)
@settings(max_examples=100)
def test_cumsum_axis_none_matches_numpy(rows, cols):
    values = np.random.randn(rows, cols)

    xr_result = duck_array_ops.cumsum(values, axis=None)
    np_result = np.cumsum(values, axis=None)

    assert xr_result.shape == np_result.shape, \
        f"Shape mismatch: {xr_result.shape} != {np_result.shape}"
    assert np.allclose(xr_result.flatten(), np_result), \
        f"Values mismatch"


# Run tests
if __name__ == "__main__":
    print("Testing cumprod...")
    test_cumprod_axis_none_matches_numpy()
    print("Testing cumsum...")
    test_cumsum_axis_none_matches_numpy()
```

<details>

<summary>
**Failing input**: `rows=2, cols=2`
</summary>
```
Testing cumprod...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 43, in <module>
    test_cumprod_axis_none_matches_numpy()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 7, in test_cumprod_axis_none_matches_numpy
    st.integers(min_value=2, max_value=8),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 17, in test_cumprod_axis_none_matches_numpy
    assert xr_result.shape == np_result.shape, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Shape mismatch: (2, 2) != (4,)
Falsifying example: test_cumprod_axis_none_matches_numpy(
    rows=2,
    cols=2,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from xarray.core import duck_array_ops

# Test array
arr = np.array([[1, 2], [3, 4]])

print("Testing cumprod with axis=None")
print("=" * 50)

# xarray cumprod
xarray_cumprod = duck_array_ops.cumprod(arr, axis=None)
print(f"xarray result: shape={xarray_cumprod.shape}, values={xarray_cumprod.flatten().tolist()}")

# numpy cumprod
numpy_cumprod = np.cumprod(arr, axis=None)
print(f"numpy result:  shape={numpy_cumprod.shape}, values={numpy_cumprod.tolist()}")

print("\nTesting cumsum with axis=None")
print("=" * 50)

# xarray cumsum
xarray_cumsum = duck_array_ops.cumsum(arr, axis=None)
print(f"xarray cumsum: shape={xarray_cumsum.shape}, values={xarray_cumsum.flatten().tolist()}")

# numpy cumsum
numpy_cumsum = np.cumsum(arr, axis=None)
print(f"numpy cumsum:  shape={numpy_cumsum.shape}, values={numpy_cumsum.tolist()}")
```

<details>

<summary>
Shape mismatch and incorrect values in xarray results
</summary>
```
Testing cumprod with axis=None
==================================================
xarray result: shape=(2, 2), values=[1, 2, 3, 24]
numpy result:  shape=(4,), values=[1, 2, 6, 24]

Testing cumsum with axis=None
==================================================
xarray cumsum: shape=(2, 2), values=[1, 3, 4, 10]
numpy cumsum:  shape=(4,), values=[1, 3, 6, 10]
```
</details>

## Why This Is A Bug

According to numpy's documentation, when `axis=None` is passed to cumulative operations like `cumprod` and `cumsum`, the array should be flattened first, then the cumulative operation applied along the flattened 1D array. The result should be a 1D array with the same total number of elements as the input.

xarray's implementation violates this expected behavior. The bug is in the `_nd_cum_func` function (lines 785-786 in duck_array_ops.py) which incorrectly converts `axis=None` to `tuple(range(array.ndim))`. This causes xarray to apply the cumulative operation sequentially along each axis (first along axis 0, then along axis 1 for a 2D array), which produces entirely different mathematical results than numpy's flatten-then-accumulate approach.

For the test array [[1, 2], [3, 4]]:
- **numpy cumprod**: Flattens to [1, 2, 3, 4], then computes cumulative product: [1, 1×2=2, 2×3=6, 6×4=24]
- **xarray cumprod**: First applies along axis 0: [[1, 2], [3, 8]], then along axis 1: [[1, 2], [3, 24]]
- **numpy cumsum**: Flattens to [1, 2, 3, 4], then computes cumulative sum: [1, 1+2=3, 3+3=6, 6+4=10]
- **xarray cumsum**: First applies along axis 0: [[1, 2], [4, 6]], then along axis 1: [[1, 3], [4, 10]]

## Relevant Context

The underlying implementations (`cumprod_1d` and `cumsum_1d`) are created via `_create_nan_agg_method` and ultimately delegate to numpy's functions (via `xp.cumprod` and `xp.cumsum`), which would handle `axis=None` correctly by flattening. The bug is introduced solely by the `_nd_cum_func` wrapper which intercepts `axis=None` and incorrectly transforms it.

xarray generally aims for numpy compatibility and builds upon numpy's array interface. The current behavior is neither documented nor intuitive, and appears to be an unintentional bug rather than a design choice.

Documentation references:
- numpy.cumprod: https://numpy.org/doc/stable/reference/generated/numpy.cumprod.html
- numpy.cumsum: https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
- xarray duck_array_ops.py: /home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/core/duck_array_ops.py:785-786

## Proposed Fix

```diff
--- a/xarray/core/duck_array_ops.py
+++ b/xarray/core/duck_array_ops.py
@@ -783,8 +783,10 @@ def _mean(array, axis=None, skipna=None, **kwargs):
 def _nd_cum_func(cum_func, array, axis, **kwargs):
     array = asarray(array)
     if axis is None:
-        axis = tuple(range(array.ndim))
+        # Match numpy behavior: flatten array first for axis=None
+        return cum_func(array.flatten(), axis=0, **kwargs)
     if isinstance(axis, int):
         axis = (axis,)

     out = array
     for ax in axis:
```