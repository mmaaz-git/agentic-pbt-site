# Bug Report: numpy.matrixlib.matrix.ptp Crashes When Using out Parameter

**Target**: `numpy.matrixlib.defmatrix.matrix.ptp`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `matrix.ptp()` method crashes with a ValueError when called with the `out` parameter, failing to handle the dimensional requirements correctly for matrix objects.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np
from numpy import matrix


@given(arrays(np.float64, shape=st.tuples(st.integers(3, 20), st.integers(3, 20)),
              elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
@settings(max_examples=300)
def test_ptp_with_out_parameter(arr):
    m = matrix(arr)
    out = matrix(np.zeros((1, 1)))
    result = m.ptp(axis=None, out=out)
    assert result is out
```

<details>

<summary>
**Failing input**: `array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 17, in <module>
    test_ptp_with_out_parameter()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 8, in test_ptp_with_out_parameter
    elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
     ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 13, in test_ptp_with_out_parameter
    result = m.ptp(axis=None, out=out)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 798, in ptp
    return N.ptp(self, axis, out)._align(axis)
           ~~~~~^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 3043, in ptp
    return _methods._ptp(a, axis=axis, out=out, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_methods.py", line 236, in _ptp
    umr_maximum(a, axis, None, out, keepdims),
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: output parameter for reduction operation maximum has the wrong number of dimensions: Found 2 but expected 0
Falsifying example: test_ptp_with_out_parameter(
    arr=array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]]),  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from numpy import matrix

# Create a simple 2x2 matrix
m = matrix([[1, 2], [3, 4]])
print("Matrix m:")
print(m)
print()

# Create an output matrix
out = matrix([[0.0]])
print("Output matrix:")
print(out)
print()

# Try to call ptp with the out parameter
try:
    print("Calling m.ptp(axis=None, out=out)...")
    result = m.ptp(axis=None, out=out)
    print("Result:")
    print(result)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError: output parameter for reduction operation maximum has the wrong number of dimensions
</summary>
```
Matrix m:
[[1 2]
 [3 4]]

Output matrix:
[[0.]]

Calling m.ptp(axis=None, out=out)...
Error occurred: ValueError: output parameter for reduction operation maximum has the wrong number of dimensions: Found 2 but expected 0
```
</details>

## Why This Is A Bug

The `matrix.ptp()` method accepts an `out` parameter according to its documented interface, but fails to handle it correctly due to a missing `keepdims=True` parameter when calling the underlying NumPy function. This inconsistency violates the expected behavior in several ways:

1. **API Contract Violation**: The method signature and documentation indicate that `out` is a valid parameter, yet using it causes a crash rather than working as expected.

2. **Inconsistent Implementation**: All other reduction methods in the matrix class (`sum`, `mean`, `std`, `var`, `prod`, `max`, `min`, `any`, `all`) correctly use `keepdims=True` when calling their underlying NumPy functions to maintain matrix dimensionality. The `ptp` method is the sole exception.

3. **Dimension Mismatch**: When `axis=None`, the underlying `numpy.ptp` function expects a 0-dimensional (scalar) output, but the matrix class provides a 2-dimensional matrix as the `out` parameter. Without `keepdims=True`, this dimension mismatch causes the ValueError.

## Relevant Context

The matrix class implementation shows a clear pattern for reduction operations. Looking at the code in `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py`:

- Line 327 (sum): `return N.ndarray.sum(self, axis, dtype, out, keepdims=True)._collapse(axis)`
- Line 451 (mean): `return N.ndarray.mean(self, axis, dtype, out, keepdims=True)._collapse(axis)`
- Line 486 (std): Uses `keepdims=True`
- Line 521 (var): Uses `keepdims=True`
- Line 554 (prod): Uses `keepdims=True`
- Line 652 (max): Uses `keepdims=True`
- Line 726 (min): Uses `keepdims=True`
- Line 798 (ptp): `return N.ptp(self, axis, out)._align(axis)` - **Missing keepdims=True**

The `_collapse` and `_align` methods (lines 248-268) are helper methods designed to handle the dimension preservation and collapsing logic specific to matrix objects after the reduction operation.

Documentation reference: https://numpy.org/doc/stable/reference/generated/numpy.matrixlib.defmatrix.matrix.ptp.html

## Proposed Fix

The fix is straightforward - add the `keepdims=True` parameter to maintain consistency with all other reduction methods:

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -795,7 +795,7 @@ class matrix(N.ndarray):
                 [3]])

         """
-        return N.ptp(self, axis, out)._align(axis)
+        return N.ptp(self, axis, out, keepdims=True)._align(axis)
```