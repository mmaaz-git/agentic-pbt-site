# Bug Report: numpy.matrixlib.matrix.I Pseudoinverse Documentation

**Target**: `numpy.matrixlib.defmatrix.matrix.I`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The docstring for `matrix.I` property claims that for both square and non-square matrices, `ret * self == self * ret == np.matrix(np.eye(self[0,:].size))` all return True. This claim is mathematically impossible for non-square matrices due to shape mismatch.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis import strategies as st

@settings(max_examples=300)
@given(
    arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(1, 5), st.integers(1, 5)),
        elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
    )
)
def test_inverse_property_square(arr):
    m = np.matrix(arr)
    try:
        inv = m.I
    except np.linalg.LinAlgError:
        return

    product1 = inv * m
    product2 = m * inv
    eye = np.matrix(np.eye(m[0,:].size))

    assert np.allclose(product1, eye, atol=1e-8), f"inv * m should equal identity"
    assert np.allclose(product2, eye, atol=1e-8), f"m * inv should equal identity"
```

**Failing input**: `array([[0., 0., 0.], [0., 0., 0.]])`

## Reproducing the Bug

```python
import numpy as np

m = np.matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

inv = m.I
product1 = inv * m
product2 = m * inv
eye_from_docstring = np.matrix(np.eye(m[0,:].size))

print(f"Matrix shape: {m.shape}")
print(f"Pseudoinverse shape: {inv.shape}")
print(f"inv * m shape: {product1.shape}")
print(f"m * inv shape: {product2.shape}")
print(f"eye(m[0,:].size) shape: {eye_from_docstring.shape}")

try:
    result = (product2 == eye_from_docstring)
except ValueError as e:
    print(f"ValueError: {e}")
```

Output:
```
Matrix shape: (2, 3)
Pseudoinverse shape: (3, 2)
inv * m shape: (3, 3)
m * inv shape: (2, 2)
eye(m[0,:].size) shape: (3, 3)
ValueError: operands could not be broadcast together with shapes (2,2) (3,3)
```

## Why This Is A Bug

The docstring at lines 812-813 in `/numpy/matrixlib/defmatrix.py` states:

> If `self` is non-singular, `ret` is such that `ret * self` == `self * ret` == `np.matrix(np.eye(self[0,:].size))` all return `True`.

For a non-square matrix of shape (M, N) where M ≠ N:
- The pseudoinverse has shape (N, M)
- `inv * m` has shape (N, N)
- `m * inv` has shape (M, M)
- `eye(m[0,:].size)` has shape (N, N)

The claim that both `inv * m` and `m * inv` equal `eye(N)` is impossible because they have different shapes when M ≠ N.

## Fix

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -809,9 +809,12 @@ class matrix(N.ndarray):
         -------
         ret : matrix object
-            If `self` is non-singular, `ret` is such that ``ret * self`` ==
-            ``self * ret`` == ``np.matrix(np.eye(self[0,:].size))`` all return
-            ``True``.
+            For square non-singular matrices, `ret` is the inverse such that
+            ``ret * self`` and ``self * ret`` both equal the identity matrix.
+            For non-square matrices (M×N), `ret` is the pseudoinverse such that:
+            - ``ret * self`` approximates ``np.eye(N)`` (N×N identity)
+            - ``self * ret`` approximates ``np.eye(M)`` (M×M identity)
+            where M, N are the matrix dimensions.

         Raises
         ------
```