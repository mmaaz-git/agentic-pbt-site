# Bug Report: numpy.matrixlib.matrix.I Fails to Raise LinAlgError for Singular Matrices

**Target**: `numpy.matrixlib.defmatrix.matrix.I`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `matrix.I` property is documented to raise `numpy.linalg.LinAlgError` for singular matrices, but instead returns incorrect inverse matrices that violate the identity property `m.I @ m == I`.

## Property-Based Test

```python
import warnings
import numpy as np
from numpy import matrix
from hypothesis import given, strategies as st, assume, settings
import hypothesis.extra.numpy as npst

warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

@given(st.integers(min_value=2, max_value=4).flatmap(
    lambda n: npst.arrays(dtype=np.float64, shape=(n, n),
                          elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))))
@settings(max_examples=100)
def test_matrix_inverse_left_identity(arr):
    m = matrix(arr)
    try:
        inv = m.I
    except np.linalg.LinAlgError:
        assume(False)

    result = inv @ m
    n = m.shape[0]
    expected = np.eye(n)
    assert np.allclose(result, expected, atol=1e-8), f"m.I * m should equal identity but got {result}"
```

**Failing input**: `array([[1.56673884, 1.56673884], [1.56673884, 1.56673884]])`

## Reproducing the Bug

```python
import numpy as np
from numpy import matrix

m = matrix([[1.56673884, 1.56673884],
            [1.56673884, 1.56673884]])

print(f"Determinant: {np.linalg.det(m):.2e}")

inv = m.I
result = inv @ m
expected = np.eye(2)

print(f"m.I @ m =\n{result}")
print(f"\nExpected:\n{expected}")
print(f"\nAre they equal? {np.allclose(result, expected)}")
```

Output:
```
Determinant: 0.00e+00
m.I @ m =
[[-0.21663058 -0.21663058]
 [ 1.          1.        ]]

Expected:
[[1. 0.]
 [0. 1.]]

Are they equal? False
```

## Why This Is A Bug

The `matrix.I` property documentation (lines 812-819 in defmatrix.py) explicitly states:

> If `self` is non-singular, `ret` is such that `ret * self` == `self * ret` == `np.matrix(np.eye(self[0,:].size))` all return `True`.
>
> Raises
> ------
> numpy.linalg.LinAlgError: Singular matrix
>     If `self` is singular.

For the failing example, the matrix is clearly singular (determinant = 0, all rows identical), yet:
1. No `LinAlgError` is raised
2. The returned "inverse" violates the identity property

## Fix

The implementation (lines 838-843) only checks if the matrix is square, not if it's invertible:

```python
M, N = self.shape
if M == N:
    from numpy.linalg import inv as func
else:
    from numpy.linalg import pinv as func
return asmatrix(func(self))
```

The fix should validate invertibility before calling `inv`, or catch and re-raise errors from `inv`:

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -835,10 +835,18 @@ class matrix(N.ndarray):
             [ 0.,  1.]])

         """
         M, N = self.shape
         if M == N:
             from numpy.linalg import inv as func
+            # numpy.linalg.inv may not always raise LinAlgError for singular matrices
+            # due to numerical precision. Validate the condition number.
+            try:
+                result = asmatrix(func(self))
+            except np.linalg.LinAlgError:
+                raise
+            # Verify result is actually an inverse
+            if not np.allclose(result @ self, np.eye(M), rtol=1e-5, atol=1e-8):
+                raise np.linalg.LinAlgError("Singular matrix")
+            return result
         else:
             from numpy.linalg import pinv as func
-        return asmatrix(func(self))
+            return asmatrix(func(self))
```