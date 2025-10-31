# Bug Report: scipy.linalg.inv Singular Matrix Detection Failure

**Target**: `scipy.linalg.inv`
**Severity**: High
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.linalg.inv` fails to detect singularity for matrices containing subnormal floating-point values, returning NaN/Inf values instead of raising `LinAlgError` as documented.

## Property-Based Test

```python
import numpy as np
import scipy.linalg as la
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst


@given(npst.arrays(
    dtype=np.float64,
    shape=(2, 2),
    elements=st.floats(
        min_value=-1e100,
        max_value=1e100,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=True
    )
))
@settings(max_examples=1000)
def test_inv_singular_detection(A):
    try:
        A_inv = la.inv(A, check_finite=True)

        if np.any(np.isnan(A_inv)) or np.any(np.isinf(A_inv)):
            rank = np.linalg.matrix_rank(A)
            det = la.det(A)
            assert False, f"inv() returned NaN/Inf for singular matrix (rank={rank}, det={det})"

    except la.LinAlgError:
        pass
```

**Failing input**: `array([[2.22507386e-311, 2.22507386e-311], [2.22507386e-311, 2.22507386e-311]])`

## Reproducing the Bug

```python
import numpy as np
import scipy.linalg as la

A = np.array([[2.22507386e-311, 2.22507386e-311],
              [2.22507386e-311, 2.22507386e-311]])

print(f"Matrix A:\n{A}")
print(f"Rank: {np.linalg.matrix_rank(A)}")
print(f"Determinant: {la.det(A)}")

A_inv = la.inv(A)
print(f"inv(A):\n{A_inv}")
```

**Output:**
```
Matrix A:
[[2.22507386e-311 2.22507386e-311]
 [2.22507386e-311 2.22507386e-311]]
Rank: 1
Determinant: 0.0
inv(A):
[[ inf -inf]
 [-inf  inf]]
```

## Why This Is A Bug

The `scipy.linalg.inv` function documentation states:

> **Raises**
>
> **LinAlgError**
>     If `a` is singular.

The matrix in the example is clearly singular (rank 1, determinant 0), yet `inv()` does not raise `LinAlgError`. Instead, it returns a result containing infinity values, which violates the documented API contract.

This is a **contract violation** because:
1. The function promises to raise `LinAlgError` for singular matrices
2. The matrix is demonstrably singular (rank < n, det = 0)
3. The function returns an invalid result instead of raising the documented exception

The bug occurs specifically when matrices contain subnormal floating-point values, which are valid IEEE 754 numbers but very close to zero. The LAPACK routines used internally may have numerical precision issues that cause them to fail to detect singularity in these edge cases.

## Fix

The issue arises from numerical precision limitations in the singularity detection within LAPACK's `getrf` and `getri` routines when dealing with subnormal values. One approach is to add an explicit singularity check after the LU factorization by examining the diagonal of the U matrix for near-zero or NaN/Inf values:

```diff
--- a/scipy/linalg/_basic.py
+++ b/scipy/linalg/_basic.py
@@ -1182,6 +1182,12 @@ def inv(a, overwrite_a=False, check_finite=True):
     lu, piv, info = getrf(a1, overwrite_a=overwrite_a)
     if info == 0:
+        # Check for NaN/Inf in LU factorization result (indicates numerical issues)
+        if np.any(np.isnan(lu)) or np.any(np.isinf(lu)):
+            raise LinAlgError("singular matrix")
+        # Additional check: ensure diagonal elements are not too small
+        if np.any(np.abs(np.diag(lu)) < np.finfo(a1.dtype).tiny):
+            raise LinAlgError("singular matrix")
+
         lwork = _compute_lwork(getri_lwork, a1.shape[0])

         # XXX: the following line fixes curious SEGFAULT when
```

This adds explicit checks for NaN/Inf values in the LU factorization result and verifies that diagonal elements are not smaller than the smallest normal positive number for the dtype, catching cases where subnormal values lead to numerical breakdown.