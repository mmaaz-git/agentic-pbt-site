# Bug Report: scipy.linalg.inv Fails to Detect Singular Matrices with Subnormal Values

**Target**: `scipy.linalg.inv`
**Severity**: High
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.linalg.inv` fails to detect singularity for matrices containing subnormal floating-point values, returning matrices with Inf values instead of raising the documented `LinAlgError` exception.

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

if __name__ == "__main__":
    test_inv_singular_detection()
```

<details>

<summary>
**Failing input**: `array([[5.e-324, 5.e-324], [5.e-324, 5.e-324]])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 32, in <module>
    test_inv_singular_detection()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 8, in test_inv_singular_detection
    dtype=np.float64,
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 26, in test_inv_singular_detection
    assert False, f"inv() returned NaN/Inf for singular matrix (rank={rank}, det={det})"
           ^^^^^
AssertionError: inv() returned NaN/Inf for singular matrix (rank=1, det=0.0)
Falsifying example: test_inv_singular_detection(
    A=array([[5.e-324, 5.e-324],
           [5.e-324, 5.e-324]]),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/.local/lib/python3.13/site-packages/scipy/linalg/_basic.py:1260
        /home/npc/.local/lib/python3.13/site-packages/scipy/linalg/_basic.py:1296
        /home/npc/pbt/agentic-pbt/worker_/17/hypo.py:24
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.linalg as la

# Using the failing example from Hypothesis
A = np.array([[5.e-324, 5.e-324],
              [5.e-324, 5.e-324]])

print(f"Matrix A:\n{A}")
print(f"Rank: {np.linalg.matrix_rank(A)}")
print(f"Determinant: {la.det(A)}")

try:
    A_inv = la.inv(A)
    print(f"inv(A):\n{A_inv}")
    print("No LinAlgError was raised!")
except la.LinAlgError as e:
    print(f"LinAlgError was raised: {e}")
```

<details>

<summary>
Output shows singular matrix returns Inf values without raising LinAlgError
</summary>
```
Matrix A:
[[5.e-324 5.e-324]
 [5.e-324 5.e-324]]
Rank: 1
Determinant: 0.0
inv(A):
[[ inf -inf]
 [-inf  inf]]
No LinAlgError was raised!
```
</details>

## Why This Is A Bug

This is a clear contract violation of the scipy.linalg.inv API. The documentation explicitly states:

> **Raises**
> **LinAlgError**
>     If `a` is singular.

The matrix in the test case is mathematically singular:
- **Rank deficient**: The matrix has rank 1 (less than the required rank 2 for a 2x2 matrix to be invertible)
- **Zero determinant**: The determinant is exactly 0.0, which is the mathematical definition of a singular matrix
- **Identical rows**: Both rows contain identical values, making the matrix linearly dependent

The function returns a matrix containing Inf values instead of raising `LinAlgError`. This violates the documented behavior with no mentioned exceptions for subnormal values. The check_finite parameter only validates input matrices for Inf/NaN, not output behavior.

## Relevant Context

**Subnormal numbers**: The failing values (5e-324) are subnormal floating-point numbers - valid IEEE 754 values between 0 and the smallest normal positive float64 (~2.225e-308). These can arise naturally in scientific computing through underflow or extreme scaling.

**LAPACK internals**: The issue stems from numerical precision limitations in LAPACK's `getrf` (LU factorization) and `getri` (inversion) routines. Looking at the scipy source code in `/scipy/linalg/_basic.py`, the singularity check only occurs when `info > 0` is returned by LAPACK, but with subnormal values, `info == 0` while the result contains Inf.

**Impact**: Any code relying on `LinAlgError` for singular matrix detection could silently propagate Inf values through calculations, causing hard-to-debug failures in scientific computations.

**Related scipy documentation**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html

## Proposed Fix

```diff
--- a/scipy/linalg/_basic.py
+++ b/scipy/linalg/_basic.py
@@ -1182,6 +1182,11 @@ def inv(a, overwrite_a=False, check_finite=True):
         # more such SEGFAULTs occur.
         lwork = int(1.01 * lwork)
         inv_a, info = getri(lu, piv, lwork=lwork, overwrite_lu=1)
+    # Check for NaN/Inf in result which indicates numerical failure
+    if info == 0:
+        if np.any(np.isnan(inv_a)) or np.any(np.isinf(inv_a)):
+            raise LinAlgError("singular matrix")
     if info > 0:
         raise LinAlgError("singular matrix")
     if info < 0:
```