# Bug Report: scipy.linalg.invpascal - Incorrect Matrix Inversion for n >= 17

**Target**: `scipy.linalg.invpascal`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `scipy.linalg.invpascal` function produces an incorrect inverse for symmetric Pascal matrices when `n >= 17`. The product `pascal(n) @ invpascal(n)` does not equal the identity matrix for these values, violating the fundamental property of matrix inverses.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.linalg

@settings(max_examples=100)
@given(st.integers(min_value=2, max_value=30))
def test_pascal_invpascal_are_inverses(n):
    P = scipy.linalg.pascal(n)
    P_inv = scipy.linalg.invpascal(n, exact=False)

    I = np.eye(n)
    product1 = P @ P_inv
    product2 = P_inv @ P

    assert np.allclose(product1, I, rtol=1e-10, atol=1e-10), \
        f"pascal @ invpascal != I for n={n}, ||P @ P_inv - I|| = {np.linalg.norm(product1 - I)}"
    assert np.allclose(product2, I, rtol=1e-10, atol=1e-10), \
        f"invpascal @ pascal != I for n={n}, ||P_inv @ P - I|| = {np.linalg.norm(product2 - I)}"
```

**Failing input**: `n=19`

## Reproducing the Bug

```python
import numpy as np
import scipy.linalg

n = 19
P = scipy.linalg.pascal(n)
P_inv = scipy.linalg.invpascal(n, exact=False)

product = P @ P_inv
I = np.eye(n)

print(f"||P @ P_inv - I||_F = {np.linalg.norm(product - I, 'fro')}")
print(f"Max absolute error: {np.max(np.abs(product - I))}")

print("\nNon-zero off-diagonal elements in P @ P_inv:")
for i in range(n):
    for j in range(n):
        if i != j and abs(product[i, j]) > 1e-8:
            print(f"  [{i}, {j}] = {product[i, j]}")
```

**Output:**
```
||P @ P_inv - I||_F = 5.916079783099616
Max absolute error: 4.0

Non-zero off-diagonal elements in P @ P_inv:
  [16, 7] = -2.0
  [16, 8] = -1.0
  [16, 9] = -1.0
  [16, 10] = 1.0
  [16, 11] = -2.0
  [17, 10] = -2.0
  [18, 10] = -4.0
  [18, 12] = -2.0
```

## Why This Is A Bug

1. **Violates documented behavior**: The docstring for `invpascal` states it "Returns the inverse of the n x n Pascal matrix" and shows examples where `p.dot(invp)` equals the identity matrix.

2. **Violates mathematical definition**: By definition, if `P_inv` is the inverse of `P`, then `P @ P_inv = I` must hold exactly (or within numerical precision for floating-point).

3. **Large numerical errors**: The Frobenius norm error of ~5.92 and maximum absolute error of 4.0 are far too large to be numerical precision issues. Correct implementations of matrix inverses typically have errors on the order of machine epsilon times the condition number.

4. **Affects real usage**: Any code using `invpascal(n)` for `n >= 17` will get wildly incorrect results, potentially causing silent failures in numerical algorithms.

## Root Cause

The bug is in the computation of inverse matrix elements in `_special_matrices.py` at lines 880-883:

```python
for k in range(n - i):
    v += comb(i + k, k, exact=exact) * comb(i + k, i + k - j, exact=exact)
```

The loop iterates from `k=0` to `k=n-i-1`, but according to the mathematical formula for the inverse of a symmetric Pascal matrix, the sum should have a different upper limit. The correct upper limit should be `j + 1` (summing from `k=0` to `k=j`) based on standard formulas for Pascal matrix inverses.

## Fix

```diff
--- a/scipy/linalg/_special_matrices.py
+++ b/scipy/linalg/_special_matrices.py
@@ -877,7 +877,7 @@ def invpascal(n, kind='symmetric', exact=True):
         for i in range(n):
             for j in range(0, i + 1):
                 v = 0
-                for k in range(n - i):
+                for k in range(j + 1):
                     v += comb(i + k, k, exact=exact) * comb(i + k, i + k - j,
                                                             exact=exact)
                 invp[i, j] = (-1)**(i - j) * v
```

This changes the summation to iterate from `k=0` to `k=j`, which matches the standard mathematical formula for computing elements of the inverse symmetric Pascal matrix. This ensures the computed inverse is correct for all values of `n`.