# Bug Report: numpy.linalg.matrix_rank Violates Rank Product Inequality

**Target**: `numpy.linalg.matrix_rank`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.linalg.matrix_rank` violates the fundamental mathematical property that rank(A @ A) ≤ rank(A) when using default tolerances on matrices with extreme (but valid) floating-point values.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np


@given(
    st.integers(min_value=1, max_value=8).flatmap(
        lambda size: st.integers(min_value=0, max_value=size).flatmap(
            lambda rank: arrays(
                dtype=np.float64,
                shape=(size, rank),
                elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
            ).flatmap(
                lambda u: arrays(
                    dtype=np.float64,
                    shape=(rank, size),
                    elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
                ).map(lambda v: u @ v)
            )
        )
    )
)
@settings(max_examples=200)
def test_matrix_rank_product_bound(matrix):
    rank_a = np.linalg.matrix_rank(matrix)
    rank_aa = np.linalg.matrix_rank(matrix @ matrix)
    assert rank_aa <= rank_a
```

**Failing input**:
```python
matrix = np.array([[1.49956052e-308, 1.49956052e-308],
                   [1.00000000e+000, 1.49956052e-308]])
```

## Reproducing the Bug

```python
import numpy as np

matrix = np.array([[1.49956052e-308, 1.49956052e-308],
                   [1.00000000e+000, 1.49956052e-308]])

rank_a = np.linalg.matrix_rank(matrix)
rank_aa = np.linalg.matrix_rank(matrix @ matrix)

print(f"rank(A): {rank_a}")
print(f"rank(A @ A): {rank_aa}")
print(f"Property violated: {rank_aa} > {rank_a}")

svd_a = np.linalg.svd(matrix, compute_uv=False)
svd_aa = np.linalg.svd(matrix @ matrix, compute_uv=False)
print(f"\nSingular values of A: {svd_a}")
print(f"Singular values of A @ A: {svd_aa}")

eps = np.finfo(float).eps
print(f"\nDefault tolerance for A: {svd_a[0] * max(matrix.shape) * eps}")
print(f"Default tolerance for A @ A: {svd_aa[0] * max(matrix.shape) * eps}")
```

**Output**:
```
rank(A): 1
rank(A @ A): 2
Property violated: 2 > 1

Singular values of A: [1.00000000e+000 1.49956052e-308]
Singular values of A @ A: [3.62025934e-308 6.21138305e-309]

Default tolerance for A: 4.440892098500626e-16
Default tolerance for A @ A: 1.607153631294337e-323
```

## Why This Is A Bug

The mathematical property rank(A @ A) ≤ rank(A) is a fundamental theorem in linear algebra and must hold for all matrices. The violation occurs because:

1. `matrix_rank` uses tolerance = `max_singular_value * max(shape) * machine_epsilon`
2. For A, this gives tolerance ≈ 4.4e-16, correctly classifying 1.5e-308 as zero
3. For A @ A (all tiny values), tolerance ≈ 1.6e-323, incorrectly classifying 6.2e-309 as non-zero

This inconsistent tolerance scaling causes the function to violate a core mathematical invariant when matrices contain extreme but valid floating-point values.

## Fix

The tolerance calculation should be more robust to prevent such inconsistencies. One approach would be to use an absolute minimum tolerance threshold:

```diff
--- a/numpy/linalg/_linalg.py
+++ b/numpy/linalg/_linalg.py
@@ -2100,7 +2100,8 @@ def matrix_rank(A, tol=None, hermitian=False, *, rtol=None):
     if tol is None:
         if rtol is None:
-            tol = S.max(axis=-1, keepdims=True) * max(M, N) * eps
+            default_tol = S.max(axis=-1, keepdims=True) * max(M, N) * eps
+            tol = np.maximum(default_tol, eps * 1e2)
         else:
             tol = S.max(axis=-1, keepdims=True) * rtol
```

This ensures the tolerance never drops below a reasonable absolute threshold, preventing the rank function from treating values near the floating-point underflow limit as significant.