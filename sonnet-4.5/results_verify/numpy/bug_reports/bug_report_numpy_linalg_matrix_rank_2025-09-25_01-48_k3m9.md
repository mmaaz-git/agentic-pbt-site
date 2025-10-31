# Bug Report: numpy.linalg.matrix_rank Tolerance Calculation

**Target**: `numpy.linalg.matrix_rank`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The default tolerance calculation in `matrix_rank` is scale-dependent, causing it to violate the fundamental mathematical property that rank(A @ A) ≤ rank(A) when matrices contain extremely small values.

## Property-Based Test

```python
import numpy as np
import numpy.linalg as la
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

@given(
    st.integers(min_value=2, max_value=5).flatmap(
        lambda n: arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
        )
    )
)
def test_matrix_rank_after_multiplication(a):
    rank_a = la.matrix_rank(a)
    rank_aa = la.matrix_rank(a @ a)
    assert rank_aa <= rank_a, f"rank(A @ A) should be <= rank(A)"
```

**Failing input**: `array([[1.40129846e-45, 1.0], [1.40129846e-45, 1.40129846e-45]])`

## Reproducing the Bug

```python
import numpy as np
import numpy.linalg as la

a = np.array([[1.40129846e-45, 1.00000000e+00],
              [1.40129846e-45, 1.40129846e-45]])

rank_a = la.matrix_rank(a)
rank_aa = la.matrix_rank(a @ a)

print(f"rank(A) = {rank_a}")
print(f"rank(A @ A) = {rank_aa}")

u, s, vh = la.svd(a)
print(f"\nSingular values of A: {s}")
print(f"Default tolerance for A: {s[0] * max(a.shape) * np.finfo(a.dtype).eps}")

u2, s2, vh2 = la.svd(a @ a)
print(f"\nSingular values of A @ A: {s2}")
print(f"Default tolerance for A @ A: {s2[0] * max((a @ a).shape) * np.finfo(a.dtype).eps}")

assert rank_aa <= rank_a, f"rank(A @ A) = {rank_aa} > rank(A) = {rank_a}"
```

Output:
```
rank(A) = 1
rank(A @ A) = 2
AssertionError: rank(A @ A) = 2 > rank(A) = 1
```

## Why This Is A Bug

In linear algebra, rank(A @ A) ≤ rank(A) is always true. The bug occurs because `matrix_rank` uses a default tolerance of `max_singular_value * max(dimensions) * machine_epsilon`. When matrix values are extremely small, multiplying the matrix by itself produces even smaller values, leading to a smaller tolerance threshold. This allows numerical noise to be counted as significant, artificially inflating the rank.

For matrix A:
- Largest singular value: 1.0
- Tolerance: ~4.4e-16
- Second singular value (1.4e-45) is below tolerance → rank = 1 ✓

For matrix A @ A:
- Largest singular value: 3.4e-45
- Tolerance: ~6.6e-238 (much smaller!)
- Both singular values (3.4e-45 and 5.8e-46) are above tolerance → rank = 2 ✗

## Fix

The fix should make the tolerance calculation more robust to matrix scale. One approach is to use an absolute tolerance floor or normalize the tolerance calculation:

```diff
--- a/numpy/linalg/_linalg.py
+++ b/numpy/linalg/_linalg.py
@@ -2285,7 +2285,7 @@ def matrix_rank(A, tol=None, hermitian=False, *, rtol=None):
     ...
     if tol is None:
-        tol = S.max() * max(M, N) * eps
+        tol = max(S.max() * max(M, N) * eps, eps)
     ...
```

This ensures the tolerance never falls below machine epsilon, preventing numerical noise from being counted as significant for matrices with very small values.