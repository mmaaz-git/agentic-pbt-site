# Bug Report: numpy.linalg.matrix_rank Incorrect Rank When Tolerance Underflows

**Target**: `numpy.linalg.matrix_rank`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When the default tolerance calculation underflows to exactly 0.0, `matrix_rank` incorrectly counts all non-zero singular values (even those at the limit of floating point precision like 5e-324) as contributing to the matrix rank, violating fundamental linear algebra properties.

## Property-Based Test

```python
from hypothesis import assume, given, settings, strategies as st
from hypothesis.extra import numpy as hpnp
import numpy as np


@given(
    hpnp.arrays(
        dtype=np.float64,
        shape=hpnp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=5),
        elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
    ),
    hpnp.arrays(
        dtype=np.float64,
        shape=hpnp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=5),
        elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
    )
)
@settings(max_examples=200)
def test_matrix_rank_product_bound(a, b):
    assume(a.shape[1] == b.shape[0])

    rank_a = np.linalg.matrix_rank(a)
    rank_b = np.linalg.matrix_rank(b)
    rank_ab = np.linalg.matrix_rank(a @ b)

    assert rank_ab <= min(rank_a, rank_b)
```

**Failing input**:
```python
a = np.array([[5.e-324, 1.e+000],
              [5.e-324, 5.e-324]])
b = np.array([[2.e+000, 5.e-324],
              [5.e-324, 5.e-324]])
```

## Reproducing the Bug

```python
import numpy as np

ab = np.array([[1.5e-323, 4.9e-324],
               [9.9e-324, 0.0e+000]])

u, s, vh = np.linalg.svd(ab)
print(f"Singular values: {s}")

default_tol = max(ab.shape) * np.max(s) * np.finfo(ab.dtype).eps
print(f"Default tolerance: {default_tol}")
print(f"Underflowed to 0.0: {default_tol == 0.0}")

rank_default = np.linalg.matrix_rank(ab)
rank_explicit = np.linalg.matrix_rank(ab, tol=1e-15)

print(f"\nRank with default tol: {rank_default}")
print(f"Rank with tol=1e-15: {rank_explicit}")
```

Output:
```
Singular values: [2.e-323 5.e-324]
Default tolerance: 0.0
Underflowed to 0.0: True

Rank with default tol: 2
Rank with tol=1e-15: 0
```

## Why This Is A Bug

1. **Violates fundamental theorem**: The bug causes `rank(AB) > min(rank(A), rank(B))`, violating a fundamental property of matrix rank in linear algebra.

2. **Unexpected behavior**: When tolerance underflows to 0.0 due to numerical underflow (not user choice), the function treats singular values like 5e-324 (at the absolute limit of float64 precision) as significant, reporting rank 2 for what is effectively a zero matrix.

3. **Inconsistent with explicit tolerance**: Setting `tol=1e-300` (still extremely small) gives the correct rank 0, but the default gives rank 2.

4. **Silent failure**: Users have no indication that the tolerance underflowed, leading to silently incorrect rank calculations.

## Fix

The bug likely occurs when comparing singular values with `tol=0.0`. When the default tolerance underflows to 0.0, a minimum threshold should be used instead:

```diff
--- a/numpy/linalg/linalg.py
+++ b/numpy/linalg/linalg.py
@@ matrix_rank function
     if tol is None:
         tol = S.max(axis=-1, keepdims=True) * max(M, N) * finfo(S.dtype).eps
+        # Ensure tolerance doesn't underflow to exactly 0.0
+        tol = np.maximum(tol, np.finfo(S.dtype).tiny * max(M, N))
     return count_nonzero(S > tol, axis=-1)
```