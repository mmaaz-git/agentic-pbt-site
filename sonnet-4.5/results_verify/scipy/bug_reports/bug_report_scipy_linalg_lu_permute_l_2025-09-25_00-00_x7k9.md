# Bug Report: scipy.linalg.lu LU Decomposition with permute_l=True

**Target**: `scipy.linalg.lu`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `scipy.linalg.lu(A, permute_l=True)` is called on certain rectangular matrices, the returned matrices L and U do not satisfy the documented property `A = L @ U`, violating the API contract.

## Property-Based Test

```python
from hypothesis import given, settings, assume, strategies as st
from hypothesis.extra import numpy as npst
import numpy as np
import scipy.linalg


def matrices(min_side=1, max_side=10, dtype=np.float64):
    elements = st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False
    )
    return npst.arrays(
        dtype=dtype,
        shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=min_side, max_side=max_side),
        elements=elements
    )


@settings(max_examples=100)
@given(matrices(min_side=2, max_side=8))
def test_lu_reconstruction_permute_l(A):
    try:
        L, U = scipy.linalg.lu(A, permute_l=True)
    except Exception:
        assume(False)

    reconstructed = L @ U

    assert np.allclose(reconstructed, A, rtol=1e-6, atol=1e-8), \
        f"L @ U != A (permute_l=True)\nA:\n{A}\nReconstructed:\n{reconstructed}"
```

**Failing input**: A 7x6 matrix with two non-zero rows at positions 2 and 6.

## Reproducing the Bug

```python
import numpy as np
import scipy.linalg

A = np.array([
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 2., 4.30312538e-308, 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0.]
])

L, U = scipy.linalg.lu(A, permute_l=True)
reconstructed = L @ U

print(f"||A - L @ U||: {np.linalg.norm(reconstructed - A)}")
print(f"Expected: ~0.0")
print(f"Actual: {np.linalg.norm(reconstructed - A)}")
```

Output:
```
||A - L @ U||: 1.4142135623730951
Expected: ~0.0
Actual: 1.4142135623730951
```

## Why This Is A Bug

The `scipy.linalg.lu` documentation explicitly states:

> If `permute_l` is set to `True` then `L` is returned already permuted and hence satisfying `A = L @ U`.

The documentation also includes an example:
```python
>>> PL, U = lu(A, permute_l=True)
>>> np.allclose(A, PL @ U)
True
```

This contract is violated for the input matrix above. The reconstruction error of 1.414... is significant and indicates that the permutation is not being applied correctly for certain rectangular matrices.

Specifically:
- Original A[2] = [0, 0, 2, 4.3e-308, 0, 0]
- Original A[6] = [0, 0, 1, 0, 0, 0]

But the reconstruction gives:
- (L @ U)[2] = [0, 0, 2, 4.3e-308, 0, 0] ✓
- (L @ U)[3] = [0, 0, 1, 2.15e-308, 0, 0] ✗ (appears at wrong position)
- (L @ U)[6] = [0, 0, 0, 0, 0, 0] ✗ (should be [0, 0, 1, 0, 0, 0])

## Fix

The issue appears to be in how the permutation is applied when `permute_l=True`. The current implementation computes `L_permuted = P @ L` where P is the permutation matrix, but this does not correctly handle all rectangular matrix cases.

The correct fix would need to ensure that when `permute_l=True`, the returned L and U matrices always satisfy `A = L @ U` as documented. This may require a different approach to how the permutation is incorporated into the L matrix for rectangular matrices.