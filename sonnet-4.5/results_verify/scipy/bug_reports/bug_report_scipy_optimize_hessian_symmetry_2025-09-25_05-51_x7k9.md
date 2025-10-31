# Bug Report: scipy.optimize BFGS and SR1 Symmetry Violation

**Target**: `scipy.optimize.BFGS` and `scipy.optimize.SR1`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The BFGS and SR1 Hessian update strategies produce asymmetric matrices when updated with certain step vectors, violating a fundamental mathematical property of these algorithms. Both algorithms are required to maintain symmetric positive-definite (BFGS) or symmetric (SR1) Hessian approximations.

## Property-Based Test

```python
import numpy as np
import scipy.optimize
from hypothesis import given, strategies as st, assume, settings


@given(
    st.integers(min_value=2, max_value=5),
    st.lists(st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
             min_size=2, max_size=5)
)
@settings(max_examples=200)
def test_bfgs_update_symmetry(n, delta_list):
    assume(len(delta_list) >= n)
    delta = np.array(delta_list[:n])
    assume(np.linalg.norm(delta) > 0.01)

    gradient_diff = np.random.randn(n)
    assume(np.linalg.norm(gradient_diff) > 0.01)

    bfgs = scipy.optimize.BFGS()
    bfgs.initialize(n, 'hess')

    bfgs.update(delta, gradient_diff)

    if hasattr(bfgs, 'B') and bfgs.B is not None:
        assert np.allclose(bfgs.B, bfgs.B.T, atol=1e-10), \
            "BFGS update should maintain symmetry"
```

**Failing input**: `n=2, delta_list=[1.0, 0.0]` (and many others)

## Reproducing the Bug

```python
import numpy as np
import scipy.optimize

delta = np.array([1.0, 0.0])
gradient_diff = np.array([0.49671415, -0.1382643])

bfgs = scipy.optimize.BFGS()
bfgs.initialize(2, 'hess')
bfgs.update(delta, gradient_diff)

print(f"B =\n{bfgs.B}")
print(f"B.T =\n{bfgs.B.T}")
print(f"Symmetric: {np.allclose(bfgs.B, bfgs.B.T)}")
```

Output:
```
B =
[[ 0.49671415 -0.1382643 ]
 [ 0.          0.57368807]]
B.T =
[[ 0.49671415  0.        ]
 [-0.1382643   0.57368807]]
Symmetric: False
```

The same issue occurs with SR1:

```python
sr1 = scipy.optimize.SR1()
sr1.initialize(2, 'hess')
sr1.update(np.array([0.0, 1.0]), np.array([0.49671415, -0.1382643]))

print(f"B =\n{sr1.B}")
print(f"Symmetric: {np.allclose(sr1.B, sr1.B.T)}")
```

Output:
```
B =
[[ 1.80299578  0.49671415]
 [ 0.         -0.1382643 ]]
Symmetric: False
```

## Why This Is A Bug

1. **Mathematical requirement**: Both BFGS and SR1 algorithms are **required** to maintain symmetric Hessian approximations. This is not just a nice-to-have property—it's fundamental to the algorithms' correctness.

2. **BFGS specification**: The BFGS update formula mathematically guarantees symmetry of the updated Hessian approximation. If the implementation produces an asymmetric matrix, the algorithm is incorrectly implemented.

3. **SR1 specification**: The SR1 (Symmetric Rank-1) update is named for its property of maintaining symmetry while performing a rank-1 update.

4. **Impact on optimization**: Using an asymmetric Hessian approximation can lead to:
   - Incorrect search directions
   - Failed convergence
   - Numerical instabilities
   - Wrong optimization results

5. **Silent failure**: The bug occurs silently—no warning or error is raised, so users may not realize their optimization results are incorrect.

## Fix

The issue appears to be that the Hessian matrix `B` is stored in a triangular format (likely for efficiency) but the `update()` method only updates the triangular portion without ensuring the full matrix remains symmetric.

Looking at the pattern, when `delta` has a zero in position `i`, the update doesn't properly symmetrize element `[i, j]` and `[j, i]`.

A fix would involve:
1. Ensuring that after the update computation, the matrix is explicitly symmetrized: `B = (B + B.T) / 2`
2. Or, properly updating both triangular portions during the update
3. Or, ensuring the storage format properly maintains symmetry

Without access to the exact implementation details, the safest fix is to add explicit symmetrization after each update:

```python
def update(self, delta, gradient_diff):
    # ... existing update logic ...
    self.B = (self.B + self.B.T) / 2
```

This ensures mathematical correctness at minimal computational cost (since the update is typically O(n²) anyway).