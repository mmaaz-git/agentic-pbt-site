# Bug Report: scipy.linalg.lu LU Decomposition Incorrect for Rectangular Matrices with Near-Minimum Float64 Values

**Target**: `scipy.linalg.lu`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `scipy.linalg.lu(A, permute_l=True)` is called on certain rectangular matrices containing values near the float64 minimum (around 1e-308), the returned matrices L and U do not satisfy the documented property `A = L @ U`, resulting in an incorrect reconstruction with significant error.

## Property-Based Test

```python
from hypothesis import given, settings, assume, strategies as st, example
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


# The specific failing input from the bug report
failing_matrix = np.array([
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 2., 4.30312538e-308, 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0.]
])


@settings(max_examples=100, deadline=None)
@given(matrices(min_side=2, max_side=8))
@example(failing_matrix)  # Add the specific failing case
def test_lu_reconstruction_permute_l(A):
    try:
        L, U = scipy.linalg.lu(A, permute_l=True)
    except Exception:
        assume(False)

    reconstructed = L @ U

    assert np.allclose(reconstructed, A, rtol=1e-6, atol=1e-8), \
        f"L @ U != A (permute_l=True)\\nA:\\n{A}\\nReconstructed:\\n{reconstructed}"
```

<details>

<summary>
**Failing input**: `7x6 matrix with two non-zero rows containing values near float64 minimum`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/24
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo_failing.py::test_lu_reconstruction_permute_l FAILED                 [100%]

=================================== FAILURES ===================================
_______________________ test_lu_reconstruction_permute_l _______________________
hypo_failing.py:34: in test_lu_reconstruction_permute_l
    @given(matrices(min_side=2, max_side=8))
                   ^^^
/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py:1613: in _raise_to_user
    raise the_error_hypothesis_found
hypo_failing.py:44: in test_lu_reconstruction_permute_l
    assert np.allclose(reconstructed, A, rtol=1e-6, atol=1e-8), \
E   AssertionError: L @ U != A (permute_l=True)\nA:\n[[0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
E       0.00000000e+000 0.00000000e+000]
E      [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
E       0.00000000e+000 0.00000000e+000]
E      [0.00000000e+000 0.00000000e+000 2.00000000e+000 4.30312538e-308
E       0.00000000e+000 0.00000000e+000]
E      [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
E       0.00000000e+000 0.00000000e+000]
E      [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
E       0.00000000e+000 0.00000000e+000]
E      [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
E       0.00000000e+000 0.00000000e+000]
E      [0.00000000e+000 0.00000000e+000 1.00000000e+000 0.00000000e+000
E       0.00000000e+000 0.00000000e+000]]\nReconstructed:\n[[0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
E       0.00000000e+000 0.00000000e+000]
E      [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
E       0.00000000e+000 0.00000000e+000]
E      [0.00000000e+000 0.00000000e+000 2.00000000e+000 4.30312538e-308
E       0.00000000e+000 0.00000000e+000]
E      [0.00000000e+000 0.00000000e+000 1.00000000e+000 2.15156269e-308
E       0.00000000e+000 0.00000000e+000]
E      [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
E       0.00000000e+000 0.00000000e+000]
E      [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
E       0.00000000e+000 0.00000000e+000]
E      [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
E       0.00000000e+000 0.00000000e+000]]
E   assert False
E    +  where False = <function allclose at 0x796e9ff4adb0>(array([[0.00000000e+000, 0.00000000e+000, 0.00000000e+000,\n        0.00000000e+000, 0.00000000e+000, 0.00000000e+000],\n       [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,\n        0.00000000e+000, 0.00000000e+000, 0.00000000e+000],\n       [0.00000000e+000, 0.00000000e+000, 2.00000000e+000,\n        4.30312538e-308, 0.00000000e+000, 0.00000000e+000],\n       [0.00000000e+000, 0.00000000e+000, 1.00000000e+000,\n        2.15156269e-308, 0.00000000e+000, 0.00000000e+000],\n       [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,\n        0.00000000e+000, 0.00000000e+000, 0.00000000e+000],\n       [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,\n        0.00000000e+000, 0.00000000e+000, 0.00000000e+000],\n       [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,\n        0.00000000e+000, 0.00000000e+000, 0.00000000e+000]]), array([[0.00000000e+000, 0.00000000e+000, 0.00000000e+000,\n        0.00000000e+000, 0.00000000e+000, 0.00000000e+000],\n       [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,\n        0.00000000e+000, 0.00000000e+000, 0.00000000e+000],\n       [0.00000000e+000, 0.00000000e+000, 2.00000000e+000,\n        4.30312538e-308, 0.00000000e+000, 0.00000000e+000],\n       [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,\n        0.00000000e+000, 0.00000000e+000, 0.00000000e+000],\n       [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,\n        0.00000000e+000, 0.00000000e+000, 0.00000000e+000],\n       [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,\n        0.00000000e+000, 0.00000000e+000, 0.00000000e+000],\n       [0.00000000e+000, 0.00000000e+000, 1.00000000e+000,\n        0.00000000e+000, 0.00000000e+000, 0.00000000e+000]]), rtol=1e-06, atol=1e-08)
E    +    where <function allclose at 0x796e9ff4adb0> = np.allclose
E   Falsifying explicit example: test_lu_reconstruction_permute_l(
E       A=array([[0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
E               0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
E              [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
E               0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
E              [0.00000000e+000, 0.00000000e+000, 2.00000000e+000,
E               4.30312538e-308, 0.00000000e+000, 0.00000000e+000],
E              [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
E               0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
E              [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
E               0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
E              [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
E               0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
E              [0.00000000e+000, 0.00000000e+000, 1.00000000e+000,
E               0.00000000e+000, 0.00000000e+000, 0.00000000e+000]]),
E   )
=========================== short test summary info ============================
FAILED hypo_failing.py::test_lu_reconstruction_permute_l - AssertionError: L ...
============================== 1 failed in 0.30s ===============================
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.linalg

# Problematic matrix from the bug report
A = np.array([
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 2., 4.30312538e-308, 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0.]
])

print("Original matrix A:")
print(A)
print()

# Test with permute_l=True
print("Testing with permute_l=True:")
L, U = scipy.linalg.lu(A, permute_l=True)
reconstructed = L @ U

print(f"L shape: {L.shape}")
print(f"U shape: {U.shape}")
print()

print("Reconstructed matrix (L @ U):")
print(reconstructed)
print()

print("Difference (A - L @ U):")
print(A - reconstructed)
print()

print(f"||A - L @ U||: {np.linalg.norm(A - reconstructed)}")
print(f"Expected: ~0.0")
print()

# Check if reconstruction matches original
is_close = np.allclose(A, reconstructed, rtol=1e-6, atol=1e-8)
print(f"np.allclose(A, L @ U): {is_close}")
print()

# Show specific problematic rows
print("Comparing specific rows:")
print(f"Original A[2]: {A[2]}")
print(f"Reconstructed[2]: {reconstructed[2]}")
print()
print(f"Original A[6]: {A[6]}")
print(f"Reconstructed[6]: {reconstructed[6]}")
print()

# Also test with permute_l=False for comparison
print("Testing with permute_l=False:")
P, L2, U2 = scipy.linalg.lu(A, permute_l=False)
reconstructed2 = P @ L2 @ U2
print(f"||A - P @ L @ U||: {np.linalg.norm(A - reconstructed2)}")
is_close2 = np.allclose(A, reconstructed2, rtol=1e-6, atol=1e-8)
print(f"np.allclose(A, P @ L @ U): {is_close2}")
```

<details>

<summary>
Output shows incorrect reconstruction with significant error
</summary>
```
Original matrix A:
[[0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000 2.00000000e+000 4.30312538e-308
  0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000 1.00000000e+000 0.00000000e+000
  0.00000000e+000 0.00000000e+000]]

Testing with permute_l=True:
L shape: (7, 6)
U shape: (6, 6)

Reconstructed matrix (L @ U):
[[0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000 2.00000000e+000 4.30312538e-308
  0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000 1.00000000e+000 2.15156269e-308
  0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
  0.00000000e+000 0.00000000e+000]]

Difference (A - L @ U):
[[ 0.00000000e+000  0.00000000e+000  0.00000000e+000  0.00000000e+000
   0.00000000e+000  0.00000000e+000]
 [ 0.00000000e+000  0.00000000e+000  0.00000000e+000  0.00000000e+000
   0.00000000e+000  0.00000000e+000]
 [ 0.00000000e+000  0.00000000e+000  0.00000000e+000  0.00000000e+000
   0.00000000e+000  0.00000000e+000]
 [ 0.00000000e+000  0.00000000e+000 -1.00000000e+000 -2.15156269e-308
   0.00000000e+000  0.00000000e+000]
 [ 0.00000000e+000  0.00000000e+000  0.00000000e+000  0.00000000e+000
   0.00000000e+000  0.00000000e+000]
 [ 0.00000000e+000  0.00000000e+000  0.00000000e+000  0.00000000e+000
   0.00000000e+000  0.00000000e+000]
 [ 0.00000000e+000  0.00000000e+000  1.00000000e+000  0.00000000e+000
   0.00000000e+000  0.00000000e+000]]

||A - L @ U||: 1.4142135623730951
Expected: ~0.0

np.allclose(A, L @ U): False

Comparing specific rows:
Original A[2]: [0.00000000e+000 0.00000000e+000 2.00000000e+000 4.30312538e-308
 0.00000000e+000 0.00000000e+000]
Reconstructed[2]: [0.00000000e+000 0.00000000e+000 2.00000000e+000 4.30312538e-308
 0.00000000e+000 0.00000000e+000]

Original A[6]: [0. 0. 1. 0. 0. 0.]
Reconstructed[6]: [0. 0. 0. 0. 0. 0.]

Testing with permute_l=False:
||A - P @ L @ U||: 1.4142135623730951
np.allclose(A, P @ L @ U): False
```
</details>

## Why This Is A Bug

The `scipy.linalg.lu` documentation explicitly states that when `permute_l` is set to `True`, the function returns L already permuted such that `A = L @ U`. This is a fundamental mathematical property of LU decomposition that must hold for any valid input matrix.

The documentation includes this explicit promise:
> If `permute_l` is set to ``True`` then ``L`` is returned already permuted and hence satisfying ``A = L @ U``.

And provides this example:
```python
>>> PL, U = lu(A, permute_l=True)
>>> np.allclose(A, PL @ U)
True
```

This contract is violated for the input matrix above. The reconstruction error of 1.414 (which is sqrt(2)) indicates a significant structural error, not a minor numerical precision issue. Specifically:

- The row that should appear at position 6 (`[0, 0, 1, 0, 0, 0]`) appears at position 3 with modified values (`[0, 0, 1, 2.15e-308, 0, 0]`)
- The row at position 6 in the reconstruction is all zeros instead of the expected `[0, 0, 1, 0, 0, 0]`
- This represents a complete misplacement of matrix rows, not a rounding error

## Relevant Context

The bug appears to be related to numerical issues when handling values near the float64 minimum positive normal value (approximately 2.225e-308). The value 4.30312538e-308 is very close to this threshold, which may be causing issues in the pivoting logic of the LU decomposition algorithm.

The bug affects both `permute_l=True` and `permute_l=False` modes, suggesting the issue is in the core decomposition logic rather than just the permutation application. The implementation uses LAPACK's `*GETRF` routines through `scipy.linalg.lapack.get_lapack_funcs`.

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu.html

## Proposed Fix

The issue likely stems from numerical comparison problems when determining pivot elements for values near the float64 minimum. The algorithm may be incorrectly treating the tiny value 4.30312538e-308 as effectively zero during pivoting decisions, leading to incorrect row swaps.

A high-level fix approach would be:

1. Implement proper handling of near-minimum float64 values in the pivoting logic
2. Use a more robust numerical comparison threshold that accounts for the scale of values being compared
3. Ensure the permutation logic correctly handles cases where pivoting decisions involve extremely small but non-zero values

The fix would likely need to be implemented in the Cython layer (`_decomp_lu_cython.lu_dispatcher`) or in the underlying LAPACK routines' usage to ensure proper handling of these edge cases.