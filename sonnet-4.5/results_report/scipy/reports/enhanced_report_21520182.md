# Bug Report: scipy.linalg.logm Round-Trip Property Violation for Matrices with Small Elements

**Target**: `scipy.linalg.logm`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `scipy.linalg.logm` function violates its documented round-trip property `expm(logm(A)) == A` for well-conditioned matrices containing very small floating-point values (≤ 1e-25), returning catastrophically inaccurate results with errors exceeding 1e+9.

## Property-Based Test

```python
from hypothesis import given, settings, example
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
import numpy as np
import scipy.linalg


def invertible_matrices(min_size=2, max_size=5):
    return st.integers(min_value=min_size, max_value=max_size).flatmap(
        lambda n: arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(
                min_value=-10,
                max_value=10,
                allow_nan=False,
                allow_infinity=False
            )
        )
    ).filter(lambda A: abs(np.linalg.det(A)) > 1e-10)


@given(invertible_matrices(min_size=2, max_size=4))
@settings(max_examples=50)
@example(np.array([[-1.0, -1e-50], [1.0, -1.0]]))  # Add our specific failing case
def test_logm_expm_round_trip(A):
    logA = scipy.linalg.logm(A)
    if np.any(np.isnan(logA)) or np.any(np.isinf(logA)):
        return
    result = scipy.linalg.expm(logA)
    assert np.allclose(result, A, rtol=1e-4, atol=1e-6), f"Failed for A={A}"

# Run the test and catch the failing example
if __name__ == "__main__":
    try:
        test_logm_expm_round_trip()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
```

<details>

<summary>
**Failing input**: `np.array([[-1.0, -1e-50], [1.0, -1.0]])`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_util.py:1233: RuntimeWarning: logm result may be inaccurate, approximate err = 763475521.8680009
  return f(*arrays, *other_args, **kwargs)
Test failed: Failed for A=[[-1.e+00 -1.e-50]
 [ 1.e+00 -1.e+00]]
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.linalg

A = np.array([[-1.0, -1e-50],
               [ 1.0, -1.0]])

print("Original matrix A:")
print(A)
print(f"Determinant: {np.linalg.det(A):.6f}")
print(f"Condition number: {np.linalg.cond(A):.2f}")

logA = scipy.linalg.logm(A)
result = scipy.linalg.expm(logA)

print("\nexpm(logm(A)):")
print(result)

print("\nExpected (original A):")
print(A)

print("\nError:")
print(f"||expm(logm(A)) - A|| = {np.linalg.norm(result - A):.2e}")
```

<details>

<summary>
RuntimeWarning and catastrophic error in round-trip computation
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_util.py:1233: RuntimeWarning: logm result may be inaccurate, approximate err = 763475521.8680009
  return f(*arrays, *other_args, **kwargs)
Original matrix A:
[[-1.e+00 -1.e-50]
 [ 1.e+00 -1.e+00]]
Determinant: 1.000000
Condition number: 2.62

expm(logm(A)):
[[-1.00000000e+00  0.00000000e+00]
 [ 1.52695104e+09 -1.00000000e+00]]

Expected (original A):
[[-1.e+00 -1.e-50]
 [ 1.e+00 -1.e+00]]

Error:
||expm(logm(A)) - A|| = 1.53e+09
```
</details>

## Why This Is A Bug

The function violates its explicitly documented mathematical property for a well-conditioned, non-singular matrix. The docstring for `scipy.linalg.logm` states: "The matrix logarithm is the inverse of expm: expm(logm(`A`)) == `A`" (line 154 in _matfuncs.py). This property fails catastrophically for the given input:

1. **Well-conditioned input**: The matrix has determinant 1.0 (non-singular) and condition number 2.62 (well-conditioned), making it mathematically well-behaved.

2. **Catastrophic error magnitude**: The element at position [1,0] is 1.52695104e+09 instead of 1.0, representing an error of over 9 orders of magnitude.

3. **Valid floating-point values**: The value 1e-50 is a perfectly valid IEEE 754 float64 value, well above the smallest positive normal float64 (~2.2e-308).

4. **Warning acknowledges the problem**: The function itself warns about an approximate error of 7.6e+8, confirming it knows the result is severely inaccurate.

5. **Silent data corruption risk**: While a warning is emitted, the function still returns a result, potentially causing silent data corruption in downstream calculations when warnings are suppressed.

## Relevant Context

The issue appears to stem from numerical instability in the inverse scaling and squaring algorithm when dealing with matrices containing elements with vastly different scales. The algorithm uses a triangular decomposition and applies logarithms to the diagonal entries. When very small values appear in specific positions, the numerical errors compound during the squaring and scaling steps.

The warning threshold in the code (line 225-227 in _matfuncs.py) checks if `errest >= errtol` where `errtol = 1000*eps` (approximately 2.22e-13). The actual error of 7.6e+8 far exceeds this threshold, yet the function proceeds to return the incorrect result.

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.logm.html
Source code: scipy/linalg/_matfuncs.py and scipy/linalg/_matfuncs_inv_ssq.py

## Proposed Fix

The issue requires improving the numerical stability of the logarithm algorithm for matrices with extreme element scales. A defensive fix would be to raise an exception when catastrophic errors are detected:

```diff
--- a/scipy/linalg/_matfuncs.py
+++ b/scipy/linalg/_matfuncs.py
@@ -223,10 +223,15 @@ def logm(A, disp=_NoValue):
         errest = norm(expm(F)-A, 1) / np.asarray(norm(A, 1), dtype=A.dtype).real[()]
     if disp:
         if not isfinite(errest) or errest >= errtol:
-            message = f"logm result may be inaccurate, approximate err = {errest}"
-            warnings.warn(message, RuntimeWarning, stacklevel=2)
+            if errest > 1e6:  # Catastrophic error threshold
+                raise np.linalg.LinAlgError(
+                    f"logm result is catastrophically inaccurate (err = {errest:.2e}). "
+                    "The matrix may contain extremely small values that cause numerical instability."
+                )
+            else:
+                message = f"logm result may be inaccurate, approximate err = {errest}"
+                warnings.warn(message, RuntimeWarning, stacklevel=2)
         return F
     else:
         return F, errest
```