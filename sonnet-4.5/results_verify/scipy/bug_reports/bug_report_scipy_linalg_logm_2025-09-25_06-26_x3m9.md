# Bug Report: scipy.linalg.logm Round-Trip Property Violation

**Target**: `scipy.linalg.logm` and `scipy.linalg.expm`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `scipy.linalg.logm` function violates its documented property `expm(logm(A)) == A` for well-conditioned matrices containing very small but valid floating-point values (≤ 1e-25). The function returns catastrophically inaccurate results (errors of magnitude 1e+9 to 1e+117) instead of either handling the values correctly or raising an exception.

## Property-Based Test

```python
from hypothesis import given, settings
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
def test_logm_expm_round_trip(A):
    logA = scipy.linalg.logm(A)
    if np.any(np.isnan(logA)) or np.any(np.isinf(logA)):
        return
    result = scipy.linalg.expm(logA)
    assert np.allclose(result, A, rtol=1e-4, atol=1e-6)
```

**Failing input**:
```python
A = np.array([[-1.0, -1e-50],
               [ 1.0, -1.0]])
```

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

**Output:**
```
Original matrix A:
[[-1.e+00 -1.e-50]
 [ 1.e+00 -1.e+00]]
Determinant: 1.000000
Condition number: 2.62

expm(logm(A)):
[[-1.00000000e+000  0.00000000e+000]
 [ 1.52587891e+009 -1.00000000e+000]]

Expected (original A):
[[-1.e+00 -1.e-50]
 [ 1.e+00 -1.e+00]]

Error:
||expm(logm(A)) - A|| = 1.53e+09
```

## Why This Is A Bug

1. **Documentation violation**: The `scipy.linalg.logm` docstring explicitly states: "The matrix logarithm is the inverse of expm: expm(logm(`A`)) == `A`"

2. **Well-conditioned input**: The failing matrix is NOT ill-conditioned:
   - Determinant = 1.0 (not singular)
   - Condition number = 2.62 (well-conditioned)
   - All eigenvalues are reasonable (≈ -1)

3. **Catastrophic error**: The result element at position [1,0] is 1.53e+9 instead of 1.0 - a relative error exceeding 1e+9.

4. **Silent failure**: While a warning is emitted, the function still returns a result. Users relying on the documented property without checking warnings will get silently corrupted results.

5. **Threshold analysis**: The bug appears for matrix elements with magnitude between 1e-20 and 1e-25:
   - 1e-20: Error = 3.28e-06 (marginal)
   - 1e-25: Error = 3.55e-04 (significant)
   - 1e-50: Error = 1.53e+09 (catastrophic)

## Fix

The issue is in the numerical stability of the matrix logarithm algorithm when dealing with very small matrix elements. Potential fixes:

1. **Improved algorithm**: Use a more numerically stable algorithm that handles extreme element magnitudes
2. **Input normalization**: Scale the matrix before computation and rescale the result
3. **Raise exception**: When the error estimate is catastrophically large, raise an exception instead of returning an inaccurate result
4. **Document limitations**: If very small values cannot be supported, document this restriction

A minimal defensive fix would be to raise an exception when the estimated error is large:

```diff
--- a/scipy/linalg/_matfuncs.py
+++ b/scipy/linalg/_matfuncs.py
@@ -xxx,x +xxx,x @@ def logm(A, disp=_NoValue):
         if disp:
             if errtol > 1e-5:
-                warnings.warn("logm result may be inaccurate, approximate err = %s" % errtol)
+                raise np.linalg.LinAlgError(
+                    f"logm result is inaccurate (err = {errtol:.2e}). "
+                    "The matrix may contain extremely small values that cause numerical instability."
+                )
```

However, the better fix would be to improve the numerical stability of the underlying algorithm.