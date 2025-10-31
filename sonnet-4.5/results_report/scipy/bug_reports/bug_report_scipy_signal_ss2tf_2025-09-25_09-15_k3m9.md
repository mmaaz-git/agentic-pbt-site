# Bug Report: scipy.signal.ss2tf Incorrect Polynomial Degree for Pure Feedthrough Systems

**Target**: `scipy.signal.ss2tf`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`ss2tf` incorrectly increases the polynomial degree when converting pure feedthrough state-space systems (where B and C matrices are all zeros) back to transfer function form. This breaks the round-trip property tf2ss → ss2tf for constant transfer functions.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import numpy as np
import scipy.signal as signal

@settings(max_examples=500)
@given(
    b=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e4, max_value=1e4), min_size=1, max_size=5),
    a=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e4, max_value=1e4), min_size=1, max_size=5)
)
def test_tf2ss_ss2tf_roundtrip(b, a):
    b_arr = np.array(b)
    a_arr = np.array(a)
    assume(np.abs(a_arr[0]) > 1e-10)
    assume(np.abs(b_arr[0]) > 1e-10)
    assume(len(a_arr) >= len(b_arr))

    A, B, C, D = signal.tf2ss(b_arr, a_arr)
    b_reconstructed, a_reconstructed = signal.ss2tf(A, B, C, D)

    # Normalize and compare
    b_norm = b_arr / b_arr[0]
    a_norm = a_arr / a_arr[0]

    if b_reconstructed.ndim == 2:
        b_reconstructed = b_reconstructed[0]

    b_recon_norm = b_reconstructed / b_reconstructed[0]
    a_recon_norm = a_reconstructed / a_reconstructed[0]

    assert len(b_recon_norm) == len(b_norm)
    np.testing.assert_allclose(b_recon_norm, b_norm, rtol=1e-4, atol=1e-6)
```

**Failing input**: `b=[1.0], a=[1.0]`

## Reproducing the Bug

```python
import numpy as np
from scipy.signal import tf2ss, ss2tf

b_orig = np.array([1.0])
a_orig = np.array([1.0])

A, B, C, D = tf2ss(b_orig, a_orig)
print(f"State-space: A={A}, B={B}, C={C}, D={D}")

b_result, a_result = ss2tf(A, B, C, D)
print(f"Original: H(s) = {b_orig}/{a_orig} (degree 0)")
print(f"Result: H(s) = {b_result}/{a_result} (degree 1)")
```

Output:
```
State-space: A=[[0.]], B=[[0.]], C=[[0.]], D=[[1.]]
Original: H(s) = [1.]/[1.] (degree 0)
Result: H(s) = [[1. 0.]]/[1. 0.] (degree 1)
```

## Why This Is A Bug

For a constant transfer function H(s) = 1, the tf2ss conversion correctly produces a pure feedthrough system (A=0, B=0, C=0, D=1). However, ss2tf incorrectly converts this back to H(s) = (s + 0)/(s + 0), which:

1. Has degree 1 instead of degree 0
2. Introduces a removable singularity at s=0
3. Violates the round-trip property: tf2ss(b, a) → ss2tf should preserve the polynomial coefficients

The root cause is in `_lti_conversion.py` lines 269-280. The code checks for empty B and C matrices (`B.size == 0`), but not for zero-valued matrices. When B and C contain only zeros, the system has no state dynamics and should return a constant transfer function, but instead it computes `poly(A)` which adds spurious polynomial terms.

## Fix

```diff
--- a/scipy/signal/_lti_conversion.py
+++ b/scipy/signal/_lti_conversion.py
@@ -266,8 +266,9 @@ def ss2tf(A, B, C, D, input=0):
     except ValueError:
         den = 1

-    if (B.size == 0) and (C.size == 0):
+    if (B.size == 0 and C.size == 0) or (np.all(B == 0) and np.all(C == 0)):
         num = np.ravel(D)
+        den = np.array([1.0])
         if (D.size == 0) and (A.size == 0):
             den = []
         return num, den
```

This fix detects pure feedthrough systems (where B and C are all zeros) and correctly returns just the D matrix with a denominator of [1.0], without computing the characteristic polynomial of A.