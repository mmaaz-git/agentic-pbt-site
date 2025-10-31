# Bug Report: scipy.signal.ss2tf Incorrect Polynomial Degree for Pure Feedthrough Systems

**Target**: `scipy.signal.ss2tf`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.signal.ss2tf` incorrectly increases the polynomial degree when converting pure feedthrough state-space systems (where B and C matrices contain only zeros) back to transfer function form, breaking the round-trip property for constant transfer functions.

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

    assert len(b_recon_norm) == len(b_norm), f"Length mismatch: reconstructed {len(b_recon_norm)} vs original {len(b_norm)}"
    np.testing.assert_allclose(b_recon_norm, b_norm, rtol=1e-4, atol=1e-6)

if __name__ == "__main__":
    test_tf2ss_ss2tf_roundtrip()
```

<details>

<summary>
**Failing input**: `b=[1.0], a=[1.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 34, in <module>
    test_tf2ss_ss2tf_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 6, in test_tf2ss_ss2tf_roundtrip
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 30, in test_tf2ss_ss2tf_roundtrip
    assert len(b_recon_norm) == len(b_norm), f"Length mismatch: reconstructed {len(b_recon_norm)} vs original {len(b_norm)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Length mismatch: reconstructed 2 vs original 1
Falsifying example: test_tf2ss_ss2tf_roundtrip(
    b=[1.0],
    a=[1.0],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.signal import tf2ss, ss2tf

# Test case: constant transfer function H(s) = 1
b_orig = np.array([1.0])
a_orig = np.array([1.0])

print("Original transfer function:")
print(f"  Numerator: {b_orig}")
print(f"  Denominator: {a_orig}")
print(f"  This represents H(s) = 1 (a constant)")
print()

# Convert to state-space
A, B, C, D = tf2ss(b_orig, a_orig)
print("State-space representation:")
print(f"  A = {A}")
print(f"  B = {B}")
print(f"  C = {C}")
print(f"  D = {D}")
print()

# Convert back to transfer function
b_result, a_result = ss2tf(A, B, C, D)
print("Reconstructed transfer function:")
print(f"  Numerator: {b_result}")
print(f"  Denominator: {a_result}")
print()

# Analysis
print("Analysis:")
print(f"  Original degree: {len(b_orig) - 1} / {len(a_orig) - 1}")
if b_result.ndim == 2:
    b_result_1d = b_result[0]
else:
    b_result_1d = b_result
print(f"  Reconstructed degree: {len(b_result_1d) - 1} / {len(a_result) - 1}")
print()

print("Issue:")
print("  The original transfer function H(s) = 1 has degree 0.")
print("  After round-trip conversion, we get H(s) = (s + 0)/(s + 0),")
print("  which has degree 1 and a removable singularity at s=0.")
print("  This violates the expected round-trip property.")
```

<details>

<summary>
Output showing degree increase from 0 to 1
</summary>
```
Original transfer function:
  Numerator: [1.]
  Denominator: [1.]
  This represents H(s) = 1 (a constant)

State-space representation:
  A = [[0.]]
  B = [[0.]]
  C = [[0.]]
  D = [[1.]]

Reconstructed transfer function:
  Numerator: [[1. 0.]]
  Denominator: [1. 0.]

Analysis:
  Original degree: 0 / 0
  Reconstructed degree: 1 / 1

Issue:
  The original transfer function H(s) = 1 has degree 0.
  After round-trip conversion, we get H(s) = (s + 0)/(s + 0),
  which has degree 1 and a removable singularity at s=0.
  This violates the expected round-trip property.
```
</details>

## Why This Is A Bug

For a constant transfer function H(s) = 1, the `tf2ss` conversion correctly produces a pure feedthrough system with A=[[0.]], B=[[0.]], C=[[0.]], and D=[[1.]]. However, `ss2tf` incorrectly converts this back to H(s) = (s + 0)/(s + 0) instead of H(s) = 1/1.

This violates the mathematical definition of state-space to transfer function conversion. According to control theory, the transfer function is given by G(s) = C(sI-A)^{-1}B + D. When B and C are zero matrices, the first term vanishes, leaving only G(s) = D, which should be a constant. Instead, the implementation computes the characteristic polynomial of A, resulting in spurious polynomial terms.

The bug produces a transfer function with:
1. Degree 1 instead of degree 0 (incorrect polynomial representation)
2. A removable singularity at s=0 (mathematically equivalent but structurally incorrect)
3. Violation of the round-trip property that tf2ss → ss2tf should preserve coefficients

## Relevant Context

The root cause is in `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/signal/_lti_conversion.py` at lines 269-280. The code checks if B and C are empty matrices (`B.size == 0` and `C.size == 0`) but fails to check if they contain only zeros. When B=[[0.]] and C=[[0.]], the matrices have size 1 but represent a pure feedthrough system that should return just the D matrix with denominator [1.0].

The faulty logic at line 280 computes:
```python
num[k] = poly(A - dot(B, Ck)) + (D[k] - 1) * den
```
For zero B and C matrices, this becomes `poly(A) + 0`, unnecessarily adding the characteristic polynomial of A to the numerator.

Similar functionality in MATLAB's `ss2tf` handles this case correctly by returning the constant transfer function without spurious polynomial terms.

## Proposed Fix

```diff
--- a/scipy/signal/_lti_conversion.py
+++ b/scipy/signal/_lti_conversion.py
@@ -266,7 +266,7 @@ def ss2tf(A, B, C, D, input=0):
     except ValueError:
         den = 1

-    if (B.size == 0) and (C.size == 0):
+    if ((B.size == 0) and (C.size == 0)) or (np.all(B == 0) and np.all(C == 0)):
         num = np.ravel(D)
         if (D.size == 0) and (A.size == 0):
             den = []
```