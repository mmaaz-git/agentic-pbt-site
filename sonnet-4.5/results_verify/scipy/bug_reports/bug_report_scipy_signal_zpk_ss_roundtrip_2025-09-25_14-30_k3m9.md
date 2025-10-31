# Bug Report: scipy.signal zpk2ss/ss2zpk Round-Trip Failure

**Target**: `scipy.signal.zpk2ss` and `scipy.signal.ss2zpk`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The round-trip conversion zpk → ss → zpk does not preserve the zeros correctly when the system has fewer zeros than poles. The function `ss2tf` returns a numerator polynomial with unnecessary leading zeros, causing `tf2zpk` to compute spurious zeros with extremely large magnitudes.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from scipy import signal

@settings(max_examples=100)
@given(
    z=st.lists(st.complex_numbers(allow_nan=False, allow_infinity=False,
                                   max_magnitude=1e4), min_size=0, max_size=6),
    p=st.lists(st.complex_numbers(allow_nan=False, allow_infinity=False,
                                   max_magnitude=1e4), min_size=1, max_size=6),
    k=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e4, max_value=1e4)
)
def test_zpk_ss_round_trip(z, p, k):
    z_arr = np.array(z)
    p_arr = np.array(p)

    A, B, C, D = signal.zpk2ss(z_arr, p_arr, k)
    z2, p2, k2 = signal.ss2zpk(A, B, C, D)

    assert np.allclose(np.sort_complex(z_arr), np.sort_complex(z2), rtol=1e-4, atol=1e-6)
    assert np.allclose(np.sort_complex(p_arr), np.sort_complex(p2), rtol=1e-4, atol=1e-6)
```

**Failing input**: `z=[0j], p=[0j, 0j, 0j, 0j], k=32.0`

## Reproducing the Bug

```python
import numpy as np
from scipy import signal

z = [0]
p = [0, 0, 0, 0]
k = 32

A, B, C, D = signal.zpk2ss(z, p, k)
z2, p2, k2 = signal.ss2zpk(A, B, C, D)

print(f"Expected zeros: {z}")
print(f"Actual zeros:   {z2}")
print(f"Expected gain:  {k}")
print(f"Actual gain:    {k2}")
```

**Output:**
```
Expected zeros: [0]
Actual zeros:   [-3.00239975e+15  0.00000000e+00]
Expected gain:  32
Actual gain:    1.0658141036401503e-14
```

## Why This Is A Bug

The round-trip conversion should preserve the transfer function representation. The system H(s) = 32s / s^4 has exactly one zero at s=0, but after the round-trip, we get two zeros (one at 0, and one spurious zero at -3e15). Additionally, the gain is completely wrong (should be 32, got 1.07e-14).

**Root cause**: The `ss2tf` function (in `_lti_conversion.py`, line 277) creates the numerator array with length `num_states + 1`, which equals the denominator length. For systems where the number of zeros is less than the number of poles, this creates unnecessary leading zeros in the numerator polynomial. When `tf2zpk` computes the roots of this padded numerator, it finds spurious zeros with extremely large magnitudes due to numerical instability.

Example:
- Input to `tf2ss`: `b=[32, 0]`, `a=[1, 0, 0, 0, 0]`
- Output from `ss2tf`: `b=[[0, -3.55e-15, 1.07e-14, 32, 0]]`, `a=[1, 0, 0, 0, 0]`

The leading zeros in `b` should be trimmed.

## Fix

The issue is in `ss2tf` function at `scipy/signal/_lti_conversion.py:277`. The numerator should be trimmed to remove leading zeros before returning:

```diff
--- a/scipy/signal/_lti_conversion.py
+++ b/scipy/signal/_lti_conversion.py
@@ -275,7 +275,13 @@ def ss2tf(A, B, C, D, input=0):
     num_states = A.shape[0]
     type_test = A[:, 0] + B[:, 0] + C[0, :] + D + 0.0
     num = np.empty((nout, num_states + 1), type_test.dtype)
     for k in range(nout):
         Ck = atleast_2d(C[k, :])
         num[k] = poly(A - dot(B, Ck)) + (D[k] - 1) * den

+    # Trim leading zeros from numerator to match actual system order
+    for k in range(nout):
+        num[k] = np.trim_zeros(num[k], 'f')
+        if num[k].size == 0:
+            num[k] = np.array([0])
+
     return num, den
```

However, this might break other parts of the code that expect `num` and `den` to have the same length. A more conservative fix would be to trim the zeros in `zpk2ss`/`ss2zpk` or in `tf2zpk` itself, or document this as a known limitation.