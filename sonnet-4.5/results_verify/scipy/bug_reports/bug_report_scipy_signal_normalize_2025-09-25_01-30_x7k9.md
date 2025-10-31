# Bug Report: scipy.signal.normalize Leading Zero Stripping

**Target**: `scipy.signal.normalize`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `scipy.signal.normalize()` function incorrectly strips leading zeros from the numerator coefficients of a transfer function, which silently changes the filter's behavior. This affects multiple conversion functions including `tf2zpk()`, `zpk2tf()`, `tf2sos()`, and `sos2tf()`, causing them to violate round-trip properties.

## Property-Based Test

```python
import numpy as np
import scipy.signal
from hypothesis import given, strategies as st, assume, settings


@settings(max_examples=100)
@given(
    b_coeffs=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=1, max_size=10),
    a_coeffs=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=1, max_size=10)
)
def test_tf2zpk_zpk2tf_roundtrip(b_coeffs, a_coeffs):
    b = np.array(b_coeffs)
    a = np.array(a_coeffs)

    assume(abs(a[0]) > 1e-10)

    b = b / a[0]
    a = a / a[0]

    z, p, k = scipy.signal.tf2zpk(b, a)
    b_reconstructed, a_reconstructed = scipy.signal.zpk2tf(z, p, k)

    # Check that filter behavior is preserved
    impulse = np.array([1, 0, 0, 0, 0])
    y_orig = scipy.signal.lfilter(b, a, impulse)
    y_recon = scipy.signal.lfilter(b_reconstructed, a_reconstructed, impulse)

    np.testing.assert_allclose(y_orig, y_recon, rtol=1e-8, atol=1e-10)
```

**Failing input**: `b=[0.0, 1.0], a=[1.0]`

## Reproducing the Bug

```python
import numpy as np
import scipy.signal

b = np.array([0.0, 1.0])
a = np.array([1.0])

print("Original filter:")
print(f"  b = {b}, a = {a}")

impulse = np.array([1, 0, 0, 0, 0])
y_original = scipy.signal.lfilter(b, a, impulse)
print(f"  Impulse response: {y_original}")
print(f"  This represents H(z) = z^(-1), a delay of 1 sample")

b_norm, a_norm = scipy.signal.normalize(b, a)
print(f"\nAfter normalize():")
print(f"  b = {b_norm}, a = {a_norm}")

y_normalized = scipy.signal.lfilter(b_norm, a_norm, impulse)
print(f"  Impulse response: {y_normalized}")
print(f"  This represents H(z) = 1, no delay")

print(f"\nFilter behavior preserved: {np.allclose(y_original, y_normalized)}")
```

**Output:**
```
Original filter:
  b = [0. 1.], a = [1.]
  Impulse response: [0. 1. 0. 0. 0.]
  This represents H(z) = z^(-1), a delay of 1 sample

After normalize():
  b = [1.], a = [1. 0.]
  Impulse response: [1. 0. 0. 0. 0.]
  This represents H(z) = 1, no delay

Filter behavior preserved: False
```

## Why This Is A Bug

The `normalize()` function strips leading zeros from the numerator coefficients, which changes the order and behavior of the transfer function.

In the example above:
- Original: `b=[0, 1], a=[1]` represents the difference equation `y[n] = 0*x[n] + 1*x[n-1] = x[n-1]` (delay by 1)
- After normalize: `b=[1], a=[1]` represents `y[n] = 1*x[n]` (no delay)

This is a silent data corruption bug because:
1. The function doesn't error or warn that it's changing behavior
2. Users expect `normalize()` to only scale coefficients, not change filter characteristics
3. It causes round-trip conversions (tf→zpk→tf, tf→sos→tf) to fail
4. Leading zeros are mathematically valid in transfer functions

## Fix

The `normalize()` function should not strip leading zeros from the numerator. It should only normalize by dividing all coefficients by `a[0]` to ensure the denominator's leading coefficient is 1.

```diff
--- a/scipy/signal/_filter_design.py
+++ b/scipy/signal/_filter_design.py
@@ -1227,10 +1227,7 @@ def normalize(b, a):
     if len(a) == 0:
         raise ValueError("Denominator of transfer function cannot be empty.")
     b, a = b / a[0], a / a[0]
-    outb = b[np.nonzero(b)[0][0]:] if np.any(b) else b
-    return outb, a
+    return b, a
```

Note: The exact line numbers and implementation may vary, but the fix is to remove the logic that strips leading zeros from the numerator (`b`).