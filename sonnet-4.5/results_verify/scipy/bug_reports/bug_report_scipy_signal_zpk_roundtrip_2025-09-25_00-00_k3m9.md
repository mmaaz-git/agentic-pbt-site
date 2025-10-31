# Bug Report: scipy.signal Repeated Poles Precision Loss

**Target**: `scipy.signal.zpk2tf` and `scipy.signal.tf2zpk`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The round-trip conversion zpk2tf → tf2zpk loses significant precision when the original zero-pole-gain representation contains repeated poles or zeros. For 4 repeated poles at s=-5, the error can exceed 0.001, which is 0.02% relative error.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
import scipy.signal as signal

@st.composite
def repeated_poles_zpk(draw):
    pole_value = draw(st.floats(min_value=-5, max_value=-0.1, allow_nan=False, allow_infinity=False))
    n_repeats = draw(st.integers(min_value=2, max_value=5))

    poles = np.full(n_repeats, pole_value, dtype=complex)
    zeros = np.array([], dtype=complex)
    gain = 1.0

    return zeros, poles, gain

@settings(max_examples=50)
@given(repeated_poles_zpk())
def test_repeated_poles_roundtrip(zpk_data):
    z1, p1, k1 = zpk_data

    num, den = signal.zpk2tf(z1, p1, k1)
    z2, p2, k2 = signal.tf2zpk(num, den)

    p1_sorted = np.sort(p1.real)
    p2_sorted = np.sort(p2.real)

    max_error = np.max(np.abs(p1_sorted - p2_sorted))

    assert max_error < 1e-3, \
        f"Repeated poles lost precision:\nOriginal: {p1_sorted}\nRecovered: {p2_sorted}\nMax error: {max_error}"
```

**Failing input**: `zpk_data=(array([], dtype=complex128), array([-5.+0.j, -5.+0.j, -5.+0.j, -5.+0.j]), 1.0)`

## Reproducing the Bug

```python
import numpy as np
import scipy.signal as signal

z = np.array([], dtype=complex)
p = np.array([-5.+0.j, -5.+0.j, -5.+0.j, -5.+0.j])
k = 1.0

num, den = signal.zpk2tf(z, p, k)
z2, p2, k2 = signal.tf2zpk(num, den)

p_sorted = np.sort(p.real)
p2_sorted = np.sort(p2.real)

max_error = np.max(np.abs(p_sorted - p2_sorted))

print(f"Original poles: {p_sorted}")
print(f"Recovered poles: {p2_sorted}")
print(f"Maximum error: {max_error:.6e}")
```

Output:
```
Original poles: [-5. -5. -5. -5.]
Recovered poles: [-5.00103355 -4.99999988 -4.99999988 -4.99896668]
Maximum error: 1.033553e-03
```

## Why This Is A Bug

1. **Mathematical property violated**: The round-trip conversion zpk2tf(zpk2tf(...)) should preserve the pole locations to high precision, but repeated poles introduce numerical instability that causes errors exceeding 0.1%.

2. **Practical impact**: Many real-world filters have repeated poles:
   - Butterworth filters have multiple poles at the same location
   - Bessel filters can have closely spaced or repeated poles
   - Higher-order filters commonly have pole multiplicity > 1

3. **Undocumented limitation**: The documentation for `tf2zpk` and `zpk2tf` does not warn users about precision loss with repeated roots.

4. **Root cause**: The conversion chain involves:
   - `zpk2tf` computes polynomial coefficients from roots: (s-p)^n
   - `tf2zpk` finds roots of the polynomial using numpy's polynomial root finder
   - Finding repeated roots from polynomial coefficients is numerically unstable

## Fix

The fundamental issue is that polynomial root finding is ill-conditioned for repeated roots. Possible solutions:

1. **Document the limitation**: Add a warning to both `tf2zpk` and `zpk2tf` documentation that repeated poles/zeros may lose precision through round-trip conversion.

2. **Use a more stable algorithm**: Consider using algorithms specifically designed for finding multiple roots, such as:
   - Root polishing with Newton's method knowing the multiplicity
   - Using the companion matrix eigenvalue approach with better conditioning
   - Deflation techniques that account for multiplicity

3. **Detect and warn**: Add detection of repeated roots and emit a warning when precision loss is likely.

Since this is a fundamental numerical analysis issue, the most practical fix is option 1 (documentation) combined with option 3 (warnings). A proper algorithmic fix would require significant research and testing.

### Documentation patch suggestion

```diff
diff --git a/scipy/signal/_filter_design.py b/scipy/signal/_filter_design.py
index 1234567..89abcdef 100644
--- a/scipy/signal/_filter_design.py
+++ b/scipy/signal/_filter_design.py
@@ -1175,6 +1175,11 @@ def tf2zpk(b, a):
     Notes
     -----
+    .. warning::
+        Finding poles and zeros from polynomial coefficients is numerically
+        unstable for repeated roots. Round-trip conversions zpk2tf -> tf2zpk
+        may lose precision when the original representation has repeated poles
+        or zeros.
+
     If some values of `b` are too close to 0, they are removed. In that case,
     a BadCoefficients warning is emitted.
```