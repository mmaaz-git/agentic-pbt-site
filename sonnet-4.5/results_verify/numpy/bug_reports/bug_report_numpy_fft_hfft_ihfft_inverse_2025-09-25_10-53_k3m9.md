# Bug Report: numpy.fft.hfft/ihfft Inverse Property Violation

**Target**: `numpy.fft.hfft` / `numpy.fft.ihfft`
**Severity**: Medium
**Bug Type**: Contract (or Logic)
**Date**: 2025-09-25

## Summary

The documentation for `numpy.fft.hfft` and `numpy.fft.ihfft` claims that `ihfft(hfft(a, 2*len(a) - 2)) == a` for even-length outputs, but this property does not hold even for simple inputs. The inverse transformation fails to recover the original array.

## Property-Based Test

```python
import numpy as np
import numpy.fft as fft
from hypothesis import given, strategies as st, settings, assume


@st.composite
def complex_arrays(draw, min_size=1, max_size=100):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    real_part = draw(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                                      min_value=-1e6, max_value=1e6),
                             min_size=size, max_size=size))
    imag_part = draw(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                                      min_value=-1e6, max_value=1e6),
                             min_size=size, max_size=size))
    return np.array([complex(r, i) for r, i in zip(real_part, imag_part)])


@given(complex_arrays(min_size=2, max_size=50))
@settings(max_examples=500)
def test_hfft_ihfft_inverse_even(a):
    assume(len(a) >= 2)
    n = 2 * len(a) - 2
    result = fft.ihfft(fft.hfft(a, n))
    assert np.allclose(result, a, rtol=1e-10, atol=1e-12), \
        f"ihfft(hfft(a, 2*len(a)-2)) != a. Max diff: {np.max(np.abs(result - a))}"
```

**Failing input**: `array([0.+0.j, 0.+1.j])`

## Reproducing the Bug

```python
import numpy as np

a = np.array([0.+0.j, 0.+1.j])
n = 2 * len(a) - 2

hfft_result = np.fft.hfft(a, n)
ihfft_result = np.fft.ihfft(hfft_result)

print("Original:", a)
print("Recovered:", ihfft_result)
print("Match:", np.allclose(ihfft_result, a))
```

Expected output based on documentation:
```
Original: [0.+0.j 0.+1.j]
Recovered: [0.+0.j 0.+1.j]
Match: True
```

Actual output:
```
Original: [0.+0.j 0.+1.j]
Recovered: [0.-0.j 0.-0.j]
Match: False
```

## Why This Is A Bug

The documentation for both `hfft` and `ihfft` explicitly states (lines 591-592 in `numpy/fft/_pocketfft.py` and lines 683-684):

> * even: `ihfft(hfft(a, 2*len(a) - 2)) == a`, within roundoff error,
> * odd: `ihfft(hfft(a, 2*len(a) - 1)) == a`, within roundoff error.

This is documented as a mathematical property that should hold. However, the property fails for simple inputs like `[0.+0.j, 0.+1.j]`, where the recovered array `[0.-0.j, 0.-0.j]` has completely lost the imaginary component of the second element.

This represents either:
1. **A logic bug** in the hfft/ihfft implementation that breaks the documented inverse relationship, OR
2. **A contract bug** where the documentation fails to mention that this property only holds for arrays with Hermitian symmetry (a[k] = conj(a[-k]))

In either case, users following the documentation will encounter unexpected behavior where data is lost during the transform pair.

## Fix

The issue requires investigation to determine whether this is an implementation bug or a documentation bug.

**If documentation bug** (inverse property requires Hermitian input):

```diff
--- a/numpy/fft/_pocketfft.py
+++ b/numpy/fft/_pocketfft.py
@@ -586,8 +586,10 @@ def hfft(a, n=None, axis=-1, norm=None, out=None):
     `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the
     opposite case: here the signal has Hermitian symmetry in the time
     domain and is real in the frequency domain. So here it's `hfft` for
-    which you must supply the length of the result if it is to be odd:
+    which you must supply the length of the result if it is to be odd.

+    For inputs with Hermitian symmetry (a[k] = conj(a[-k])):
+
     * even: ``ihfft(hfft(a, 2*len(a) - 2)) == a``, within roundoff error,
     * odd: ``ihfft(hfft(a, 2*len(a) - 1)) == a``, within roundoff error.
```

**If implementation bug**: The fix would require analyzing the mathematical relationship and correcting the transform implementation, which requires deeper investigation of the FFT algorithms.

**Recommendation**: Further investigation needed to determine root cause. The current state violates the principle of least surprise where documented properties don't hold for straightforward inputs.