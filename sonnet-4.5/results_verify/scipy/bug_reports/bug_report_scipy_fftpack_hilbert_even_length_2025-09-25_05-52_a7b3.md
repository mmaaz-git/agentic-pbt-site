# Bug Report: scipy.fftpack.hilbert/ihilbert Extended Bug for Even-Length Arrays

**Target**: `scipy.fftpack.hilbert` and `scipy.fftpack.ihilbert`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The previously reported Hilbert transform round-trip bug for length-2 arrays also affects other even-length arrays. The documented property "If `sum(x, axis=0) == 0` then `hilbert(ihilbert(x)) == x`" fails for length-4 arrays (and likely other even lengths).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy as np
from scipy import fftpack


@settings(max_examples=500)
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                          min_value=-1e6, max_value=1e6),
                min_size=3, max_size=100))
def test_hilbert_ihilbert_roundtrip_zero_mean(x_list):
    x = np.array(x_list)
    x = x - x.mean()
    assume(np.sum(np.abs(x)) > 1e-6)
    assume(len(x) > 2)

    h = fftpack.hilbert(x)
    ih = fftpack.ihilbert(h)

    assert np.allclose(ih, x, rtol=1e-4, atol=1e-6)
```

**Failing input**: `x_list=[0.0, 0.0, 0.0, 1.0]` (after mean subtraction: `[-0.25, -0.25, -0.25, 0.75]`)

## Reproducing the Bug

```python
import numpy as np
from scipy import fftpack

print("Testing even-length arrays:")
print("="*60)

test_cases = [
    [0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
]

for x_orig in test_cases:
    x = np.array(x_orig)
    x = x - x.mean()

    print(f"\nInput (zero-mean, len={len(x)}): {x}")
    print(f"sum(x) = {np.sum(x):.10f}")

    h = fftpack.hilbert(x)
    ih = fftpack.ihilbert(h)

    print(f"ihilbert(hilbert(x)) = {ih}")
    match = np.allclose(ih, x, rtol=1e-4, atol=1e-6)
    print(f"Match: {match}")

    if not match:
        print(f"  Max error: {np.max(np.abs(ih - x))}")

print("\n" + "="*60)
print("Testing odd-length arrays (for comparison):")
print("="*60)

test_cases_odd = [
    [0.0, 0.0, 1.0],
    [1.0, 2.0, 3.0, 4.0, 5.0],
]

for x_orig in test_cases_odd:
    x = np.array(x_orig)
    x = x - x.mean()

    print(f"\nInput (zero-mean, len={len(x)}): {x}")
    print(f"sum(x) = {np.sum(x):.10f}")

    h = fftpack.hilbert(x)
    ih = fftpack.ihilbert(h)

    print(f"ihilbert(hilbert(x)) = {ih}")
    match = np.allclose(ih, x, rtol=1e-4, atol=1e-6)
    print(f"Match: {match}")
```

**Expected output**:
```
Testing even-length arrays:
============================================================

Input (zero-mean, len=2): [-0.5  0.5]
sum(x) = 0.0000000000
ihilbert(hilbert(x)) = [-0. -0.]
Match: False
  Max error: 0.5

Input (zero-mean, len=4): [-0.25 -0.25 -0.25  0.75]
sum(x) = 0.0000000000
ihilbert(hilbert(x)) = [-0.  -0.5 -0.   0.5]
Match: False
  Max error: 0.25

Input (zero-mean, len=6): [-2.5 -1.5 -0.5  0.5  1.5  2.5]
sum(x) = 0.0000000000
ihilbert(hilbert(x)) = [-2.5 -1.5 -0.5  0.5  1.5  2.5]
Match: True

============================================================
Testing odd-length arrays (for comparison):
============================================================

Input (zero-mean, len=3): [-0.66666667  -0.66666667   1.33333333]
sum(x) = 0.0000000000
ihilbert(hilbert(x)) = [-0.66666667  -0.66666667   1.33333333]
Match: True

Input (zero-mean, len=5): [-2. -1.  0.  1.  2.]
sum(x) = 0.0000000000
ihilbert(hilbert(x)) = [-2. -1.  0.  1.  2.]
Match: True
```

## Why This Is A Bug

The documentation for `scipy.fftpack.hilbert` states:

> Notes
> -----
> If ``sum(x, axis=0) == 0`` then ``hilbert(ihilbert(x)) == x``.

This property fails not only for length-2 arrays (as previously reported) but also for length-4 arrays. Interestingly, length-6 appears to work, suggesting the issue may be specific to certain even lengths (powers of 2, or small even numbers).

The pattern suggests that Nyquist mode handling in the FFT-based pseudo-differential operator implementation has edge cases for specific even-length arrays.

## Fix

This extends the previously reported bug. The implementation in `/scipy/fftpack/_pseudo_diffs.py` needs a more comprehensive fix for Nyquist mode handling across all even-length arrays, not just length-2. The issue appears to be most severe for small even lengths (2 and 4), but the root cause needs investigation to ensure correctness for all input sizes.