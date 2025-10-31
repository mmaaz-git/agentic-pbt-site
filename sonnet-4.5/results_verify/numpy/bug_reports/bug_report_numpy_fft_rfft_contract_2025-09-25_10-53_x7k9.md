# Bug Report: numpy.fft.rfft Documentation Contract Violation

**Target**: `numpy.fft.rfft`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `numpy.fft.rfft` function documentation claims that imaginary parts of complex input are "silently discarded", but the implementation actually raises a TypeError when given complex input. This is a contract violation between documented and actual behavior.

## Property-Based Test

```python
import numpy as np
import numpy.fft as fft
from hypothesis import given, strategies as st, settings


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


@given(complex_arrays(min_size=1, max_size=100))
@settings(max_examples=500)
def test_rfft_discards_imaginary(a):
    real_part_only = a.real
    result_complex = fft.rfft(a)
    result_real = fft.rfft(real_part_only)
    assert np.allclose(result_complex, result_real, rtol=1e-10, atol=1e-12), \
        f"rfft doesn't discard imaginary part as claimed"
```

**Failing input**: `array([0.+0.j])` (or any complex array)

## Reproducing the Bug

```python
import numpy as np

a = np.array([1.0 + 2.0j, 3.0 + 4.0j])

try:
    result = np.fft.rfft(a)
    print("Success:", result)
except TypeError as e:
    print("Bug confirmed - TypeError:", e)
    print()
    print("Documentation says (numpy/fft/_pocketfft.py:399):")
    print("  'If the input `a` contains an imaginary part,")
    print("   it is silently discarded.'")
    print()
    print("But rfft raises TypeError instead.")
```

Output:
```
Bug confirmed - TypeError: ufunc 'rfft_n_even' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

Documentation says (numpy/fft/_pocketfft.py:399):
  'If the input `a` contains an imaginary part,
   it is silently discarded.'

But rfft raises TypeError instead.
```

## Why This Is A Bug

The documentation for `numpy.fft.rfft` at line 399 in `numpy/fft/_pocketfft.py` explicitly states:

> If the input `a` contains an imaginary part, it is silently discarded.

However, the actual implementation rejects complex input with a TypeError. This creates a contract violation where:
1. Users following the documentation expect complex input to work (with imaginary parts ignored)
2. The implementation crashes instead of silently discarding imaginary parts
3. Code written based on the documentation will fail at runtime

## Fix

The fix requires either updating the documentation or changing the implementation. Two options:

**Option 1: Update documentation to match implementation (simpler)**

```diff
--- a/numpy/fft/_pocketfft.py
+++ b/numpy/fft/_pocketfft.py
@@ -396,7 +396,7 @@ def rfft(a, n=None, axis=-1, norm=None, out=None):
     and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely
     real. If `n` is odd, there is no term at fs/2; ``A[-1]`` contains
     the largest positive frequency (fs/2*(n-1)/n), and is complex in the
     general case.

-    If the input `a` contains an imaginary part, it is silently discarded.
+    The input `a` must be real-valued. Complex input will raise TypeError.

     Examples
```

**Option 2: Update implementation to match documentation**

```diff
--- a/numpy/fft/_pocketfft.py
+++ b/numpy/fft/_pocketfft.py
@@ -414,6 +414,7 @@ def rfft(a, n=None, axis=-1, norm=None, out=None):
     """
     a = asarray(a)
+    a = a.real  # Silently discard imaginary part as documented
     if n is None:
         n = a.shape[axis]
     output = _raw_fft(a, n, axis, True, True, norm, out=out)
```

**Recommendation**: Option 1 is safer as it documents existing behavior without changing semantics. Option 2 matches the documentation but could hide bugs in user code where complex arrays are passed unintentionally.