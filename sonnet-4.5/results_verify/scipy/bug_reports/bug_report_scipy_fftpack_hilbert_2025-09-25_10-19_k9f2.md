# Bug Report: scipy.fftpack.hilbert Round-Trip Failure for Even-Length Arrays

**Target**: `scipy.fftpack.hilbert` and `scipy.fftpack.ihilbert`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Hilbert transform round-trip property fails for all even-length arrays, even when `sum(x) == 0`, violating the documented behavior that states "If `sum(x, axis=0) == 0` then `hilbert(ihilbert(x)) == x`".

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import scipy.fftpack as fftpack


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100))
def test_hilbert_ihilbert_roundtrip(x_list):
    x = np.array(x_list)
    x = x - np.mean(x)
    if np.abs(np.sum(x)) < 1e-9:
        result = fftpack.hilbert(fftpack.ihilbert(x))
        assert np.allclose(result, x, atol=1e-9)
```

**Failing input**: `[0.0, 1.0]` (after centering: `[-0.5, 0.5]`)

## Reproducing the Bug

```python
import numpy as np
import scipy.fftpack as fftpack

print("Testing Hilbert round-trip for even vs odd length arrays:")

for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    x = np.arange(n, dtype=float)
    x = x - np.mean(x)

    result = fftpack.hilbert(fftpack.ihilbert(x))
    matches = np.allclose(result, x, atol=1e-10)

    print(f"Length {n} ({'even' if n % 2 == 0 else 'odd '}): {'PASS' if matches else 'FAIL'}")

print("\nSpecific example with length 2:")
x = np.array([-1.0, 1.0])
print(f"x = {x}, sum(x) = {np.sum(x)}")
result = fftpack.hilbert(fftpack.ihilbert(x))
print(f"hilbert(ihilbert(x)) = {result}")
print(f"Expected: {x}")
print(f"Match: {np.allclose(result, x)}")
```

Output:
```
Testing Hilbert round-trip for even vs odd length arrays:
Length 2 (even): FAIL
Length 3 (odd ): PASS
Length 4 (even): FAIL
Length 5 (odd ): PASS
Length 6 (even): FAIL
Length 7 (odd ): PASS
Length 8 (even): FAIL
Length 9 (odd ): PASS
Length 10 (even): FAIL

Specific example with length 2:
x = [-1.  1.], sum(x) = 0.0
hilbert(ihilbert(x)) = [ 0. -0.]
Expected: [-1.  1.]
Match: False
```

## Why This Is A Bug

The documentation for `scipy.fftpack.hilbert` explicitly states:

> If `sum(x, axis=0) == 0` then `hilbert(ihilbert(x)) == x`.

However, this property fails consistently for all even-length arrays. The bug appears to be related to the Nyquist frequency handling mentioned in the docstring: "For even len(x), the Nyquist mode of x is taken zero."

This special handling of the Nyquist mode for even-length arrays breaks the round-trip property, contradicting the documented guarantee.

## Fix

The issue likely stems from how the Nyquist frequency component is handled differently for even-length arrays. The documentation note states "For even len(x), the Nyquist mode of x is taken zero", which appears to cause information loss during the transform.

The fix would require either:
1. Updating the documentation to clarify that the round-trip property only holds for odd-length arrays, OR
2. Fixing the implementation to preserve the Nyquist mode information to maintain the round-trip property for even-length arrays when `sum(x) == 0`

Recommended approach: Update the documentation to accurately reflect the current behavior:

```diff
 Notes
 -----
-If ``sum(x, axis=0) == 0`` then ``hilbert(ihilbert(x)) == x``.
+If ``sum(x, axis=0) == 0`` and ``len(x)`` is odd, then ``hilbert(ihilbert(x)) == x``.

 For even len(x), the Nyquist mode of x is taken zero.
+This causes the round-trip property to fail for even-length arrays.
```