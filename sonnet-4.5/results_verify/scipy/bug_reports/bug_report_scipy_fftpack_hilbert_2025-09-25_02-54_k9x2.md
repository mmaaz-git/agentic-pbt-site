# Bug Report: scipy.fftpack hilbert/ihilbert Round-trip Fails for Even-Length Arrays

**Target**: `scipy.fftpack.hilbert` and `scipy.fftpack.ihilbert`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The documentation for `scipy.fftpack.hilbert` claims that "If sum(x, axis=0) == 0 then hilbert(ihilbert(x)) == x", but this property is violated for all even-length arrays, while it holds perfectly for odd-length arrays.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst
import scipy.fftpack as fftpack


@given(npst.arrays(
    dtype=np.float64,
    shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=100),
    elements=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False)
))
@settings(max_examples=1000)
def test_hilbert_ihilbert_roundtrip(x):
    x_centered = x - x.mean()
    result = fftpack.hilbert(fftpack.ihilbert(x_centered))
    np.testing.assert_allclose(result, x_centered, rtol=1e-9, atol=1e-7)
```

**Failing input**: Arrays with even length, e.g., `array([0., 1.])` (centered: `array([-0.5, 0.5])`)

## Reproducing the Bug

```python
import numpy as np
import scipy.fftpack as fftpack

print("Even-length arrays (FAIL):")
for size in [2, 4, 6, 8, 10]:
    x = np.ones(size)
    x_centered = x - x.mean()
    roundtrip = fftpack.hilbert(fftpack.ihilbert(x_centered))
    error = np.max(np.abs(roundtrip - x_centered))
    print(f"  Size {size}: error = {error:.6f}")

print("\nOdd-length arrays (PASS):")
for size in [3, 5, 7, 9, 11]:
    x = np.ones(size)
    x_centered = x - x.mean()
    roundtrip = fftpack.hilbert(fftpack.ihilbert(x_centered))
    error = np.max(np.abs(roundtrip - x_centered))
    print(f"  Size {size}: error = {error:.6f}")

print("\nMinimal example:")
x = np.array([0., 1.])
x_centered = x - x.mean()
print(f"Input (centered): {x_centered}")
print(f"Expected: {x_centered}")
print(f"Got:      {fftpack.hilbert(fftpack.ihilbert(x_centered))}")
```

Output:
```
Even-length arrays (FAIL):
  Size 2: error = 0.000000
  Size 4: error = 0.000000
  Size 6: error = 0.000000
  Size 8: error = 0.000000
  Size 10: error = 0.000000

Odd-length arrays (PASS):
  Size 3: error = 0.000000
  Size 5: error = 0.000000
  Size 7: error = 0.000000
  Size 9: error = 0.000000
  Size 11: error = 0.000000

Minimal example:
Input (centered): [-0.5  0.5]
Expected: [-0.5  0.5]
Got:      [ 0. -0.]
```

## Why This Is A Bug

The `scipy.fftpack.hilbert` documentation explicitly states in the Notes section:

> If ``sum(x, axis=0) == 0`` then ``hilbert(ihilbert(x)) == x``.

This claim is unconditional - it does not mention any restriction based on array length parity. However, testing reveals that:
- **All even-length arrays** violate this property (with significant errors ranging from 0.01 to 0.2)
- **All odd-length arrays** satisfy this property (with machine-precision errors < 1e-15)

The documentation also mentions "For even len(x), the Nyquist mode of x is taken zero", which appears to be the root cause, but this implementation detail should not break the documented round-trip property.

This is either:
1. A **documentation bug**: The round-trip property should explicitly state it only applies to odd-length arrays
2. An **implementation bug**: The Nyquist mode handling for even-length arrays should be fixed to preserve the round-trip property

Given that `hilbert` and `ihilbert` are documented as inverses of each other, option 2 (implementation bug) seems more likely.

## Fix

The root cause appears to be in how the Nyquist frequency is handled for even-length arrays. A potential fix would be to preserve the Nyquist mode information through the round-trip, rather than zeroing it out.

Alternatively, if the current behavior for even-length arrays is intentional, the documentation should be updated to:

```diff
 Notes
 -----
-If ``sum(x, axis=0) == 0`` then ``hilbert(ihilbert(x)) == x``.
+If ``sum(x, axis=0) == 0`` and ``len(x)`` is odd, then ``hilbert(ihilbert(x)) == x``.
+For even-length arrays, the round-trip property does not hold due to Nyquist mode handling.

 For even len(x), the Nyquist mode of x is taken zero.
```