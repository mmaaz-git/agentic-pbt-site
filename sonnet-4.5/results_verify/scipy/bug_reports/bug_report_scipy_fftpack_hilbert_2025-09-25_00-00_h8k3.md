# Bug Report: scipy.fftpack.hilbert - Round-trip Property Fails for Even-Length Arrays

**Target**: `scipy.fftpack.hilbert` and `scipy.fftpack.ihilbert`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The documentation for `scipy.fftpack.hilbert` claims that "If `sum(x, axis=0) == 0` then `hilbert(ihilbert(x)) == x`", but this property only holds for odd-length arrays. For even-length arrays, the round-trip fails completely, returning zeros instead of the original input.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst
import scipy.fftpack as fftpack


@given(
    st.data(),
    npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100)
)
@settings(max_examples=200)
def test_hilbert_ihilbert_round_trip(data, shape):
    dtype = data.draw(st.sampled_from([np.float32, np.float64]))
    max_val = 1e6 if dtype == np.float32 else 1e10

    x = data.draw(npst.arrays(
        dtype=dtype,
        shape=shape,
        elements=st.floats(min_value=-max_val, max_value=max_val, allow_nan=False, allow_infinity=False)
    ))

    if np.sum(x) != 0:
        x = x - np.mean(x)

    y = fftpack.hilbert(x)
    result = fftpack.ihilbert(y)
    atol, rtol = (1e-6, 1e-6) if dtype == np.float32 else (1e-10, 1e-10)
    assert np.allclose(result, x, atol=atol, rtol=rtol)
```

**Failing input**: Arrays with even length (e.g., `array([-0.5, 0.5])`)

## Reproducing the Bug

```python
import numpy as np
import scipy.fftpack as fftpack

x_even = np.array([-0.5, 0.5])
y_even = fftpack.hilbert(x_even)
result_even = fftpack.ihilbert(y_even)

print("Even-length array (n=2):")
print(f"  Input:  {x_even}")
print(f"  Output: {result_even}")
print(f"  Round-trip successful: {np.allclose(result_even, x_even)}")

x_odd = np.array([-1.0, 0.0, 1.0])
y_odd = fftpack.hilbert(x_odd)
result_odd = fftpack.ihilbert(y_odd)

print("\nOdd-length array (n=3):")
print(f"  Input:  {x_odd}")
print(f"  Output: {result_odd}")
print(f"  Round-trip successful: {np.allclose(result_odd, x_odd)}")
```

Output:
```
Even-length array (n=2):
  Input:  [-0.5  0.5]
  Output: [-0. -0.]
  Round-trip successful: False

Odd-length array (n=3):
  Input:  [-1.  0.  1.]
  Output: [-1.00000000e+00 -5.55111512e-17  1.00000000e+00]
  Round-trip successful: True
```

## Why This Is A Bug

The docstring for `scipy.fftpack.hilbert` explicitly states:

> If `sum(x, axis=0) == 0` then `hilbert(ihilbert(x)) == x`.

However, this claim is false for even-length arrays. The documentation also mentions:

> For even len(x), the Nyquist mode of x is taken zero.

This behavior of zeroing the Nyquist mode for even-length arrays makes the Hilbert transform non-invertible, which contradicts the documented round-trip property.

The issue is that for even-length arrays:
1. The Hilbert transform zeros out the Nyquist frequency mode
2. This destroys information in the signal
3. The inverse Hilbert transform cannot recover the original signal

## Fix

The documentation should be corrected to explicitly state the limitation:

```diff
     Notes
     -----
-    If ``sum(x, axis=0) == 0`` then ``hilbert(ihilbert(x)) == x``.
+    If ``sum(x, axis=0) == 0`` and ``len(x)`` is odd, then
+    ``hilbert(ihilbert(x)) == x``. For even-length arrays, the Nyquist
+    mode is set to zero, which makes the transform non-invertible.

     For even len(x), the Nyquist mode of x is taken zero.
```

Alternatively, the implementation could be fixed to handle even-length arrays correctly by not zeroing the Nyquist mode, though this would require careful consideration of the mathematical definition and compatibility with existing code.