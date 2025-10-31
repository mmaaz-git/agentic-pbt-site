# Bug Report: scipy.fftpack hilbert/ihilbert Not True Inverses

**Target**: `scipy.fftpack.hilbert` and `scipy.fftpack.ihilbert`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The functions `scipy.fftpack.hilbert` and `scipy.fftpack.ihilbert` are documented as inverse operations, but `ihilbert(hilbert(x))` does not return `x`. The functions incorrectly zero out the DC component and Nyquist frequency component in the Fourier domain, breaking the inverse relationship.

## Property-Based Test

```python
import numpy as np
import scipy.fftpack as fftpack
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis import strategies as st

@given(arrays(dtype=np.float64, shape=st.integers(min_value=1, max_value=100)))
def test_hilbert_ihilbert_roundtrip(x):
    result = fftpack.ihilbert(fftpack.hilbert(x))
    assert np.allclose(result, x, rtol=1e-10, atol=1e-12)
```

**Failing input**: `x = np.array([1., 2., 3., 4.])`

## Reproducing the Bug

```python
import numpy as np
import scipy.fftpack as fftpack

x = np.array([1., 2., 3., 4.])
result = fftpack.ihilbert(fftpack.hilbert(x))

print(f"Input:    {x}")
print(f"Output:   {result}")
print(f"Expected: {x}")
print(f"Match:    {np.allclose(result, x)}")
```

Output:
```
Input:    [1. 2. 3. 4.]
Output:   [-1. -1.  1.  1.]
Expected: [1. 2. 3. 4.]
Match:    False
```

## Why This Is A Bug

The documentation for these functions states:

**`hilbert`**: `y_j = sqrt(-1)*sign(j) * x_j, y_0 = 0`

**`ihilbert`**: `y_j = -sqrt(-1)*sign(j) * x_j, y_0 = 0`

Mathematically, these operations should be inverses:
```
ihilbert(hilbert(x_j)) = -sqrt(-1)*sign(j) * (sqrt(-1)*sign(j) * x_j)
                       = -sqrt(-1) * sqrt(-1) * sign(j)^2 * x_j
                       = -(-1) * 1 * x_j
                       = x_j
```

However, in practice, both functions zero out not only the DC component (j=0) but also the Nyquist frequency component (j=n/2 for even n). This causes information loss and breaks the inverse property.

The Fourier domain analysis shows:
- Original FFT at j=0: `10-0j`
- After hilbert+ihilbert at j=0: `0-0j`
- Original FFT at j=2 (Nyquist): `-2-0j`
- After hilbert+ihilbert at j=2: `0-0j`

This violates the API contract that claims these functions are inverses.

## Fix

The functions should either:

1. **Preserve the DC and Nyquist components** during the transform (not zero them out), making them true inverses, or
2. **Update the documentation** to clarify they are not true inverses and explain which frequency components are lost

Option 1 is preferable as it makes the functions more useful and matches the documented behavior. The fix would involve modifying the kernel that performs the convolution to not zero out the Nyquist frequency component.

A proper implementation should apply the sign function correctly:
- For `j=0`: `sign(0) = 0`, so `y_0 = 0` (as documented)
- For `j in [1, n/2)`: `sign(j) = +1`
- For `j = n/2` (Nyquist, even n): `sign(n/2)` should be defined (either 0 or ±1 consistently)
- For `j in (n/2, n)`: `sign(j) = -1`

The current implementation appears to treat the Nyquist frequency the same as DC (setting it to 0), which breaks the inverse property.