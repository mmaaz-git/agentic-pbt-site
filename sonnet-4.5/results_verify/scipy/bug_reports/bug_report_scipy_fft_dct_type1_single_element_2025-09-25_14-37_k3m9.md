# Bug Report: scipy.fft.dct DCT Type 1 Single Element Crash

**Target**: `scipy.fft.dct` and `scipy.fft.idct`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

DCT type 1 crashes with RuntimeError when given a single-element array, while DCT types 2, 3, and 4 handle single-element arrays correctly.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst
import numpy as np
import scipy.fft

@given(npst.arrays(
    dtype=np.float64,
    shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100),
    elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e7, max_value=1e7)
))
@settings(max_examples=500)
def test_dct_idct_roundtrip(x):
    for dct_type in [1, 2, 3, 4]:
        transformed = scipy.fft.dct(x, type=dct_type)
        result = scipy.fft.idct(transformed, type=dct_type)
        assert np.allclose(result, x, rtol=1e-8, atol=1e-8), f"DCT type {dct_type} roundtrip failed"
```

**Failing input**: `array([0.])`

## Reproducing the Bug

```python
import numpy as np
import scipy.fft

x = np.array([0.])
scipy.fft.dct(x, type=1)
```

Output:
```
RuntimeError: zero-length FFT requested
```

## Why This Is A Bug

1. Single-element arrays are valid input for DCT operations
2. DCT types 2, 3, and 4 all handle single-element arrays correctly:
   - `scipy.fft.dct([0.], type=2)` returns `[0.]`
   - `scipy.fft.dct([0.], type=3)` returns `[0.]`
   - `scipy.fft.dct([0.], type=4)` returns `[0.]`
3. Only DCT type 1 crashes with this error
4. The error message "zero-length FFT requested" suggests an off-by-one error or incorrect size calculation in the DCT type 1 implementation

## Fix

The bug appears to be in `/home/npc/.local/lib/python3.13/site-packages/scipy/fft/_pocketfft/realtransforms.py` around line 45. The DCT type 1 implementation likely needs special handling for single-element arrays, similar to how types 2, 3, and 4 handle them.

A potential fix would be to add a check for single-element arrays in the DCT type 1 code path and return the input unchanged (or with appropriate normalization), as the DCT of a single element should be mathematically well-defined.