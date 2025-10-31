# Bug Report: scipy.fftpack.shift Composition Fails for Even-Length Arrays

**Target**: `scipy.fftpack.shift`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `scipy.fftpack.shift` function violates the mathematical composition property `shift(shift(x, a), b) = shift(x, a+b)` for all even-length arrays, while correctly implementing it for odd-length arrays.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import numpy as np
from scipy import fftpack


@settings(max_examples=200)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False,
                       min_value=-1e6, max_value=1e6), min_size=2, max_size=100).filter(lambda x: len(x) % 2 == 0),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-10.0, max_value=10.0),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-10.0, max_value=10.0)
)
def test_shift_composition_even_length(x, a, b):
    x_arr = np.array(x)
    result1 = fftpack.shift(fftpack.shift(x_arr, a), b)
    result2 = fftpack.shift(x_arr, a + b)
    assert np.allclose(result1, result2, rtol=1e-9, atol=1e-9)
```

**Failing input**: `x=[1.0, 0.0], a=1.0, b=1.0`

## Reproducing the Bug

```python
import numpy as np
from scipy import fftpack

x = np.array([1.0, 0.0])
a, b = 1.0, 1.0

result1 = fftpack.shift(fftpack.shift(x, a), b)
result2 = fftpack.shift(x, a + b)

print("shift(shift(x, 1.0), 1.0):", result1)
print("shift(x, 2.0):", result2)
print("Difference:", result1 - result2)
```

Output:
```
shift(shift(x, 1.0), 1.0): [ 1.45464871 -0.45464871]
shift(x, 2.0): [0.7465753 0.2534247]
Difference: [ 0.70807342 -0.70807342]
```

## Pattern Analysis

The bug manifests exclusively for even-length arrays:

```python
import numpy as np
from scipy import fftpack

for n in range(2, 21):
    x = np.random.rand(n)
    a, b = 1.0, 1.0

    result1 = fftpack.shift(fftpack.shift(x, a), b)
    result2 = fftpack.shift(x, a + b)

    matches = np.allclose(result1, result2, rtol=1e-9)
    print(f"n={n}: {'PASS' if matches else 'FAIL'}")
```

Results:
- **ALL odd lengths (3, 5, 7, 9, 11, 13, 15, 17, 19)**: PASS
- **ALL even lengths (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)**: FAIL

## Why This Is A Bug

The `shift` function is documented to implement the periodic shift operation `y(u) = x(u+a)` using the Fourier coefficient relation:

```
y_j = exp(j*a*2*pi/period*sqrt(-1)) * x_j
```

For any linear shift operation, the composition property `shift(x, a+b) = shift(shift(x, a), b)` is a fundamental mathematical requirement. The fact that this property holds for odd-length arrays but fails for even-length arrays indicates an implementation bug, likely in how the convolution kernel is initialized or applied for even-length sequences.

The magnitude of errors (up to 0.7 in the minimal example) far exceeds numerical precision issues, confirming this is a logic error rather than floating-point accumulation.

## Fix

The bug appears to be in the `scipy.fftpack._pseudo_diffs.shift` function's use of `convolve.init_convolution_kernel` with the `zero_nyquist` parameter. For even-length arrays, the Nyquist frequency component (at index n/2) requires special handling in FFT-based operations.

The likely issue is in `/home/npc/.local/lib/python3.13/site-packages/scipy/fftpack/_pseudo_diffs.py` at the kernel initialization:

```python
omega_real = convolve.init_convolution_kernel(n,kernel_real,d=0,
                                              zero_nyquist=0)
omega_imag = convolve.init_convolution_kernel(n,kernel_imag,d=1,
                                              zero_nyquist=0)
```

The `zero_nyquist` parameter handling needs to be corrected for even-length arrays to ensure the shift operator composes correctly. A proper fix would require examining the Cython implementation in `scipy/fftpack/convolve.pyx` to correct how the Nyquist frequency is handled in the shift operation's phase multiplication.