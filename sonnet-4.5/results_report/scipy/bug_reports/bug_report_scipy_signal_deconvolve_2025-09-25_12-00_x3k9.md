# Bug Report: scipy.signal.deconvolve Numerical Instability

**Target**: `scipy.signal.deconvolve`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.signal.deconvolve` exhibits severe numerical instability for certain divisor patterns, violating its documented property that `signal = convolve(divisor, quotient) + remainder`. For signals of moderate length (>20 elements), the reconstruction error can exceed 6% relative error.

## Property-Based Test

```python
from hypothesis import given, settings, assume, strategies as st
import numpy as np
import scipy.signal

@settings(max_examples=200)
@given(
    signal=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=10, max_size=50),
    divisor=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=3, max_size=10)
)
def test_deconvolve_inverse_property(signal, divisor):
    signal_arr = np.array(signal, dtype=np.float64)
    divisor_arr = np.array(divisor, dtype=np.float64)

    assume(np.abs(divisor_arr[0]) > 1e-10)

    quotient, remainder = scipy.signal.deconvolve(signal_arr, divisor_arr)

    reconstructed = scipy.signal.convolve(divisor_arr, quotient, mode='full')
    reconstructed_with_remainder = reconstructed.copy()
    min_len = min(len(reconstructed_with_remainder), len(remainder))
    reconstructed_with_remainder[:min_len] += remainder[:min_len]

    expected_len = len(signal_arr)
    reconstructed_trimmed = reconstructed_with_remainder[:expected_len]

    np.testing.assert_allclose(signal_arr, reconstructed_trimmed, rtol=1e-5, atol=1e-8)
```

**Failing input**:
- `signal=[65.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.875]`
- `divisor=[0.125, 0.0, 2.0]`

## Reproducing the Bug

```python
import numpy as np
import scipy.signal

signal = np.array([65.0] + [0.0]*21 + [1.875])
divisor = np.array([0.125, 0.0, 2.0])

quotient, remainder = scipy.signal.deconvolve(signal, divisor)
reconstructed = scipy.signal.convolve(divisor, quotient, mode='full')
reconstructed[:len(remainder)] += remainder

print(f"Expected last value: {signal[-1]}")
print(f"Reconstructed last value: {reconstructed[len(signal)-1]}")
print(f"Error: {abs(signal[-1] - reconstructed[len(signal)-1])}")
print(f"Max quotient magnitude: {np.max(np.abs(quotient)):.2e}")
```

Output:
```
Expected last value: 1.875
Reconstructed last value: 2.0
Error: 0.125
Max quotient magnitude: 5.72e+14
```

## Why This Is A Bug

The `deconvolve` docstring explicitly states:

> Returns the quotient and remainder such that
> `signal = convolve(divisor, quotient) + remainder`

However, this property is violated. The reconstruction error of 0.125 represents a 6.7% relative error. The root cause is numerical instability - the quotient values grow exponentially (reaching ~5.72×10¹⁴), causing catastrophic cancellation when combined with the remainder.

This occurs even though:
1. The inputs are well-formed floating-point numbers
2. The divisor's first element is non-zero (0.125)
3. The signal length is modest (23 elements)

The function provides no warning about numerical instability and does not validate whether the divisor will lead to stable deconvolution.

## Fix

The numerical instability arises from the polynomial division algorithm used by `deconvolve` when the divisor represents a potentially unstable filter. The divisor `[0.125, 0.0, 2.0]` corresponds to the polynomial 0.125 + 2.0z², which can amplify errors exponentially during deconvolution.

**Recommended fixes:**

1. **Add numerical stability checking**: Before performing deconvolution, check if the divisor will lead to numerical instability (e.g., check pole locations or condition number).

2. **Add warnings to documentation**: Document that `deconvolve` may be numerically unstable for certain divisors, particularly those representing unstable filters.

3. **Use regularized deconvolution**: Implement a more numerically stable algorithm such as Tikhonov regularization for ill-conditioned cases.

4. **Add error checking**: After deconvolution, verify that the reconstruction property holds within tolerance and raise a warning if it doesn't.

Example validation check that could be added:

```python
def deconvolve(signal, divisor):
    # ... existing implementation ...

    # Validate reconstruction
    reconstructed = np.convolve(divisor, quotient, mode='full')
    reconstructed[:len(remainder)] += remainder
    error = np.max(np.abs(signal - reconstructed[:len(signal)]))

    if error > 1e-8 * np.max(np.abs(signal)):
        import warnings
        warnings.warn(
            f"Deconvolution may be numerically unstable. "
            f"Reconstruction error: {error:.2e}",
            RuntimeWarning
        )

    return quotient, remainder
```