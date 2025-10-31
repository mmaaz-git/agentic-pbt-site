# Bug Report: scipy.fft.irfft Default n Calculation for Single Element

**Target**: `scipy.fft.irfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `scipy.fft.irfft` is called on a single-element complex array without specifying `n`, it raises a ValueError due to computing an invalid default value of 0.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import scipy.fft


@given(
    st.lists(
        st.complex_numbers(
            allow_nan=False,
            allow_infinity=False,
            min_magnitude=0.0,
            max_magnitude=1e10
        ),
        min_size=1,
        max_size=100
    )
)
def test_rfft_irfft_implicit_n(x):
    x_arr = np.array(x, dtype=complex)
    result = scipy.fft.irfft(x_arr)
    assert result.shape[0] > 0
```

**Failing input**: `x=[0j]`

## Reproducing the Bug

```python
import numpy as np
import scipy.fft

x = np.array([1.0 + 0.0j])
result = scipy.fft.irfft(x)
```

**Output:**
```
ValueError: Invalid number of data points (0) specified
```

**Note:** Calling `scipy.fft.irfft(x, n=1)` works correctly and returns `[1.0]`.

## Why This Is A Bug

According to the docstring, when `n` is not specified, `irfft` should use the default `n = 2*(m-1)` where `m` is the length of the input. For a single-element input, this computes to `2*(1-1) = 0`, which is then rejected as invalid.

This breaks the natural usage pattern where users expect `irfft` to work without explicitly specifying `n`, particularly when the input comes from `rfft`. While `rfft` of a single real element produces a single complex element, calling `irfft` on that result without `n` fails.

## Fix

The default value calculation should handle the single-element case specially. For `m=1`, a reasonable default would be `n=1` (returning a single real value), which maintains consistency with the round-trip property when `n` is explicitly specified:

```python
if n is None:
    m = x.shape[axis]
    n = max(1, 2 * (m - 1))
```

Alternatively, the documentation could be updated to clarify that `n` must be explicitly specified for single-element inputs, though this would be less user-friendly.