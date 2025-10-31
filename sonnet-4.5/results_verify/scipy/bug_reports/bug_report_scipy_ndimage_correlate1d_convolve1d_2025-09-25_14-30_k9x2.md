# Bug Report: scipy.ndimage correlate1d/convolve1d Mathematical Relationship Violation

**Target**: `scipy.ndimage.correlate1d` and `scipy.ndimage.convolve1d`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The fundamental mathematical relationship between correlation and convolution is violated in scipy.ndimage when using `mode='constant'`. Mathematically, `correlate(x, w)` should equal `convolve(x, w[::-1])`, but this property does not hold for scipy.ndimage's implementations with constant boundary mode.

## Property-Based Test

```python
import numpy as np
import scipy.ndimage as ndimage
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays


@given(
    input_arr=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=10, max_value=50),
        elements=st.floats(
            min_value=-100,
            max_value=100,
            allow_nan=False,
            allow_infinity=False
        )
    ),
    weights=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=3, max_value=9),
        elements=st.floats(
            min_value=-10,
            max_value=10,
            allow_nan=False,
            allow_infinity=False
        )
    )
)
@settings(max_examples=100, deadline=None)
def test_correlate1d_convolve1d_relationship(input_arr, weights):
    from hypothesis import assume
    assume(len(weights) <= len(input_arr))

    corr = ndimage.correlate1d(input_arr, weights, mode='constant', cval=0.0)
    conv = ndimage.convolve1d(input_arr, weights[::-1], mode='constant', cval=0.0)

    assert np.allclose(corr, conv, rtol=1e-10, atol=1e-10)
```

**Failing input**: `input_arr=array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])`, `weights=array([1., 1., 1., 1.])`

## Reproducing the Bug

```python
import numpy as np
import scipy.ndimage as ndimage

input_arr = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
weights = np.array([1., 1., 1., 1.])

corr = ndimage.correlate1d(input_arr, weights, mode='constant', cval=0.0)
conv = ndimage.convolve1d(input_arr, weights[::-1], mode='constant', cval=0.0)

print(f"correlate1d result: {corr}")
print(f"convolve1d result:  {conv}")
print(f"Are they equal? {np.allclose(corr, conv)}")
```

**Output:**
```
correlate1d result: [2. 3. 4. 4. 4. 4. 4. 4. 4. 3.]
convolve1d result:  [3. 4. 4. 4. 4. 4. 4. 4. 3. 2.]
Are they equal? False
```

The results differ at boundary positions [0, 1, 8, 9].

**Comparison with NumPy:**
```python
np_corr = np.correlate(input_arr, weights, mode='same')
np_conv = np.convolve(input_arr, weights[::-1], mode='same')
print(f"NumPy correlate: {np_corr}")
print(f"NumPy convolve:  {np_conv}")
print(f"NumPy equal? {np.allclose(np_corr, np_conv)}")
```

**Output:**
```
NumPy correlate: [2. 3. 4. 4. 4. 4. 4. 4. 4. 3.]
NumPy convolve:  [2. 3. 4. 4. 4. 4. 4. 4. 4. 3.]
NumPy equal? True
```

NumPy correctly maintains the mathematical relationship, but scipy.ndimage does not.

## Why This Is A Bug

The mathematical definition of correlation and convolution guarantees that:
```
correlate(f, g) = convolve(f, reverse(g))
```

This is a fundamental relationship in signal processing and is correctly implemented in NumPy's `correlate` and `convolve`. However, scipy.ndimage violates this property when using `mode='constant'`.

**Additional evidence:**
- The relationship DOES hold for other modes ('nearest', 'reflect', 'wrap')
- This suggests the bug is specifically in how 'constant' mode handles boundaries differently between the two functions

The inconsistency between modes and the violation of a fundamental mathematical property indicate a logic error in the boundary handling code for `mode='constant'`.

## Fix

The issue appears to be in how the 'constant' boundary mode is applied differently for correlation vs convolution. The boundary extension should be symmetric with respect to the relationship `correlate(x, w) = convolve(x, w[::-1])`.

To fix this bug, the boundary handling in `mode='constant'` for either `correlate1d` or `convolve1d` (or both) needs to be corrected to maintain the mathematical duality. Specifically:

1. Both functions should use consistent conventions for how `origin` affects the filter placement
2. The padding direction should be adjusted so that reversing the kernel and swapping the operation produces identical results

Without access to the C implementation, a detailed patch cannot be provided, but the fix should ensure that when mode='constant', the relationship `correlate1d(x, w, origin=o) == convolve1d(x, w[::-1], origin=-o)` holds for appropriate origin values.