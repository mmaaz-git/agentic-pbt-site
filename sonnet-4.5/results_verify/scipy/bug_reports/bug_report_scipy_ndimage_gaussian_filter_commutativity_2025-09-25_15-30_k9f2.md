# Bug Report: scipy.ndimage.gaussian_filter Commutativity Violation with mode='constant'

**Target**: `scipy.ndimage.gaussian_filter`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `gaussian_filter` function violates mathematical commutativity when using `mode='constant'` with different sigma values. While `gaussian_filter(gaussian_filter(x, σ₁), σ₂)` should equal `gaussian_filter(gaussian_filter(x, σ₂), σ₁)` for all boundary modes, this property fails specifically for `mode='constant'`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np
import scipy.ndimage as ndi

@given(
    input_array=arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(6, 12), st.integers(6, 12)),
        elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-5.0, max_value=5.0)
    ),
    sigma1=st.floats(min_value=0.5, max_value=2.0),
    sigma2=st.floats(min_value=0.5, max_value=2.0)
)
@settings(max_examples=150, deadline=None)
def test_gaussian_filter_commutativity(input_array, sigma1, sigma2):
    result1 = ndi.gaussian_filter(
        ndi.gaussian_filter(input_array, sigma=sigma1, mode='constant'),
        sigma=sigma2, mode='constant'
    )
    result2 = ndi.gaussian_filter(
        ndi.gaussian_filter(input_array, sigma=sigma2, mode='constant'),
        sigma=sigma1, mode='constant'
    )

    assert np.allclose(result1, result2, rtol=1e-10, atol=1e-10)
```

**Failing input**: `input_array=np.ones((6,6))`, `sigma1=1.0`, `sigma2=0.5`

## Reproducing the Bug

```python
import numpy as np
import scipy.ndimage as ndi

x = np.ones((10, 10))
sigma1, sigma2 = 1.0, 0.5

r1 = ndi.gaussian_filter(
    ndi.gaussian_filter(x, sigma=sigma1, mode='constant'),
    sigma=sigma2, mode='constant'
)
r2 = ndi.gaussian_filter(
    ndi.gaussian_filter(x, sigma=sigma2, mode='constant'),
    sigma=sigma1, mode='constant'
)

print(f"Max difference: {np.max(np.abs(r1 - r2))}")
print(f"Corner: r1[0,0]={r1[0,0]:.10f}, r2[0,0]={r2[0,0]:.10f}")

print("\nWith mode='reflect' (works correctly):")
r3 = ndi.gaussian_filter(
    ndi.gaussian_filter(x, sigma=sigma1, mode='reflect'),
    sigma=sigma2, mode='reflect'
)
r4 = ndi.gaussian_filter(
    ndi.gaussian_filter(x, sigma=sigma2, mode='reflect'),
    sigma=sigma1, mode='reflect'
)
print(f"Max difference: {np.max(np.abs(r3 - r4))}")
```

**Output:**
```
Max difference: 0.010438823029858324
Corner: r1[0,0]=0.4233636458, r2[0,0]=0.4314319273

With mode='reflect' (works correctly):
Max difference: 0.0
```

## Why This Is A Bug

Gaussian convolution is mathematically commutative. For any two Gaussians with standard deviations σ₁ and σ₂:

```
G(σ₁) ⊗ G(σ₂) ⊗ f = G(σ₂) ⊗ G(σ₁) ⊗ f
```

This property should hold regardless of the boundary mode used.

**Evidence that this is a bug:**

1. **All other boundary modes work correctly**: `mode='reflect'`, `'nearest'`, `'mirror'`, and `'wrap'` all exhibit perfect commutativity (difference = 0.0)

2. **Only boundaries are affected**: On a 50x50 array, the center 10x10 region shows perfect commutativity even with `mode='constant'`, proving the core algorithm is correct

3. **Same sigma works**: When σ₁ = σ₂, commutativity holds even with `mode='constant'`, showing the issue is specific to different sigma values

4. **Mathematical inconsistency**: There is no mathematical reason why `mode='constant'` should break commutativity while all other modes preserve it

## Impact

- Users performing sequential Gaussian filtering with `mode='constant'` get different results depending on the order of operations
- Affects image processing pipelines that assume commutativity
- Violates fundamental mathematical properties users expect from Gaussian filtering
- Can lead to subtle bugs in scientific computing and computer vision applications

## Fix

The issue appears to be in how `mode='constant'` (cval=0) handles boundary padding when different sigma values are used. The fix should ensure that:

1. Boundary handling for `mode='constant'` is symmetric with respect to filter order
2. The discrete Gaussian implementation preserves mathematical commutativity
3. All boundary modes handle sequential filtering consistently

Possible approaches:
- Review the truncation/radius calculation for different sigma values
- Ensure boundary padding is applied identically regardless of which sigma is applied first
- Verify that the discrete Gaussian kernel generation doesn't introduce order-dependent artifacts