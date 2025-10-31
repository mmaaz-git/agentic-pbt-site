# Bug Report: scipy.ndimage convolve/correlate Even-Kernel Inconsistency

**Target**: `scipy.ndimage.convolve1d` and `scipy.ndimage.correlate1d` (also affects 2D/ND versions)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The mathematical relationship `convolve(x, w) = correlate(x, flip(w))` holds for odd-sized kernels but fails for even-sized kernels in scipy.ndimage, producing results shifted by 1 element.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np
import scipy.ndimage as ndi

@given(
    input_arr=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=10, max_value=20),
        elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
    ),
    kernel_size=st.integers(min_value=3, max_value=7)
)
@settings(max_examples=200)
def test_correlate1d_convolve1d_relationship(input_arr, kernel_size):
    kernel = np.random.RandomState(42).rand(kernel_size)

    corr_result = ndi.correlate1d(input_arr, kernel, mode='constant')
    conv_result = ndi.convolve1d(input_arr, kernel[::-1], mode='constant')

    assert np.allclose(corr_result, conv_result, rtol=1e-12, atol=1e-12), \
        "correlate1d(x, w) should equal convolve1d(x, flip(w))"
```

**Failing input**: Even-sized kernels (e.g., `kernel_size=4`)

## Reproducing the Bug

```python
import numpy as np
import scipy.ndimage as ndi

input_arr = np.array([1., 2., 3., 4., 5.])
kernel = np.array([1., 2.])

conv_result = ndi.convolve1d(input_arr, kernel, mode='constant')
corr_result = ndi.correlate1d(input_arr, kernel[::-1], mode='constant')

print("Input:", input_arr)
print("Kernel:", kernel)
print("Flipped kernel:", kernel[::-1])
print()
print("convolve1d(input, kernel):", conv_result)
print("correlate1d(input, flip(kernel)):", corr_result)
print()
print("Are they equal?", np.array_equal(conv_result, corr_result))
```

**Output:**
```
Input: [1. 2. 3. 4. 5.]
Kernel: [1. 2.]
Flipped kernel: [2. 1.]

convolve1d(input, kernel): [ 4.  7. 10. 13. 10.]
correlate1d(input, flip(kernel)): [ 1.  4.  7. 10. 13.]

Are they equal? False
```

**Comparison with odd-sized kernel (working correctly):**
```python
kernel_odd = np.array([1., 2., 3.])
conv_odd = ndi.convolve1d(input_arr, kernel_odd, mode='constant')
corr_odd = ndi.correlate1d(input_arr, kernel_odd[::-1], mode='constant')

print("With odd kernel [1, 2, 3]:")
print("convolve1d:", conv_odd)
print("correlate1d with flip:", corr_odd)
print("Equal?", np.array_equal(conv_odd, corr_odd))
```

**Output:**
```
With odd kernel [1, 2, 3]:
convolve1d: [ 4. 10. 16. 22. 22.]
correlate1d with flip: [ 4. 10. 16. 22. 22.]
Equal? True
```

## Why This Is A Bug

The mathematical definition of convolution and correlation states:

- **Convolution**: `(f * g)[n] = Σ f[m] g[n - m]`
- **Correlation**: `(f ⋆ g)[n] = Σ f[m] g[n + m]`

This implies: `convolve(x, w) = correlate(x, flip(w))`

This relationship should hold **regardless of kernel size**. However, scipy.ndimage violates this property for even-sized kernels due to inconsistent handling of the kernel's "origin" (center position).

For odd-sized kernels, both functions use the mathematical center. For even-sized kernels, there is no unique center, and `convolve1d` and `correlate1d` make different choices about which element to treat as the origin, causing a 1-element shift.

Even adjusting the `origin` parameter cannot fix this - it just shifts the output but doesn't restore the mathematical relationship.

## Fix

The fix requires making `convolve` and `correlate` use the same convention for determining the origin of even-sized kernels. Looking at the issue more closely:

For an even-sized kernel of length `n`, both functions should use either:
- `origin = -(n // 2)`, or
- `origin = -(n // 2) + 1`

consistently.

Currently, the functions use different defaults, causing the shift. The fix would be in `/scipy/ndimage/_filters.py` to ensure both functions calculate the default origin identically for even-sized kernels.

A minimal patch concept:
```diff
# In _filters.py, ensure convolve and correlate use the same origin calculation:

def _get_default_origin(weights):
-    # Current inconsistent behavior
-    return different values for convolve vs correlate
+    # Consistent behavior
+    return -(len(weights) // 2)
```

The exact implementation would require reviewing the C code that these functions call, as the actual filtering is done in compiled code.