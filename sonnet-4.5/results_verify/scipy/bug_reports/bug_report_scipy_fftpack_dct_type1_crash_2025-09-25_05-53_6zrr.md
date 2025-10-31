# Bug Report: scipy.fftpack DCT Type 1 Crashes on Single-Element Arrays

**Target**: `scipy.fftpack.dct` (type 1)
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `scipy.fftpack.dct` function with `type=1` crashes with a `RuntimeError: zero-length FFT requested` when given a single-element array. DCT types 2, 3, and 4 handle single-element arrays correctly, making this an inconsistency that breaks valid use cases.

## Property-Based Test

```python
import numpy as np
import scipy.fftpack as fftpack
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis import strategies as st

@given(
    arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    ),
    st.integers(min_value=1, max_value=4)
)
def test_dct_handles_all_sizes(x, dct_type):
    result = fftpack.dct(x, type=dct_type, norm='ortho')
    assert len(result) == len(x)
```

**Failing input**: `x = np.array([1.])` with `dct_type=1`

## Reproducing the Bug

```python
import numpy as np
import scipy.fftpack as fftpack

x = np.array([1.])
result = fftpack.dct(x, type=1)
```

Output:
```
RuntimeError: zero-length FFT requested
```

## Why This Is A Bug

1. **Single-element arrays are valid input**: There's no mathematical reason why DCT type 1 cannot be computed on a single element. The DCT-I of a single element should be well-defined.

2. **Inconsistent behavior**: DCT types 2, 3, and 4 all successfully handle single-element arrays:
   ```python
   fftpack.dct(np.array([1.]), type=2)  # Works: returns [1.]
   fftpack.dct(np.array([1.]), type=3)  # Works: returns [1.]
   fftpack.dct(np.array([1.]), type=4)  # Works: returns [1.]
   fftpack.dct(np.array([1.]), type=1)  # Crashes!
   ```

3. **Uninformative error**: The error message "zero-length FFT requested" suggests an off-by-one error in calculating the internal FFT size needed for the DCT-I computation.

4. **Affects property testing**: This crash prevents comprehensive property-based testing of DCT round-trip properties across all valid input sizes.

## Fix

The issue likely lies in how the DCT type 1 implementation calculates the size of the intermediate FFT. DCT-I typically uses an FFT of size `2*(n-1)` where `n` is the input length. For `n=1`, this results in `2*0=0`, causing the "zero-length FFT" error.

The fix should:
1. Add a special case for `n=1` inputs in DCT type 1
2. For single-element input `[x]`, return `[x]` (the identity transform)
3. Alternatively, document that DCT type 1 requires arrays of length ≥ 2

The preferred approach is option 1 (handle single-element arrays) to maintain consistency with other DCT types.