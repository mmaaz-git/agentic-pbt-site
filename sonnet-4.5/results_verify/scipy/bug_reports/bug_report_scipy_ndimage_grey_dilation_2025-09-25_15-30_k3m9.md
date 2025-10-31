# Bug Report: scipy.ndimage.grey_dilation - Inconsistent Results with Even-Sized Footprints

**Target**: `scipy.ndimage.grey_dilation`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`grey_dilation` with a flat structuring element produces different results than `maximum_filter` when using even-sized footprints (e.g., size=4), violating the documented equivalence. Odd-sized footprints work correctly.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst
import scipy.ndimage as ndimage


@given(
    input_array=npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=5, max_side=10),
        elements=st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    ),
    size=st.integers(min_value=3, max_value=5),
)
@settings(max_examples=50, deadline=None)
def test_grey_dilation_is_maximum_filter(input_array, size):
    grey_dil_result = ndimage.grey_dilation(input_array, size=size)
    max_filter_result = ndimage.maximum_filter(input_array, size=size)

    assert np.allclose(
        grey_dil_result, max_filter_result, rtol=1e-10, atol=1e-10
    ), "grey_dilation with flat structuring element should equal maximum_filter"
```

**Failing input**: `input_array=array([[1., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], ...])` (all zeros except [0,0]=1), `size=4`

## Reproducing the Bug

```python
import numpy as np
import scipy.ndimage as ndimage

input_array = np.array([[1., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]])

grey_dil = ndimage.grey_dilation(input_array, size=4)
max_filter = ndimage.maximum_filter(input_array, size=4)

print("grey_dilation result:")
print(grey_dil)
print("\nmaximum_filter result:")
print(max_filter)
print("\nEqual?", np.array_equal(grey_dil, max_filter))
```

**Output:**
```
grey_dilation result:
[[1. 1. 0. 0. 0.]
 [1. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]

maximum_filter result:
[[1. 1. 1. 0. 0.]
 [1. 1. 1. 0. 0.]
 [1. 1. 1. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]

Equal? False
```

## Why This Is A Bug

The docstring for `grey_dilation` explicitly states:

> "For the simple case of a full and flat structuring element, it can be viewed as a maximum filter over a sliding window."

This property holds for **odd-sized** footprints (size=3, 5, 7, ...) but **fails for even-sized** footprints (size=4, 6, 8, ...). The functions should be equivalent regardless of whether the size is odd or even, as both use the same conceptual operation: finding the maximum value within a window.

The bug appears to be in how `grey_dilation` handles the centering/origin of even-sized structuring elements, causing it to use a different effective footprint than `maximum_filter`.

## Fix

This is likely an origin/centering issue in the implementation of `grey_dilation` when handling even-sized structuring elements. The fix would involve ensuring that `grey_dilation` and `maximum_filter` use consistent centering logic for even-sized footprints, probably by adjusting how the origin is calculated in `_morphology.py` for the even-size case.