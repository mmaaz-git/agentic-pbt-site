# Bug Report: numpy.ma Shrink Parameter Does Not Compress Scalar False Masks

**Target**: `numpy.ma.array`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When creating a masked array with `mask=False` (Python bool) and `shrink=True`, the mask is incorrectly expanded to a boolean array instead of being compressed to `nomask`, unlike the behavior with `mask=np.False_`.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=20)
)
@settings(max_examples=500)
def test_shrink_with_scalar_false_mask(data_list):
    data = np.array(data_list)

    arr_with_npfalse = ma.array(data, mask=np.False_, shrink=True)
    arr_with_pyfalse = ma.array(data, mask=False, shrink=True)

    mask_npfalse = ma.getmask(arr_with_npfalse)
    mask_pyfalse = ma.getmask(arr_with_pyfalse)

    assert mask_npfalse is ma.nomask
    assert mask_pyfalse is ma.nomask
```

**Failing input**: `data_list=[0]` (or any list)

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

arr_correct = ma.array([1, 2, 3], mask=np.False_, shrink=True)
print(f"mask=np.False_: {ma.getmask(arr_correct)}, is nomask? {ma.getmask(arr_correct) is ma.nomask}")

arr_buggy = ma.array([1, 2, 3], mask=False, shrink=True)
print(f"mask=False: {ma.getmask(arr_buggy)}, is nomask? {ma.getmask(arr_buggy) is ma.nomask}")

assert ma.getmask(arr_correct) is ma.nomask
assert ma.getmask(arr_buggy) is not ma.nomask
```

Output:
```
mask=np.False_: False, is nomask? True
mask=False: [False False False], is nomask? False
AssertionError: BUG: Python False should also shrink to nomask but got [False False False]
```

## Why This Is A Bug

The `shrink` parameter is documented to "force compression of an empty mask." When `mask=False` (a scalar with no True values), it represents an empty mask and should be compressed to `nomask` when `shrink=True`.

The inconsistency between `mask=False` and `mask=np.False_` is problematic:
- Both are scalar False values semantically representing "no masked elements"
- `mask=np.False_` correctly shrinks to `nomask`
- `mask=False` incorrectly expands to a boolean array `[False, False, ...]`

This causes:
1. **Memory inefficiency**: Stores unnecessary boolean arrays (~115 bytes vs ~25 bytes for a 3-element array)
2. **API inconsistency**: Behavior differs based on Python vs NumPy literals
3. **Violation of shrink contract**: The parameter doesn't work as documented

## Fix

The issue likely occurs during mask initialization when a Python `False` is passed. The code should check if the mask is a scalar `False` (regardless of whether it's `np.False_` or Python `False`) and, when `shrink=True`, set it to `nomask` instead of broadcasting it to an array.

A potential fix would be in the mask processing code to treat scalar `False` values uniformly:

```diff
- if shrink and mask is not nomask and not np.any(mask):
-     mask = nomask
+ if shrink and np.ndim(mask) == 0 and not mask:
+     mask = nomask
+ elif shrink and mask is not nomask and not np.any(mask):
+     mask = nomask
```

This would ensure that any scalar False value (Python or NumPy) is properly compressed to `nomask` when `shrink=True`.