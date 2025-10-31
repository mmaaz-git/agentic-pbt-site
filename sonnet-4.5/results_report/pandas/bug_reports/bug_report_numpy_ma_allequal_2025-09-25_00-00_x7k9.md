# Bug Report: numpy.ma.allequal fill_value=False Logic Error

**Target**: `numpy.ma.allequal`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.ma.allequal(a, b, fill_value=False)` returns `False` without checking unmasked values when arrays have any masked elements, even when all unmasked values are identical.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps


@st.composite
def identical_masked_arrays_with_some_masked(draw):
    size = draw(st.integers(min_value=2, max_value=30))
    data = draw(nps.arrays(dtype=np.float64, shape=(size,),
                          elements={"allow_nan": False, "allow_infinity": False,
                                   "min_value": -100, "max_value": 100}))
    mask = draw(nps.arrays(dtype=bool, shape=(size,)))

    assume(mask.any())
    assume((~mask).any())

    return data, mask


@given(identical_masked_arrays_with_some_masked())
@settings(max_examples=500)
def test_allequal_fillvalue_false_bug(data_mask):
    data, mask = data_mask

    x = ma.array(data, mask=mask)
    y = ma.array(data.copy(), mask=mask.copy())

    result_false = ma.allequal(x, y, fill_value=False)

    unmasked_equal = np.array_equal(data[~mask], data[~mask])
    if unmasked_equal:
        assert result_false == True, \
            f"allequal with fill_value=False returned False for arrays with identical unmasked values"
```

**Failing input**: `data=(array([0., 0.]), array([False, True]))`

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

x = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
y = ma.array([1.0, 999.0, 3.0], mask=[False, True, False])

print(f"Unmasked values are identical: {np.array_equal(ma.compressed(x), ma.compressed(y))}")

result_false = ma.allequal(x, y, fill_value=False)
print(f"allequal(x, y, fill_value=False): {result_false}")
```

Output:
```
Unmasked values are identical: True
allequal(x, y, fill_value=False): False
```

## Why This Is A Bug

The `fill_value` parameter should control how **masked positions** are compared, not whether to skip comparison entirely. When `fill_value=False`, the function should:
1. Check if unmasked values are equal (should return True if they are)
2. Return False only if unmasked values differ OR if masked positions differ between arrays

Currently, when `fill_value=False` and any masked values exist, the function immediately returns `False` without checking unmasked values.

## Fix

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -4080,7 +4080,14 @@ def allequal(a, b, fill_value=True):
         dm = array(d, mask=m, copy=False)
         return dm.filled(True).all(None)
     else:
-        return False
+        x = getdata(a)
+        y = getdata(b)
+        d = umath.equal(x, y)
+        dm = array(d, mask=m, copy=False)
+        # With fill_value=False, masked positions are NOT equal
+        # So if any position is masked, return False
+        # But if unmasked values are all equal, that should still count
+        return dm.all(None) and not m.all()
```

Note: A better fix would be to return `True` only when unmasked values are equal AND there are no positions where one array is masked and the other is not. The current implementation's logic of "return False when fill_value=False and any masks exist" is fundamentally flawed.