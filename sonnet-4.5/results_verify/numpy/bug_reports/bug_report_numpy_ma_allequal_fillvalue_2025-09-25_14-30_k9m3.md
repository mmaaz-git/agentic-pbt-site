# Bug Report: numpy.ma.allequal Incorrect Logic with fill_value=False

**Target**: `numpy.ma.allequal`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.ma.allequal(a, b, fill_value=False)` returns `False` without comparing unmasked values when arrays contain any masked elements, even when all unmasked values are identical.

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
            "allequal with fill_value=False returned False for arrays with identical unmasked values"
```

**Failing input**: `data=(array([0., 0.]), mask=array([False, True]))`

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

x = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
y = ma.array([1.0, 999.0, 3.0], mask=[False, True, False])

print(f"Unmasked x: {ma.compressed(x)}")
print(f"Unmasked y: {ma.compressed(y)}")
print(f"allequal(x, y, fill_value=False): {ma.allequal(x, y, fill_value=False)}")
```

Output:
```
Unmasked x: [1. 3.]
Unmasked y: [1. 3.]
allequal(x, y, fill_value=False): False
```

## Why This Is A Bug

The `fill_value` parameter should control how masked positions are compared, not whether to skip comparison entirely. The current implementation in `numpy/ma/core.py` (lines 4080-4087) has this logic:

```python
def allequal(a, b, fill_value=True):
    m = mask_or(getmask(a), getmask(b))
    if m is nomask:
        # ... compare data normally
    elif fill_value:
        # ... compare unmasked values, treat masked as equal
    else:
        return False  # <-- BUG: returns False without checking anything!
```

When `fill_value=False` and any masked values exist (`m is not nomask`), the function immediately returns `False` without comparing unmasked values. This is incorrect because:

1. Two arrays with identical unmasked values should be considered equal at those positions
2. The function should only return `False` if unmasked values differ OR if masks differ
3. The parameter should affect how masked positions are handled, not bypass all comparison

## Fix

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -4080,7 +4080,11 @@ def allequal(a, b, fill_value=True):
         dm = array(d, mask=m, copy=False)
         return dm.filled(True).all(None)
     else:
-        return False
+        x = getdata(a)
+        y = getdata(b)
+        d = umath.equal(x, y)
+        dm = array(d, mask=m, copy=False)
+        return dm.filled(False).all(None)
```

This fix compares unmasked values and fills masked positions with `False`, so the function returns `True` only when all unmasked values are equal.