# Bug Report: numpy.ma.masked_equal Mask Type Inconsistency

**Target**: `numpy.ma.masked_equal` (and related functions: `masked_greater`, `masked_less`, `masked_greater_equal`, `masked_less_equal`, `masked_inside`)
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `mask` attribute of masked arrays returned by `ma.masked_equal` is sometimes a scalar `numpy.bool` and sometimes an `ndarray`, depending on whether any elements match the masking condition. This inconsistency breaks code that expects to index the mask.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst


@given(
    arr=npst.arrays(dtype=np.int32, shape=st.integers(10, 50), elements=st.integers(-100, 100)),
    value=st.integers(-100, 100)
)
@settings(max_examples=500)
def test_masked_equal_mask_is_indexable(arr, value):
    m_arr = ma.masked_equal(arr, value)
    for i in range(len(arr)):
        try:
            mask_val = m_arr.mask[i]
            if arr[i] == value:
                assert mask_val == True
            else:
                assert mask_val == False
        except (IndexError, TypeError) as e:
            pytest.fail(f'Cannot index mask at position {i}. Mask type: {type(m_arr.mask)}, value: {m_arr.mask}. Error: {e}')
```

**Failing input**: `arr=array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32), value=1`

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

arr = np.array([0, 0, 0, 0, 0], dtype=np.int32)
value = 1

m_arr = ma.masked_equal(arr, value)

print(f'Mask type: {type(m_arr.mask)}')
print(f'Mask value: {m_arr.mask}')

try:
    _ = m_arr.mask[0]
    print('Can index mask')
except IndexError as e:
    print(f'ERROR: Cannot index mask - {e}')


arr2 = np.array([0, 1, 0], dtype=np.int32)
m_arr2 = ma.masked_equal(arr2, 1)

print(f'\nWith some matches - Mask type: {type(m_arr2.mask)}')
print(f'Can index: {m_arr2.mask[0]}')
```

Output:
```
Mask type: <class 'numpy.bool'>
Mask value: False
ERROR: Cannot index mask - invalid index to scalar variable.

With some matches - Mask type: <class 'numpy.ndarray'>
Can index: False
```

## Why This Is A Bug

The API documentation does not specify that the `mask` attribute can be either a scalar or an array depending on runtime conditions. Users naturally expect the mask to always be indexable since it corresponds to array elements. This breaks common usage patterns like:

```python
for i in range(len(data)):
    if masked_data.mask[i]:
        handle_masked_element(i)
```

This code works when some elements are masked but crashes when no elements are masked, making the bug subtle and hard to catch.

## Fix

The fix should ensure the `mask` attribute is always an array, even when all elements have the same mask value. This can be done by modifying the masked array construction to always use an array mask:

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -3530,7 +3530,10 @@ def masked_equal(x, value, copy=True):
     output = masked_where(d, x, copy=copy)
-    output.fill_value = value
+    if np.isscalar(output.mask):
+        output._mask = np.full(output.shape, output.mask, dtype=bool)
+    output.fill_value = value
     return output
```

Similar fixes would be needed for `masked_greater`, `masked_less`, `masked_greater_equal`, `masked_less_equal`, and `masked_inside`.