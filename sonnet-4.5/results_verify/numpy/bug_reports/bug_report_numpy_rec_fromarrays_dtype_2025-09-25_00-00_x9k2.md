# Bug Report: numpy.rec.fromarrays Silent Data Corruption with Dtype Mismatch

**Target**: `numpy.rec.fromarrays`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `fromarrays` is called with an explicit `dtype` parameter that has a smaller range than the input data's natural dtype, it silently corrupts data through integer overflow/underflow instead of raising an error or preserving the original values.

## Property-Based Test

```python
import numpy as np
import numpy.rec
from hypothesis import given, strategies as st

@given(st.lists(st.integers(), min_size=1, max_size=10))
def test_fromarrays_dtype_preserves_data(data):
    arr1 = np.array(data)
    arr2 = np.array(data)
    dtype = np.dtype([('a', 'i8'), ('b', 'i8')])
    rec = numpy.rec.fromarrays([arr1, arr2], dtype=dtype)
    assert np.array_equal(rec.a, arr1)
```

**Failing input**: `data=[9_223_372_036_854_775_808]` (2^63, just above max int64)

## Reproducing the Bug

```python
import numpy as np
import numpy.rec

data = [9_223_372_036_854_775_808]
arr1 = np.array(data)
arr2 = np.array(data)

dtype = np.dtype([('a', 'i8'), ('b', 'i8')])
rec = numpy.rec.fromarrays([arr1, arr2], dtype=dtype)

print(f"Original: {arr1[0]}")
print(f"After fromarrays: {rec.a[0]}")
print(f"Data corrupted: {arr1[0] != rec.a[0]}")
```

Output:
```
Original: 9223372036854775808
After fromarrays: -9223372036854775808
Data corrupted: True
```

## Why This Is A Bug

The function silently corrupts data when the input array's dtype (uint64 in this case) doesn't fit in the target dtype (int64). The value wraps around from positive to negative. This violates the expected behavior that either:
1. The data should be preserved correctly, OR
2. An error should be raised for incompatible dtypes

Silent data corruption is particularly dangerous because it can go unnoticed and lead to incorrect results downstream.

## Fix

The fix should validate that the dtype conversion is safe before performing the assignment at line 659 in `_core/records.py`:

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -656,7 +656,11 @@ def fromarrays(arrayList, dtype=None, shape=None, formats=None,
         if testshape != shape:
             raise ValueError(f'array-shape mismatch in array {k} ("{name}")')

-        _array[name] = obj
+        try:
+            temp = obj.astype(descr[k].base, casting='safe')
+            _array[name] = temp
+        except TypeError:
+            raise ValueError(f'Cannot safely cast array {k} ("{name}") from {obj.dtype} to {descr[k].base}')

     return _array
```