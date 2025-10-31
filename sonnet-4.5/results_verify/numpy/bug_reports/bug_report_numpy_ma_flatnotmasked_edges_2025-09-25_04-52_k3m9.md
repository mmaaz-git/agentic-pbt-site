# Bug Report: numpy.ma.flatnotmasked_edges Invalid Indices for Empty Array

**Target**: `numpy.ma.flatnotmasked_edges`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`flatnotmasked_edges()` returns invalid indices `[0, -1]` when given an empty array instead of returning `None` or an empty result.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings

@given(st.just(ma.array([], dtype=int)))
@settings(max_examples=5)
def test_flatnotmasked_edges_empty_array(arr):
    result = ma.flatnotmasked_edges(arr)
    if result is not None:
        assert len(result) == 0 or (len(result) == 2 and result[0] >= 0 and result[1] >= result[0])
```

**Failing input**: `masked_array(data=[], mask=False, dtype=int64)`

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

empty_arr = ma.array([], dtype=int)
result = ma.flatnotmasked_edges(empty_arr)

print(f"Result: {result}")
print(f"Expected: None")
print(f"Actual: [0, -1]")
```

## Why This Is A Bug

The docstring states: "Returns None if all values are masked." For an empty array, there are no unmasked values, so it should return `None`. Instead, it returns `[0, -1]`, which represents invalid indices (the second index -1 is negative and less than the first index 0).

The root cause is at line 1995 in `numpy/ma/extras.py`:
```python
return np.array([0, a.size - 1])
```

When `a.size = 0`, this becomes `[0, -1]`.

## Fix

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -1992,6 +1992,8 @@ def flatnotmasked_edges(a):
     """
     m = getmask(a)
     if m is nomask or not np.any(m):
+        if a.size == 0:
+            return None
         return np.array([0, a.size - 1])
     unmasked = np.flatnonzero(~m)
     if len(unmasked) > 0:
```