# Bug Report: numpy.ma.default_fill_value Type Object Crash

**Target**: `numpy.ma.default_fill_value`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`default_fill_value` crashes with AttributeError when passed numpy type objects (e.g., `np.float32`, `np.int64`) instead of dtype instances.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st

@given(st.data())
def test_default_fill_value_matches_type(data_strategy):
    dtype = data_strategy.draw(st.sampled_from([np.float32, np.float64, np.int32, np.int64]))

    fill_val = ma.default_fill_value(dtype)

    assert np.isscalar(fill_val) or fill_val is not None
```

**Failing input**: `dtype=np.float32` (or any numpy type object)

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

result = ma.default_fill_value(np.float32)
```

Output:
```
AttributeError: 'getset_descriptor' object has no attribute 'names'
```

## Why This Is A Bug

The function crashes instead of either:
1. Working correctly (converting `np.float32` to `np.dtype(np.float32)`)
2. Raising a clear error message about incorrect input type

The crash occurs because `_get_dtype_of` checks `hasattr(obj, 'dtype')`, which is True for numpy types (they have a `dtype` descriptor), but then accessing `obj.dtype` on a type returns a descriptor object rather than a dtype instance.

Users might reasonably pass `np.float32` thinking it's similar to `np.dtype(np.float32)`, so the function should either handle this gracefully or provide a clear error message.

## Fix

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -224,7 +224,11 @@ def _recursive_fill_value(dtype, f):

 def _get_dtype_of(obj):
     """ Convert the argument for *_fill_value into a dtype """
-    if isinstance(obj, np.dtype):
+    if isinstance(obj, type) and issubclass(obj, np.generic):
+        return np.dtype(obj)
+    elif isinstance(obj, np.dtype):
         return obj
     elif hasattr(obj, 'dtype'):
         return obj.dtype
```