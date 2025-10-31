# Bug Report: xarray.indexes normalize_label dtype type handling

**Target**: `xarray.core.indexes.normalize_label`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `normalize_label` function crashes with an `AttributeError` when passed a numpy dtype type (e.g., `np.float32`) instead of a dtype instance (e.g., `np.dtype('float32')`), despite such usage being standard practice in numpy APIs.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import numpy as np
from xarray.core.indexes import normalize_label

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
                min_size=1, max_size=20, unique=True))
def test_normalize_label_with_float32_dtype(float_values):
    assume(len(float_values) > 0)
    float32_arr = np.array(float_values, dtype=np.float32)
    result = normalize_label(float32_arr, dtype=np.float32)
    assert result.dtype == np.float32
```

**Failing input**: Any float32 array with `dtype=np.float32` (the type, not dtype instance)

## Reproducing the Bug

```python
import numpy as np
from xarray.core.indexes import normalize_label

values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

result = normalize_label(values, dtype=np.float32)
```

**Output:**
```
AttributeError: type object 'numpy.float32' has no attribute 'kind'
```

## Why This Is A Bug

1. **Numpy convention**: Numpy functions universally accept both dtype instances and dtype types. For example, `np.array([1,2,3], dtype=np.float32)` works correctly.

2. **No validation**: The function doesn't validate or normalize the `dtype` parameter before accessing `.kind` attribute.

3. **User expectation**: Users reasonably expect `dtype=np.float32` to work since it's the common numpy pattern.

4. **Crashes instead of handling**: The code directly accesses `dtype.kind` without checking if `dtype` is a proper dtype instance.

The bug occurs at line 610 in `/xarray/core/indexes.py`:
```python
if dtype is not None and dtype.kind == "f" and value.dtype.kind != "b":
```

## Fix

Normalize the dtype parameter before using it:

```diff
 def normalize_label(value, dtype=None) -> np.ndarray:
     if getattr(value, "ndim", 1) <= 1:
         value = _asarray_tuplesafe(value)
+    if dtype is not None:
+        dtype = np.dtype(dtype)
     if dtype is not None and dtype.kind == "f" and value.dtype.kind != "b":
         # pd.Index built from coordinate with float precision != 64
         # see https://github.com/pydata/xarray/pull/3153 for details
         # bypass coercing dtype for boolean indexers (ignore index)
         # see https://github.com/pydata/xarray/issues/5727
         value = np.asarray(value, dtype=dtype)
     return value
```

This fix converts any dtype-like object to a proper numpy dtype instance, matching numpy's own behavior.