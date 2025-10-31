# Bug Report: xarray.core.duck_array_ops.sum_where Inverted Logic

**Target**: `xarray.core.duck_array_ops.sum_where`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `sum_where` function in `xarray.core.duck_array_ops` has inverted conditional logic - it sums values where the condition is **False** instead of **True**, contradicting both its name and the behavior of `numpy.sum`'s `where` parameter.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from xarray.core import duck_array_ops


@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100),
    st.lists(st.booleans(), min_size=1, max_size=100)
)
def test_sum_where_matches_numpy(data_list, where_list):
    size = min(len(data_list), len(where_list))
    data = np.array(data_list[:size])
    where = np.array(where_list[:size])

    numpy_result = np.sum(data, where=where)
    xarray_result = duck_array_ops.sum_where(data, where=where)

    assert np.isclose(numpy_result, xarray_result)
```

**Failing input**: `data=[1.0, 2.0, 3.0, 4.0, 5.0]`, `where=[True, False, True, False, True]`

## Reproducing the Bug

```python
import numpy as np
from xarray.core import duck_array_ops

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
where = np.array([True, False, True, False, True])

numpy_result = np.sum(data, where=where)
xarray_result = duck_array_ops.sum_where(data, where=where)

print(f"numpy.sum(data, where=where): {numpy_result}")
print(f"xarray sum_where(data, where=where): {xarray_result}")

assert numpy_result != xarray_result
```

## Why This Is A Bug

The function `sum_where` is expected to sum values where the condition is `True`, matching the behavior of `numpy.sum(array, where=mask)`. However, the implementation has inverted logic:

**Current implementation** (duck_array_ops.py:389-396):
```python
def sum_where(data, axis=None, dtype=None, where=None):
    xp = get_array_namespace(data)
    if where is not None:
        a = where_method(xp.zeros_like(data), where, data)
    else:
        a = data
    result = xp.sum(a, axis=axis, dtype=dtype)
    return result
```

The line `a = where_method(xp.zeros_like(data), where, data)` expands to:
- `where(where, zeros_like(data), data)`
- Which selects `0` when condition is `True`, and `data` when condition is `False`
- This is backwards!

**Impact**: This bug affects the API contract. While `nansum` in `nanops.py` works correctly due to passing a mask that gets inverted by this bug (two wrongs make a right), any direct users of `sum_where` expecting numpy-like behavior will get incorrect results.

## Fix

```diff
--- a/xarray/core/duck_array_ops.py
+++ b/xarray/core/duck_array_ops.py
@@ -389,7 +389,7 @@ def sum_where(data, axis=None, dtype=None, where=None):
     xp = get_array_namespace(data)
     if where is not None:
-        a = where_method(xp.zeros_like(data), where, data)
+        a = where_method(data, where, xp.zeros_like(data))
     else:
         a = data
     result = xp.sum(a, axis=axis, dtype=dtype)
```

**Note**: This fix must be accompanied by updating the call site in `nanops.py:99`:

```diff
--- a/xarray/computation/nanops.py
+++ b/xarray/computation/nanops.py
@@ -97,7 +97,7 @@ def nansum(a, axis=None, dtype=None, out=None, min_count=None):
     mask = isnull(a)
-    result = sum_where(a, axis=axis, dtype=dtype, where=mask)
+    result = sum_where(a, axis=axis, dtype=dtype, where=~mask)
     if min_count is not None:
         return _maybe_null_out(result, axis, mask, min_count)
     else:
```

This ensures both the function contract is correct AND existing functionality (`nansum`) continues to work.