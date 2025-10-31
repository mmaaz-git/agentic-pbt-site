# Bug Report: xarray.namedarray.expand_dims Creates Duplicate Dimensions

**Target**: `xarray.namedarray._array_api.expand_dims`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `expand_dims()` method creates duplicate dimension names when called without an explicit `dim` parameter if the array has dimensions named `dim_N` where N equals the current number of dimensions.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, settings, strategies as st
from xarray.namedarray.core import NamedArray


@st.composite
def namedarray_with_potential_collision(draw):
    ndim = draw(st.integers(min_value=1, max_value=4))
    dim_choices = [f"dim_{i}" for i in range(10)]
    dims = draw(st.lists(st.sampled_from(dim_choices), min_size=ndim, max_size=ndim, unique=True))
    shape = tuple(2 for _ in range(ndim))
    data = np.ones(shape)
    return NamedArray(tuple(dims), data)


@given(namedarray_with_potential_collision())
@settings(max_examples=500)
def test_expand_dims_no_default_duplicates(arr):
    expanded = arr.expand_dims()
    assert len(expanded.dims) == len(set(expanded.dims)), \
        f"expand_dims() created duplicate dimensions: {expanded.dims}"
```

**Failing input**: Array with `dims=('dim_1',)`

## Reproducing the Bug

```python
import numpy as np
from xarray.namedarray.core import NamedArray

data = np.array([1, 2])
arr = NamedArray(("dim_1",), data)

expanded = arr.expand_dims()
print(expanded.dims)
```

**Output**:
```
('dim_1', 'dim_1')
```

**Additional failing cases**:
- `dims=('dim_0', 'dim_2')` → expands to `('dim_2', 'dim_0', 'dim_2')`
- `dims=('a', 'b', 'dim_3')` → expands to `('dim_3', 'a', 'b', 'dim_3')`

## Why This Is A Bug

1. **Creates invalid state**: The resulting array has duplicate dimension names, which the codebase explicitly warns against: "We do not yet support duplicate dimension names... most xarray functionality is likely to fail silently"

2. **Violates reasonable expectations**: Users would expect `expand_dims()` to create a new, unique dimension name, not duplicate an existing one

3. **Undocumented limitation**: The docstring for `expand_dims()` doesn't mention that it may create duplicate dimensions for certain inputs

4. **Inconsistent behavior**: Works correctly for some dimension naming schemes but fails for others following the same pattern (`dim_N`)

## Fix

The bug is in `xarray/namedarray/_array_api.py` lines 187-188:

```python
if dim is _default:
    dim = f"dim_{len(dims)}"
```

This generates a dimension name without checking if it already exists. The fix should find the next available `dim_N` name:

```diff
--- a/xarray/namedarray/_array_api.py
+++ b/xarray/namedarray/_array_api.py
@@ -185,7 +185,14 @@ def expand_dims(
     xp = _get_data_namespace(x)
     dims = x.dims
     if dim is _default:
-        dim = f"dim_{len(dims)}"
+        # Find the next available dim_N name
+        existing_dim_numbers = set()
+        for d in dims:
+            if isinstance(d, str) and d.startswith("dim_") and d[4:].isdigit():
+                existing_dim_numbers.add(int(d[4:]))
+        n = 0
+        while n in existing_dim_numbers:
+            n += 1
+        dim = f"dim_{n}"
     d = list(dims)
     d.insert(axis, dim)
     out = x._new(dims=tuple(d), data=xp.expand_dims(x._data, axis=axis))