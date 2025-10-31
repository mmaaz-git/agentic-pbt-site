# Bug Report: xarray.namedarray permute_dims Missing Dims Handling

**Target**: `xarray.namedarray.core.NamedArray.permute_dims`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `permute_dims` method ignores the `missing_dims` parameter when it's set to 'warn' or 'ignore'. Even when these options are used, the function raises a `ValueError` for missing dimensions instead of warning or ignoring them as documented.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import numpy as np
from xarray.namedarray.core import NamedArray


@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=100)
def test_permute_dims_with_missing_dim_ignore(n):
    data = np.arange(n * 2).reshape(n, 2)
    arr = NamedArray(('x', 'y'), data)

    result = arr.permute_dims('x', 'z', missing_dims='ignore')

    assert result.dims == ('x', 'y')
```

**Failing input**: Any valid NamedArray with at least 2 dimensions, e.g., `n=1`

## Reproducing the Bug

```python
import numpy as np
from xarray.namedarray.core import NamedArray

data = np.arange(6).reshape(2, 3)
arr = NamedArray(('x', 'y'), data)

result = arr.permute_dims('x', 'z', missing_dims='ignore')
```

**Output:**
```
ValueError: ('x', 'z') must be a permuted list of ('x', 'y'), unless `...` is included
```

**Expected behavior:** The method should ignore the missing dimension 'z' and return an array with dims ('x', 'y').

## Why This Is A Bug

According to the docstring of `permute_dims` (line 1017-1022 in core.py), the `missing_dims` parameter should control what happens when specified dimensions are not present:
- `"raise"`: raise an exception (default)
- `"warn"`: raise a warning and ignore missing dimensions
- `"ignore"`: ignore missing dimensions

However, the function raises a `ValueError` regardless of this setting when dimensions are missing and no ellipsis is present.

The bug is in `xarray.namedarray.utils.infix_dims` (line 170-176 in utils.py). After calling `drop_missing_dims` to filter out invalid dimensions, the code checks if the remaining dimensions form a permutation of all dimensions. This check fails when dimensions were dropped, causing the function to raise an error even when `missing_dims='ignore'` or `missing_dims='warn'`.

## Fix

```diff
--- a/xarray/namedarray/utils.py
+++ b/xarray/namedarray/utils.py
@@ -169,9 +169,10 @@ def infix_dims(
                 yield d
     else:
         existing_dims = drop_missing_dims(dims_supplied, dims_all, missing_dims)
-        if set(existing_dims) ^ set(dims_all):
+        # Only check for complete permutation if we're in 'raise' mode
+        if missing_dims == "raise" and set(existing_dims) ^ set(dims_all):
             raise ValueError(
                 f"{dims_supplied} must be a permuted list of {dims_all}, unless `...` is included"
             )
         yield from existing_dims