# Bug Report: xarray.structure.alignment.broadcast() String Exclude Parameter

**Target**: `xarray.structure.alignment.broadcast`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `broadcast()` function incorrectly handles string values for the `exclude` parameter, treating the string as a sequence of characters rather than a single dimension name. This causes dimension names that are substrings of the exclude string to be incorrectly excluded.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import xarray as xr
from xarray.structure.alignment import broadcast


@given(
    dim_name=st.text(min_size=2, max_size=5, alphabet=st.characters(whitelist_categories=('Ll',))),
    other_dim=st.text(min_size=1, max_size=1, alphabet=st.characters(whitelist_categories=('Ll',)))
)
def test_broadcast_exclude_string_exact_match(dim_name, other_dim):
    """Property: When exclude is a string dimension name, only that exact dimension should be excluded."""
    from hypothesis import assume
    assume(other_dim not in dim_name)
    assume(dim_name != other_dim)

    da1 = xr.DataArray([1, 2], dims=[dim_name])
    da2 = xr.DataArray([3, 4, 5], dims=[other_dim])

    result1, result2 = broadcast(da1, da2, exclude=dim_name)

    assert other_dim in result1.dims, \
        f"Dimension '{other_dim}' should NOT be excluded when exclude='{dim_name}'"

    if other_dim in dim_name:
        print(f"BUG FOUND: '{other_dim}' was excluded because it's a substring of '{dim_name}'")
```

**Failing input**: `dim_name='xy'`, `other_dim='x'` (and many other combinations where one dimension name contains characters from another)

## Reproducing the Bug

```python
import numpy as np
import xarray as xr
from xarray.structure.alignment import broadcast

da_x = xr.DataArray([1, 2, 3], dims=['x'])
da_y = xr.DataArray([4, 5], dims=['y'])

result_x, result_y = broadcast(da_x, da_y, exclude='xy')

print(f"da_x dims: {da_x.dims}")
print(f"da_y dims: {da_y.dims}")
print(f"result_x dims after broadcast: {result_x.dims}")
print(f"result_y dims after broadcast: {result_y.dims}")

assert 'x' not in result_y.dims, "Dimension 'x' should be excluded"
assert 'y' not in result_x.dims, "Dimension 'y' should be excluded"
```

**Expected behavior**: Only dimension 'xy' (which doesn't exist) should be excluded. Dimensions 'x' and 'y' should be broadcast normally.

**Actual behavior**: Both dimensions 'x' and 'y' are excluded because Python's `in` operator treats the string 'xy' as a sequence of characters, so `'x' in 'xy'` returns `True`.

## Why This Is A Bug

The `broadcast()` function's signature explicitly accepts `exclude: str | Iterable[Hashable] | None`, documenting that a string can be passed. According to the `align()` function (which `broadcast()` calls), a string should represent a single dimension name to exclude, not multiple dimensions.

However, the `broadcast()` function passes the string directly to `_get_broadcast_dims_map_common_coords()` and `_broadcast_helper()` without normalization. These functions use `dim not in exclude` and `for dim in exclude`, which for strings performs character-level operations rather than treating the string as a single dimension name.

This violates the principle of least surprise and is inconsistent with how `align()` handles the same parameter (align converts `str` to `[str]`).

## Fix

```diff
--- a/xarray/structure/alignment.py
+++ b/xarray/structure/alignment.py
@@ -1294,8 +1294,12 @@ def broadcast(
     """

     if exclude is None:
         exclude = set()
+    elif isinstance(exclude, str):
+        # Convert string to a set containing the single dimension name
+        # to avoid treating it as a sequence of characters
+        exclude = {exclude}
+    # else: exclude is already an iterable
     args = align(*args, join="outer", copy=False, exclude=exclude)

     dims_map, common_coords = _get_broadcast_dims_map_common_coords(args, exclude)
     result = [_broadcast_helper(arg, exclude, dims_map, common_coords) for arg in args]
```

This fix ensures that:
1. A string exclude parameter is converted to a set containing that single dimension name
2. The behavior is now consistent with `align()` which also converts string to a collection
3. The `dim not in exclude` and `for dim in exclude` operations work correctly on dimension names rather than characters