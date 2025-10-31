# Bug Report: xarray.namedarray.NamedArray.permute_dims() Silent Failure with Duplicate Dimensions

**Target**: `xarray.namedarray.core.NamedArray.permute_dims`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`NamedArray.permute_dims()` silently fails to transpose array data when dimension names are duplicates. The method returns an unchanged copy instead of transposing the array, violating its documented behavior.

## Property-Based Test

```python
import numpy as np
import warnings
from hypothesis import given, strategies as st, settings
from xarray.namedarray.core import NamedArray


@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_permute_dims_with_duplicate_names_transposes_data(rows, cols):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        arr = NamedArray(("x", "x"), np.arange(rows * cols).reshape(rows, cols))

    result = arr.permute_dims()

    np.testing.assert_array_equal(result.to_numpy(), arr.to_numpy().T,
                                   err_msg="permute_dims() should transpose data even with duplicate dimension names")
```

**Failing input**: Any 2D array with duplicate dimension names, e.g., `rows=2, cols=2`

## Reproducing the Bug

```python
import numpy as np
import warnings
from xarray.namedarray.core import NamedArray

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    arr = NamedArray(("x", "x"), np.array([[1, 2], [3, 4]]))

print("Original:")
print(arr.to_numpy())

result = arr.permute_dims()

print("\nAfter permute_dims():")
print(result.to_numpy())

print("\nExpected (transposed):")
print(arr.to_numpy().T)
```

Output:
```
Original:
[[1 2]
 [3 4]]

After permute_dims():
[[1 2]
 [3 4]]

Expected (transposed):
[[1 3]
 [2 4]]
```

## Why This Is A Bug

The documentation for `permute_dims()` states: "By default, reverse the order of the dimensions." When called without arguments, it should transpose the array regardless of dimension names. However, when dimensions have duplicate names (which the code explicitly allows with a warning), the method incorrectly short-circuits and returns an unchanged copy.

The bug occurs in `core.py:1043`:

```python
if len(dims) < 2 or dims == self.dims:
    return self.copy(deep=False)
```

When dimension names are duplicates like `('x', 'x')`, reversing them gives `('x', 'x')`, which equals `self.dims`. The code incorrectly concludes no transposition is needed, even though the underlying axes should still be swapped.

This is a "silent failure" - the exact type the duplicate dimension warning mentions: "most xarray functionality is likely to fail silently if you do not [rename dimensions]."

## Fix

```diff
--- a/xarray/namedarray/core.py
+++ b/xarray/namedarray/core.py
@@ -1040,7 +1040,10 @@ class NamedArray(NamedArrayAggregations, Generic[_ShapeType_co, _DType_co]):
         else:
             dims = tuple(infix_dims(dim, self.dims, missing_dims))

-        if len(dims) < 2 or dims == self.dims:
+        axes = self.get_axis_num(dims)
+        assert isinstance(axes, tuple)
+
+        if len(dims) < 2 or axes == tuple(range(self.ndim)):
             # no need to transpose if only one dimension
             # or dims are in same order
             return self.copy(deep=False)
@@ -1048,7 +1051,4 @@ class NamedArray(NamedArrayAggregations, Generic[_ShapeType_co, _DType_co]):
-        axes = self.get_axis_num(dims)
-        assert isinstance(axes, tuple)
-
         return permute_dims(self, axes)
```

The fix checks if the axes are in the same order (e.g., `(0, 1, 2)`) rather than checking if dimension names are the same. This correctly handles duplicate dimension names.