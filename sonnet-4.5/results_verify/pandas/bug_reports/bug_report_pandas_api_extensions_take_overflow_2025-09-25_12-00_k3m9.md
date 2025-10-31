# Bug Report: pandas.api.extensions.take raises OverflowError instead of IndexError for very large indices

**Target**: `pandas.api.extensions.take`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When `pandas.api.extensions.take()` is called with an index value larger than `sys.maxsize` (2^63-1 on 64-bit systems), it raises `OverflowError` instead of the documented `IndexError`. This is inconsistent with the function's contract and behavior for smaller out-of-bounds indices.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from hypothesis.extra import numpy as npst
import numpy as np
from pandas.api import extensions

@given(
    npst.arrays(dtype=np.int64, shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=20)),
    st.lists(st.integers(), min_size=1, max_size=10)
)
def test_take_out_of_bounds_raises(arr, indices):
    assume(len(arr) > 0)
    assume(any(idx >= len(arr) or idx < -len(arr) for idx in indices))

    try:
        extensions.take(arr, indices)
        assert False, "Expected IndexError"
    except IndexError:
        pass
```

**Failing input**: `arr=array([0])`, `indices=[9_223_372_036_854_775_808]`

## Reproducing the Bug

```python
import numpy as np
from pandas.api import extensions

arr = np.array([0, 1, 2, 3, 4])
index_too_large = 9_223_372_036_854_775_808

try:
    result = extensions.take(arr, [index_too_large])
except OverflowError as e:
    print(f"OverflowError: {e}")
except IndexError as e:
    print(f"IndexError: {e}")
```

Output:
```
OverflowError: Python int too large to convert to C long
```

Expected: `IndexError` with a message about the index being out of bounds.

## Why This Is A Bug

The function's docstring explicitly states:

> Raises
> ------
> IndexError
>     When `indices` is out of bounds for the array.

An index value of `9_223_372_036_854_775_808` is clearly out of bounds for any array, yet the function raises `OverflowError` instead of `IndexError`. This violates the documented API contract.

Additionally, for consistency, smaller out-of-bounds indices (e.g., 100 for a 5-element array) correctly raise `IndexError`. The exception type should not depend on the magnitude of the out-of-bounds index.

## Fix

The issue occurs in `pandas/core/algorithms.py` at line 1229 where `ensure_platform_int(indices)` is called before validation. The fix is to catch `OverflowError` and convert it to `IndexError`:

```diff
--- a/pandas/core/algorithms.py
+++ b/pandas/core/algorithms.py
@@ -1226,7 +1226,11 @@ def take(
     if not is_array_like(arr):
         arr = np.asarray(arr)

-    indices = ensure_platform_int(indices)
+    try:
+        indices = ensure_platform_int(indices)
+    except OverflowError as err:
+        raise IndexError(
+            f"index value too large: {err}"
+        ) from err

     if allow_fill:
         # Pandas style, -1 means NA
```