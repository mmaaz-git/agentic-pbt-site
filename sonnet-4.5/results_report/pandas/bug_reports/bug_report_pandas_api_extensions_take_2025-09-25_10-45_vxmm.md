# Bug Report: pandas.api.extensions.take Raises OverflowError Instead of IndexError

**Target**: `pandas.api.extensions.take`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `take()` function raises `OverflowError` instead of the documented `IndexError` when given extremely large negative indices that exceed C long range.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import numpy as np
import pytest
from pandas.api.extensions import take


@settings(max_examples=500)
@given(
    size=st.integers(min_value=1, max_value=100),
    idx=st.integers()
)
def test_take_out_of_bounds_negative(size, idx):
    arr = np.arange(size)
    assume(idx < -size)

    with pytest.raises(IndexError):
        take(arr, [idx])
```

**Failing input**: `size=1, idx=-9_223_372_036_854_775_809`

## Reproducing the Bug

```python
import numpy as np
from pandas.api.extensions import take

arr = np.array([1])
idx = -9_223_372_036_854_775_809

try:
    result = take(arr, [idx])
except OverflowError as e:
    print(f"OverflowError raised (BUG): {e}")
except IndexError as e:
    print(f"IndexError raised (expected): {e}")
```

## Why This Is A Bug

The documentation for `take()` explicitly states:

> Raises
> ------
> IndexError
>     When `indices` is out of bounds for the array.

However, when an index is far out of bounds (beyond C long range), the function raises `OverflowError` instead of `IndexError`. This violates the documented API contract and could break code that specifically catches `IndexError` for handling invalid indices.

## Fix

The bug occurs because `ensure_platform_int()` is called before bounds checking. The fix should catch `OverflowError` and convert it to `IndexError`:

```diff
--- a/pandas/core/algorithms.py
+++ b/pandas/core/algorithms.py
@@ -1226,7 +1226,11 @@ def take(
     indices : np.ndarray
         Same type as the input.
     """
-    indices = ensure_platform_int(indices)
+    try:
+        indices = ensure_platform_int(indices)
+    except OverflowError as e:
+        raise IndexError(
+            f"indices are out of bounds"
+        ) from e

     if not isinstance(arr, (np.ndarray, ExtensionArray, Index, ABCSeries)):
         # GH#52981
```