# Bug Report: pandas.api.extensions.take OverflowError Instead of ValueError

**Target**: `pandas.api.extensions.take`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `take` function raises `OverflowError` instead of the documented `ValueError` when `allow_fill=True` and indices contain extremely large negative values beyond the C long range.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import numpy as np
from pandas.api.extensions import take
import pytest

@given(
    arr=st.lists(st.integers(), min_size=5, max_size=100),
    negative_val=st.integers(max_value=-2)
)
@settings(max_examples=500)
def test_take_allow_fill_invalid_negative_raises_valueerror(arr, negative_val):
    np_arr = np.array(arr)
    with pytest.raises(ValueError):
        take(np_arr, [0, negative_val], allow_fill=True)
```

**Failing input**: `arr=[0, 0, 0, 0, 0], negative_val=-9_223_372_036_854_775_809`

## Reproducing the Bug

```python
import numpy as np
from pandas.api.extensions import take

arr = np.array([0, 0, 0, 0, 0])
negative_val = -9_223_372_036_854_775_809

try:
    result = take(arr, [0, negative_val], allow_fill=True)
except ValueError as e:
    print(f"Got expected ValueError: {e}")
except OverflowError as e:
    print(f"BUG: Got OverflowError instead of ValueError: {e}")
```

## Why This Is A Bug

The function's documentation states:

> Raises
> ------
> ValueError
>     When the indexer contains negative values other than ``-1``
>     and `allow_fill` is True.

However, when the negative value is too large to fit in a C long, the function raises `OverflowError` during the `ensure_platform_int(indices)` call before it can validate the indices and raise the documented `ValueError`.

## Fix

The fix would be to validate indices before converting them to platform integers, or catch the OverflowError and convert it to a ValueError with an appropriate message.

```diff
--- a/pandas/core/algorithms.py
+++ b/pandas/core/algorithms.py
@@ -1226,10 +1226,15 @@ def take(
             stacklevel=find_stack_level(),
         )

-    indices = ensure_platform_int(indices)
+    try:
+        indices = ensure_platform_int(indices)
+    except OverflowError as err:
+        if allow_fill:
+            raise ValueError("indices contains values that are too large") from err
+        raise

     if allow_fill:
-        # Pandas style, -1 means NA
         validate_indices(indices, arr.shape[axis])
         result = take_nd(
             arr, indices, axis=axis, allow_fill=True, fill_value=fill_value
```