# Bug Report: pandas.api.extensions.take with Index ignores fill_value parameter

**Target**: `pandas.api.extensions.take` when called with `pd.Index`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When `pandas.api.extensions.take()` is called with a `pd.Index` and a user-specified `fill_value`, the function ignores the provided fill_value and always uses the Index's default NA value (NaN for float). This violates the documented API contract and creates an inconsistency with how the function behaves on numpy arrays.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.api.extensions import take


@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000), min_size=2, max_size=20),
    fill_val=st.floats(allow_nan=False, allow_infinity=False, min_value=-10000, max_value=10000)
)
@settings(max_examples=300)
def test_index_allow_fill_with_value_should_use_fillvalue(values, fill_val):
    index = pd.Index(values, dtype='float64')
    arr = np.array(values)
    indices = [0, -1, 1]

    index_result = take(index, indices, allow_fill=True, fill_value=fill_val)
    array_result = take(arr, indices, allow_fill=True, fill_value=fill_val)

    assert index_result[1] == array_result[1], f"Index and array should return same fill_value"
    assert index_result[1] == fill_val, f"Expected fill_value {fill_val}, got {index_result[1]}"
```

**Failing input**: `values=[0.0, 0.0], fill_val=0.0` (indices=`[0, -1, 1]`)

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
from pandas.api.extensions import take

index = pd.Index([10.0, 20.0, 30.0])
arr = np.array([10.0, 20.0, 30.0])

index_result = take(index, [0, -1, 2], allow_fill=True, fill_value=99.0)
array_result = take(arr, [0, -1, 2], allow_fill=True, fill_value=99.0)

print("Index result:", list(index_result))
print("Array result:", list(array_result))

assert array_result[1] == 99.0
assert pd.isna(index_result[1])
print(f"\nBug confirmed: Index returns NaN instead of fill_value 99.0")
```

Output:
```
Index result: [10.0, nan, 30.0]
Array result: [10.0, 99.0, 30.0]

Bug confirmed: Index returns NaN instead of fill_value 99.0
```

## Why This Is A Bug

The `pandas.api.extensions.take` documentation explicitly states:

> **fill_value** : any, optional
>     Fill value to use for NA-indices when `allow_fill` is True.

When calling `take()` with a specific fill_value (e.g., 99.0), the user expects that value to be used for missing positions (-1 indices). However, when the input is a `pd.Index`, the function ignores the fill_value and always uses the Index's default NA value (NaN for float types).

This is a contract violation because:
1. The API documentation promises to use the provided fill_value
2. The behavior is inconsistent with numpy arrays (which correctly use fill_value)
3. Users cannot control what value is filled for missing positions when using Index

## Fix

The bug is in the `Index.take` method at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/base.py`:

```diff
--- a/pandas/core/indexes/base.py
+++ b/pandas/core/indexes/base.py
@@ -1155,11 +1155,17 @@ class Index(IndexOpsMixin):
         indices = ensure_platform_int(indices)
         allow_fill = self._maybe_disallow_fill(allow_fill, fill_value, indices)

-        # Note: we discard fill_value and use self._na_value, only relevant
-        #  in the case where allow_fill is True and fill_value is not None
+        # Use the provided fill_value, or self._na_value if None
+        actual_fill_value = fill_value if fill_value is not None else self._na_value
+
         values = self._values
         if isinstance(values, np.ndarray):
             taken = algos.take(
-                values, indices, allow_fill=allow_fill, fill_value=self._na_value
+                values, indices, allow_fill=allow_fill, fill_value=actual_fill_value
             )
         else:
             # algos.take passes 'axis' keyword which not all EAs accept
             taken = values.take(
-                indices, allow_fill=allow_fill, fill_value=self._na_value
+                indices, allow_fill=allow_fill, fill_value=actual_fill_value
             )
```

This fix:
1. Uses the provided fill_value when it's not None
2. Falls back to self._na_value when fill_value is None
3. Makes Index behavior consistent with numpy array behavior