# Bug Report: pandas.api.extensions.take Inconsistent Behavior with Series/Index and allow_fill

**Target**: `pandas.api.extensions.take` / `pandas.core.array_algos.take.take_nd`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `take` function has inconsistent and buggy behavior when passed Series or Index with `allow_fill=True`, despite documentation explicitly stating these are supported input types. For Series it crashes with TypeError, and for Index it silently ignores allow_fill or raises confusing errors.

## Property-Based Test

```python
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.api.extensions import take


@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100),
    num_fills=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100)
def test_take_series_with_allow_fill(values, num_fills):
    series = pd.Series(values)
    indices = [0] * (len(values) // 2) + [-1] * num_fills
    result = take(series, indices, allow_fill=True)
    assert len(result) == len(indices)
```

**Failing input**: `values=[0.0], num_fills=1, fill_value=None`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np
from pandas.api.extensions import take

arr = np.array([10.0, 20.0, 30.0])
series = pd.Series([10.0, 20.0, 30.0])
index = pd.Index([10, 20, 30])

print("numpy array with allow_fill=True:")
result = take(arr, [0, -1], allow_fill=True)
print(f"  {result}")

print("\nSeries with allow_fill=True:")
try:
    result = take(series, [0, -1], allow_fill=True)
    print(f"  {result}")
except TypeError as e:
    print(f"  TypeError: {e}")

print("\nIndex with allow_fill=True, fill_value=None:")
result = take(index, [0, -1], allow_fill=True, fill_value=None)
print(f"  {result}")
print(f"  Second element: {result[1]} (expected NaN, got last element)")

print("\nIndex with allow_fill=True, fill_value=-999:")
try:
    result = take(index, [0, -1], allow_fill=True, fill_value=-999)
    print(f"  {result}")
except ValueError as e:
    print(f"  ValueError: {e}")
```

Output:
```
numpy array with allow_fill=True:
  [10. nan]

Series with allow_fill=True:
  TypeError: take() got an unexpected keyword argument 'allow_fill'

Index with allow_fill=True, fill_value=None:
  Index([10, 30], dtype='int64')
  Second element: 30 (expected NaN, got last element)

Index with allow_fill=True, fill_value=-999:
  ValueError: Unable to fill values because Index cannot contain NA
```

## Why This Is A Bug

The documentation for `pandas.api.extensions.take` explicitly states that it accepts Series as input:

> arr : array-like or scalar value
>     Non array-likes (sequences/scalars without a dtype) are coerced
>     to an ndarray.
>
>     .. deprecated:: 2.1.0
>         Passing an argument other than a numpy.ndarray, ExtensionArray,
>         Index, or Series is deprecated.

However, when `allow_fill=True`, the function delegates to `take_nd`, which has faulty type checking logic. In `take_nd` (pandas/core/array_algos/take.py:104-114):

```python
if not isinstance(arr, np.ndarray):
    # i.e. ExtensionArray,
    # includes for EA to catch DatetimeArray, TimedeltaArray
    if not is_1d_only_ea_dtype(arr.dtype):
        # i.e. DatetimeArray, TimedeltaArray
        arr = cast("NDArrayBackedExtensionArray", arr)
        return arr.take(
            indexer, fill_value=fill_value, allow_fill=allow_fill, axis=axis
        )

    return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill)
```

When `arr` is a Series with a numpy dtype (like float64):
1. `isinstance(arr, np.ndarray)` is False
2. `is_1d_only_ea_dtype(arr.dtype)` is False (because float64 is a numpy dtype)
3. The code casts to NDArrayBackedExtensionArray (which is just a type hint, not a runtime conversion)
4. It calls `arr.take()` which is `Series.take()`, but Series.take() doesn't accept `allow_fill` or `fill_value` parameters

**For Index**, the behavior is even worse:
- Index.take() does accept `allow_fill` and `fill_value` parameters
- But when `fill_value=None`, Index silently ignores the `allow_fill` parameter and treats -1 as a regular negative index
- When `fill_value` is not None, Index raises ValueError saying it cannot contain NA
- This creates three different behaviors (crash for Series, silent ignore for Index with None, error for Index with value)

## Fix

The fix should check for Series/Index/DataFrame and extract the underlying array before processing:

```diff
--- a/pandas/core/array_algos/take.py
+++ b/pandas/core/array_algos/take.py
@@ -95,6 +95,13 @@ def take_nd(
     if fill_value is lib.no_default:
         fill_value = na_value_for_dtype(arr.dtype, compat=False)
     elif lib.is_np_dtype(arr.dtype, "mM"):
         dtype, fill_value = maybe_promote(arr.dtype, fill_value)
         if arr.dtype != dtype:
             # EA.take is strict about returning a new object of the same type
             # so for that case cast upfront
             arr = arr.astype(dtype)
+
+    # Extract underlying array from Series/Index/DataFrame
+    from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCSeries
+    if isinstance(arr, (ABCSeries, ABCIndex, ABCDataFrame)):
+        arr = arr._values

     if not isinstance(arr, np.ndarray):
```