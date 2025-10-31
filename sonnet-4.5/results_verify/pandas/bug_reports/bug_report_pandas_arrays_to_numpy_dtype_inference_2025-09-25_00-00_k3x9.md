# Bug Report: pandas.core.arrays._utils.to_numpy_dtype_inference Incorrect dtype_given Logic

**Target**: `pandas.core.arrays._utils.to_numpy_dtype_inference`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_numpy_dtype_inference` function incorrectly sets `dtype_given = True` when `dtype=None` for non-numeric array types (e.g., string arrays), preventing proper dtype inference and returning `None` as the dtype.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas._libs import lib


@given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=100))
def test_to_numpy_dtype_inference_returns_dtype_for_string_arrays(values):
    arr = pd.array(values, dtype="string")

    dtype, na_value = to_numpy_dtype_inference(arr, None, lib.no_default, False)

    assert dtype is not None, "dtype should not be None for string arrays"
```

**Failing input**: `values=['0']` (or any string array)

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas._libs import lib

arr = pd.array(['a', 'b', 'c'], dtype="string")

dtype, na_value = to_numpy_dtype_inference(arr, None, lib.no_default, False)

print(f"dtype: {dtype}")
print(f"na_value: {na_value}")

assert dtype is None
```

## Why This Is A Bug

The function has three branches for handling dtype inference:

1. Lines 26-39: When `dtype is None and is_numeric_dtype(arr.dtype)`, it sets `dtype_given = False` and infers the dtype.
2. Lines 40-42: When `dtype is not None`, it converts it to `np.dtype` and sets `dtype_given = True`.
3. Lines 43-44: The `else` clause (when `dtype is None` and array is non-numeric) sets `dtype_given = True`.

The bug is in line 44. When `dtype=None` is passed and the array is non-numeric (e.g., string), the code sets `dtype_given = True`, which is logically incorrect because dtype was NOT given (it's `None`). This should be `dtype_given = False`.

The consequence is that:
- The variable `dtype` remains `None`
- The fallback logic on lines 58-62 (which would upgrade dtype to object if needed) never runs because `dtype_given` is incorrectly `True`
- Callers receive `None` as the dtype and must handle it manually

## Fix

```diff
--- a/pandas/core/arrays/_utils.py
+++ b/pandas/core/arrays/_utils.py
@@ -41,7 +41,7 @@ def to_numpy_dtype_inference(
         dtype = np.dtype(dtype)
         dtype_given = True
     else:
-        dtype_given = True
+        dtype_given = False

     if na_value is lib.no_default:
         if dtype is None or not hasna:
```