# Bug Report: pandas.api.types.infer_dtype Crashes on Python Numeric Scalars

**Target**: `pandas.api.types.infer_dtype`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `infer_dtype` function is documented to accept "scalar" values but crashes with `TypeError: 'X' object is not iterable` when given Python built-in numeric scalars (int, float, bool, complex, None), while working correctly for string scalars and NumPy scalars.

## Property-Based Test

```python
import pandas.api.types as types
from hypothesis import given, strategies as st


@given(st.one_of(
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
))
def test_infer_dtype_accepts_scalars(val):
    result_scalar = types.infer_dtype(val, skipna=False)
    result_list = types.infer_dtype([val], skipna=False)
    assert result_scalar == result_list
```

**Failing input**: `0`

## Reproducing the Bug

```python
import pandas.api.types as types

print(types.infer_dtype(0, skipna=False))
```

This crashes with:
```
TypeError: 'int' object is not iterable
```

Similarly fails for: `1.5`, `True`, `1+2j`, `None`

However, it works for: `"hello"` (str), `b"bytes"` (bytes), `np.int64(5)`, `np.float64(5.5)`

## Why This Is A Bug

The function's docstring explicitly states:

> Return a string label of the type of **a scalar or list-like of values**.
>
> Parameters
> ----------
> value : **scalar, list, ndarray, or pandas type**

This clearly documents that the function should accept scalar values. However, it only works for some scalars (strings, bytes, NumPy scalars) and crashes on Python built-in numeric scalars. This is inconsistent behavior that violates the documented API contract.

Users reasonably expect that if a function accepts "scalar" it should work for all scalar types, especially common ones like int, float, and bool.

## Fix

The function should handle Python built-in scalars by either:
1. Converting them to appropriate iterables internally before processing, or
2. Detecting scalar types and returning the appropriate type string directly

A minimal fix might look like:

```diff
--- a/pandas/_libs/lib.pyx
+++ b/pandas/_libs/lib.pyx
@@ -1618,6 +1618,10 @@ def infer_dtype(object value, bint skipna=True) -> str:
     """
+    # Handle Python built-in scalars
+    if isinstance(value, (int, float, bool, complex)) and not isinstance(value, (np.generic,)):
+        value = [value]
+
     cdef:
         Py_ssize_t i, n
```

Note: The exact implementation may need to be adjusted based on the Cython codebase structure.