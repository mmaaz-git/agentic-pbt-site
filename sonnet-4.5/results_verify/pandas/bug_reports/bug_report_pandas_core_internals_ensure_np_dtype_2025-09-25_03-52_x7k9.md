# Bug Report: pandas.core.internals.base.ensure_np_dtype Fixed-Length Unicode Strings

**Target**: `pandas.core.internals.base.ensure_np_dtype`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ensure_np_dtype` function fails to convert fixed-length Unicode string dtypes (e.g., `dtype('<U10')`, `dtype('<U100')`) to `object` dtype, while it correctly converts the variable-length Unicode dtype `dtype('<U')` (returned by `np.dtype(str)`). This inconsistency can cause downstream issues in pandas internals that expect all string types to be represented as `object` dtype.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from pandas.core.internals.base import ensure_np_dtype


@given(st.sampled_from([np.dtype(str), np.dtype('U'), np.dtype('U10')]))
def test_ensure_np_dtype_string_to_object(str_dtype):
    result = ensure_np_dtype(str_dtype)
    assert isinstance(result, np.dtype), f"Expected np.dtype, got {type(result)}"
    assert result == np.dtype('object'), f"Expected object dtype, got {result}"
```

**Failing input**: `dtype('<U10')`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.internals.base import ensure_np_dtype

dtype_var_length = np.dtype(str)
dtype_fixed_length = np.dtype('U10')

result_var = ensure_np_dtype(dtype_var_length)
result_fixed = ensure_np_dtype(dtype_fixed_length)

print(f"Variable-length: {dtype_var_length} -> {result_var}")
print(f"Fixed-length:    {dtype_fixed_length} -> {result_fixed}")

assert result_var == np.dtype('object')
assert result_fixed == np.dtype('object')
```

Output:
```
Variable-length: <U -> object
Fixed-length:    <U10 -> <U10
AssertionError: ...
```

## Why This Is A Bug

The function is designed to normalize dtypes for internal pandas operations, converting types that shouldn't be used directly in numpy arrays (ExtensionDtypes, string dtypes) to `object` dtype. Lines 405-406 of `base.py` show the intent:

```python
elif dtype == np.dtype(str):
    dtype = np.dtype("object")
```

However, this check only matches `dtype('<U')` (variable-length Unicode), not fixed-length variants like `dtype('<U10')`, `dtype('<U100')`, etc. All Unicode string dtypes share `kind='U'` and should be treated identically.

This violates the principle of least surprise: users expect all string dtypes to be handled consistently, regardless of whether they specify a fixed length.

## Fix

```diff
--- a/pandas/core/internals/base.py
+++ b/pandas/core/internals/base.py
@@ -402,7 +402,7 @@ def ensure_np_dtype(dtype: DtypeObj) -> np.dtype:
         dtype = cast(np.dtype, dtype)
     elif isinstance(dtype, ExtensionDtype):
         dtype = np.dtype("object")
-    elif dtype == np.dtype(str):
+    elif dtype.kind == "U":
         dtype = np.dtype("object")
     return dtype
```

The fix changes the equality check `dtype == np.dtype(str)` to a kind check `dtype.kind == "U"`, which catches all Unicode string dtypes regardless of their fixed or variable length specification.