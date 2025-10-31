# Bug Report: pandas.core.internals.concat Complex dtype NaN

**Target**: `pandas.core.internals.concat._dtype_to_na_value`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_dtype_to_na_value` function crashes when called with complex dtypes (complex64, complex128) because it attempts to construct a complex number from the string "NaN", which is invalid in Python/NumPy.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.internals.concat import _dtype_to_na_value

@given(st.sampled_from([
    np.dtype('float32'), np.dtype('float64'),
    np.dtype('complex64'), np.dtype('complex128')
]), st.booleans())
def test_dtype_to_na_value_fc_kinds(dtype, has_none_blocks):
    result = _dtype_to_na_value(dtype, has_none_blocks)
    assert result is not None
    if dtype.kind == 'f':
        assert np.isnan(result)
    else:
        assert isinstance(result, complex) and np.isnan(result.real)
```

**Failing input**: `dtype=np.dtype('complex128'), has_none_blocks=False`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.internals.concat import _dtype_to_na_value

dtype = np.dtype('complex128')
result = _dtype_to_na_value(dtype, False)
```

This raises:
```
ValueError: complex() arg is a malformed string
```

The root cause is in line 512 of concat.py:
```python
return dtype.type("NaN")
```

For complex dtypes, `np.complex128("NaN")` fails because the complex() constructor cannot parse the string "NaN". It works for float types because `np.float64("NaN")` is valid.

## Why This Is A Bug

The function is designed to return NA values for all floating-point and complex dtypes (kind "fc"), but it crashes for complex types. This violates the function's contract and can cause DataFrame concatenation operations to fail when complex-valued data is involved.

While complex dtypes are less common in pandas, they are supported and should work correctly. The bug can be triggered during DataFrame concatenation when the concat logic needs to fill missing values for complex columns.

## Fix

```diff
--- a/pandas/core/internals/concat.py
+++ b/pandas/core/internals/concat.py
@@ -509,7 +509,7 @@ def _dtype_to_na_value(dtype: DtypeObj, has_none_blocks: bool):
     elif dtype.kind in "mM":
         return dtype.type("NaT")
     elif dtype.kind in "fc":
-        return dtype.type("NaN")
+        return dtype.type(np.nan)
     elif dtype.kind == "b":
         return None
     elif dtype.kind in "iu":
```

The fix changes `dtype.type("NaN")` to `dtype.type(np.nan)`. This works for both float and complex types:
- `np.float64(np.nan)` returns `nan`
- `np.complex128(np.nan)` returns `(nan+0j)`

Both correctly represent NA values for their respective types.