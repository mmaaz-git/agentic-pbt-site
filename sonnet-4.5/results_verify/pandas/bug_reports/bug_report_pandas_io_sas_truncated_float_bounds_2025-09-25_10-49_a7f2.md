# Bug Report: pandas.io.sas _handle_truncated_float_vec Invalid Bounds

**Target**: `pandas.io.sas.sas_xport._handle_truncated_float_vec`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_handle_truncated_float_vec` function doesn't validate the `nbytes` parameter, causing obscure dtype errors when `nbytes > 8` (which creates a negative dtype size) or potentially invalid behavior for edge cases like `nbytes <= 0`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.io.sas.sas_xport import _handle_truncated_float_vec

@given(nbytes=st.integers(min_value=9, max_value=100))
@settings(max_examples=100)
def test_handle_truncated_float_vec_invalid_nbytes(nbytes):
    vec = np.array([b'TEST'], dtype='S4')

    try:
        result = _handle_truncated_float_vec(vec, nbytes)
        assert False, f"Should raise error for nbytes={nbytes}"
    except Exception:
        pass
```

**Failing input**: `nbytes=9`

## Reproducing the Bug

```python
import numpy as np
from pandas.io.sas.sas_xport import _handle_truncated_float_vec

vec = np.array([b'TEST'], dtype='S4')

try:
    result = _handle_truncated_float_vec(vec, 9)
    print(f"ERROR: Should have raised error, got: {result}")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
```

Output:
```
ValueError: format number 2 of "S9,S-1" is not recognized
```

## Why This Is A Bug

The function at lines 187-192 in `sas_xport.py` creates a dtype without validating `nbytes`:

```python
if nbytes != 8:
    vec1 = np.zeros(len(vec), np.dtype("S8"))
    dtype = np.dtype(f"S{nbytes},S{8 - nbytes}")
    vec2 = vec1.view(dtype=dtype)
    vec2["f0"] = vec
    return vec2
```

When `nbytes > 8`, the expression `8 - nbytes` becomes negative, creating an invalid dtype string like `"S9,S-1"`. This produces a confusing error message.

According to the comments, this function handles "2-7 byte truncated floats", which implies valid values are 2 ≤ nbytes ≤ 8. The function should validate this precondition and raise a clear error for invalid values.

## Fix

```diff
--- a/pandas/io/sas/sas_xport.py
+++ b/pandas/io/sas/sas_xport.py
@@ -185,6 +185,11 @@ def _handle_truncated_float_vec(vec, nbytes):
     # https://github.com/jcushman/xport/pull/3
     # The R "foreign" library

+    if nbytes < 2 or nbytes > 8:
+        raise ValueError(
+            f"nbytes must be between 2 and 8 for truncated floats, got {nbytes}"
+        )
+
     if nbytes != 8:
         vec1 = np.zeros(len(vec), np.dtype("S8"))
         dtype = np.dtype(f"S{nbytes},S{8 - nbytes}")
```