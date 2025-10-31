# Bug Report: dask.utils.parse_bytes Accepts Invalid Whitespace-Only Input

**Target**: `dask.utils.parse_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`parse_bytes` silently accepts empty strings and whitespace-only strings (like `'\r'`, `'\n'`, `'\t'`), returning `1` instead of raising a `ValueError` as it does for other invalid inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import dask.utils
import pytest


@given(st.text())
def test_parse_bytes_rejects_whitespace_only(s):
    assume(s.strip() == '')
    assume(s != '')

    with pytest.raises(ValueError):
        dask.utils.parse_bytes(s)
```

**Failing input**: `'\r'` (or `'\n'`, `'\t'`, `' '`, `''`, etc.)

## Reproducing the Bug

```python
import dask.utils

result = dask.utils.parse_bytes('')
print(result)

result = dask.utils.parse_bytes('\r')
print(result)

result = dask.utils.parse_bytes('\n')
print(result)
```

## Why This Is A Bug

The function's docstring shows it raises `ValueError` for invalid inputs like `'5 foos'`. Whitespace-only and empty strings are similarly invalid and should raise errors rather than silently returning `1`. This violates the principle of least surprise and could mask user errors.

The bug occurs because:
1. When no digits are found, the code prepends `'1'` to the string
2. Python's `float()` accepts and strips trailing whitespace
3. The `byte_sizes` dict maps empty string `''` to `1`

So `parse_bytes('\r')` becomes `'1\r'`, then `float('1\r')` returns `1.0`, and `byte_sizes['']` returns `1`.

## Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1634,6 +1634,8 @@ def parse_bytes(s: float | str) -> int:
     """
     if isinstance(s, (int, float)):
         return int(s)
+    if not s or not s.strip():
+        raise ValueError(f"Could not interpret {s!r} as a byte string")
     s = s.replace(" ", "")
     if not any(char.isdigit() for char in s):
         s = "1" + s
```