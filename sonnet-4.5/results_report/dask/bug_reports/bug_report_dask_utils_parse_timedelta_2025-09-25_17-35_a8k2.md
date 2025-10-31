# Bug Report: dask.utils.parse_timedelta Crashes on Empty String

**Target**: `dask.utils.parse_timedelta`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`parse_timedelta` crashes with `IndexError: string index out of range` when given an empty string or space-only string, instead of raising an informative `ValueError`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import dask.utils
import pytest


@given(st.just(''))
def test_parse_timedelta_empty_string(s):
    with pytest.raises(ValueError):
        dask.utils.parse_timedelta(s)
```

**Failing input**: `''` or `' '`

## Reproducing the Bug

```python
import dask.utils

dask.utils.parse_timedelta('')
```

Output:
```
IndexError: string index out of range
```

## Why This Is A Bug

Functions should raise informative errors, not crash with IndexError. Other parsing functions in dask raise `ValueError` for invalid input. Additionally, like `parse_bytes`, it also silently accepts whitespace-only strings like `'\r'` and returns `1` instead of raising an error.

The crash occurs because after removing spaces with `s.replace(" ", "")`, the code accesses `s[0]` without checking if the string is empty:

```python
s = s.replace(" ", "")
if not s[0].isdigit():  # IndexError if s is empty!
    s = "1" + s
```

## Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1685,6 +1685,8 @@ def parse_timedelta(s, default="seconds"):
     if isinstance(s, Number):
         s = str(s)
     s = s.replace(" ", "")
+    if not s or not s.strip():
+        raise ValueError(f"Could not interpret {s!r} as a time delta")
     if not s[0].isdigit():
         s = "1" + s
```