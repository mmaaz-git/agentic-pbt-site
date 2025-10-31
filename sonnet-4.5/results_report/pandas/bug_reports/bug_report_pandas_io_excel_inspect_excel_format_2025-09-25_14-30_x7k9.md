# Bug Report: pandas.io.excel.inspect_excel_format Empty Stream Handling

**Target**: `pandas.io.excel._base.inspect_excel_format`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `inspect_excel_format` function's docstring claims it raises `ValueError` when the input stream is empty, but it actually returns `None` instead. This violates the documented contract and could cause issues for callers expecting an exception.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.excel._base import inspect_excel_format
import pytest


def test_inspect_excel_format_empty_raises():
    with pytest.raises(ValueError, match="stream is empty"):
        inspect_excel_format(b'')
```

**Failing input**: `b''` (empty bytes)

## Reproducing the Bug

```python
from pandas.io.excel._base import inspect_excel_format

result = inspect_excel_format(b'')
print(f"Result: {result}")
print(f"Expected: ValueError to be raised")
```

**Output:**
```
Result: None
Expected: ValueError to be raised
```

## Why This Is A Bug

The function's docstring explicitly states:

```
Raises
------
ValueError
    If resulting stream is empty.
```

However, when called with empty bytes, the function returns `None` instead of raising `ValueError`. This happens because the code checks `if buf is None`, but `stream.read()` returns empty bytes `b''` (not `None`) when reading from an empty stream.

## Fix

```diff
--- a/pandas/io/excel/_base.py
+++ b/pandas/io/excel/_base.py
@@ -1405,7 +1405,7 @@ def inspect_excel_format(
         stream = handle.handle
         stream.seek(0)
         buf = stream.read(PEEK_SIZE)
-        if buf is None:
+        if not buf:
             raise ValueError("stream is empty")
         assert isinstance(buf, bytes)
         peek = buf
```

Alternatively, use `if len(buf) == 0:` for more explicit checking.