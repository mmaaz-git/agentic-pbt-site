# Bug Report: pandas.io.excel.ExcelWriter.close() Not Idempotent

**Target**: `pandas.io.excel.ExcelWriter.close()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Calling `ExcelWriter.close()` twice raises `ValueError: I/O operation on closed file`, violating the common expectation that close methods should be idempotent.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import tempfile

def test_close_idempotent():
    """Calling close() multiple times should not raise errors"""
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        writer = pd.ExcelWriter(tmp_path, engine='openpyxl')
        df = pd.DataFrame({'A': [1, 2, 3]})
        df.to_excel(writer, sheet_name='Sheet1', index=False)

        writer.close()
        writer.close()  # Should not raise
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

**Failing input**: Calling `close()` twice on any `ExcelWriter` instance

## Reproducing the Bug

```python
import pandas as pd
import tempfile
import os

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    writer = pd.ExcelWriter(tmp_path, engine='openpyxl')
    df = pd.DataFrame({'A': [1, 2, 3]})
    df.to_excel(writer, sheet_name='Sheet1', index=False)

    writer.close()
    writer.close()
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
```

Output:
```
ValueError: I/O operation on closed file
```

## Why This Is A Bug

The `close()` method should be idempotent - a fundamental design principle for resource cleanup methods. This is important because:

1. It's a common pattern to call `close()` in `finally` blocks or error handlers, where it might be called multiple times
2. The context manager protocol calls `close()` in `__exit__`, and defensive code might also call it explicitly
3. Many other file-like objects in Python (including standard library file objects) handle multiple `close()` calls gracefully
4. The docstring describes `close()` as "synonym for save, to make it more file-like" - but it doesn't behave like a file

## Fix

The `close()` method should track whether it has already been called and return early on subsequent calls:

```diff
--- a/pandas/io/excel/_base.py
+++ b/pandas/io/excel/_base.py
@@ -1354,6 +1354,9 @@ class ExcelWriter(Generic[_WorkbookT]):
     def close(self) -> None:
         """synonym for save, to make it more file-like"""
+        if self._handles.handle.closed:
+            return
         self._save()
         self._handles.close()
```

Note: The exact implementation may need to check additional state or handle different writer engines differently, but the principle is to make `close()` idempotent by tracking closed state.