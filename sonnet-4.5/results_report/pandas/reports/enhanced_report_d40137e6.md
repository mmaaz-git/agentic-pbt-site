# Bug Report: pandas.ExcelWriter.close() Violates File-Like Interface Contract

**Target**: `pandas.io.excel.ExcelWriter.close()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ExcelWriter.close()` method raises `ValueError: I/O operation on closed file` when called twice, violating Python's standard file-like object contract where `close()` should be idempotent.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import tempfile
import os

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
        print("Test passed: close() is idempotent")
    except Exception as e:
        print(f"Test failed: {type(e).__name__}: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    test_close_idempotent()
```

<details>

<summary>
**Failing input**: Calling `close()` twice on any `ExcelWriter` instance
</summary>
```
Test failed: ValueError: I/O operation on closed file
```
</details>

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

    print("First close() call:")
    writer.close()
    print("First close() succeeded")

    print("\nSecond close() call:")
    writer.close()  # This should raise an error
    print("Second close() succeeded")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
```

<details>

<summary>
ValueError raised on second close() call
</summary>
```
First close() call:
First close() succeeded

Second close() call:
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/repo.py", line 18, in <module>
    writer.close()  # This should raise an error
    ~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_base.py", line 1357, in close
    self._save()
    ~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_openpyxl.py", line 110, in _save
    self.book.save(self._handles.handle)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/openpyxl/workbook/workbook.py", line 386, in save
    save_workbook(self, filename)
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/openpyxl/writer/excel.py", line 291, in save_workbook
    archive = ZipFile(filename, 'w', ZIP_DEFLATED, allowZip64=True)
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1391, in __init__
    self.start_dir = self.fp.tell()
                     ~~~~~~~~~~~~^^
ValueError: I/O operation on closed file
```
</details>

## Why This Is A Bug

The `ExcelWriter.close()` method violates the fundamental Python contract for file-like objects. According to Python's official documentation (io.IOBase.close()), the close() method for file-like objects must be idempotent: "As a convenience, it is allowed to call this method more than once; only the first call, however, will have an effect."

The bug is particularly problematic because:

1. **The docstring explicitly claims file-like behavior**: The close() method's docstring states it exists as a "synonym for save, to make it more file-like" (pandas/io/excel/_base.py:1356), creating an expectation of standard file-like behavior.

2. **It breaks common defensive programming patterns**: Multiple close() calls commonly occur in:
   - Finally blocks ensuring resource cleanup
   - Error handlers that attempt to close resources
   - Context managers combined with explicit close() calls
   - Code that defensively ensures resources are closed

3. **Inconsistent with Python's file objects**: Standard Python file objects (created with `open()`) handle multiple close() calls gracefully without raising errors.

4. **No documentation warning**: Neither the pandas documentation nor the docstring warns users that close() cannot be called multiple times, leaving developers to discover this non-standard behavior through runtime errors.

## Relevant Context

The issue occurs in pandas 2.3.2 at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_base.py:1355-1358`:

```python
def close(self) -> None:
    """synonym for save, to make it more file-like"""
    self._save()
    self._handles.close()
```

The error is triggered when `self._save()` attempts to save the workbook a second time after the file handle has already been closed by the first `close()` call. The underlying file handle (`self._handles.handle`) is closed but the ExcelWriter instance doesn't track this state.

This affects all ExcelWriter engines (openpyxl, xlsxwriter, etc.) as the issue is in the base class implementation.

## Proposed Fix

```diff
--- a/pandas/io/excel/_base.py
+++ b/pandas/io/excel/_base.py
@@ -1354,6 +1354,9 @@ class ExcelWriter(Generic[_WorkbookT]):

     def close(self) -> None:
         """synonym for save, to make it more file-like"""
+        # Make close() idempotent like standard Python file objects
+        if hasattr(self._handles.handle, 'closed') and self._handles.handle.closed:
+            return
         self._save()
         self._handles.close()
```

Alternative implementation that tracks closed state more explicitly:

```diff
--- a/pandas/io/excel/_base.py
+++ b/pandas/io/excel/_base.py
@@ -1239,6 +1239,7 @@ class ExcelWriter(Generic[_WorkbookT]):
         else:
             # GH 39681 Already have an engine
             engine_kwargs = combine_kwargs(engine_kwargs, kwargs)
+        self._closed = False

         # cast ExcelWriter to avoid adding 'if self._handles is not None'
         self._handles = IOHandles(
@@ -1354,6 +1355,9 @@ class ExcelWriter(Generic[_WorkbookT]):

     def close(self) -> None:
         """synonym for save, to make it more file-like"""
+        if self._closed:
+            return
         self._save()
         self._handles.close()
+        self._closed = True
```