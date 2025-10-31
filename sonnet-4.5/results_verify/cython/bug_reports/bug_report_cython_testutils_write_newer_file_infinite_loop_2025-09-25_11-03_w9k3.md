# Bug Report: Cython.TestUtils.write_newer_file Infinite Loop

**Target**: `Cython.TestUtils.write_newer_file`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`write_newer_file()` enters an infinite loop when the `newer_than` file doesn't exist, violating its documented behavior and causing hangs in test suites.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import tempfile
import os


@given(st.text(min_size=0, max_size=100))
def test_write_newer_file_terminates(content):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.txt")
        nonexistent = os.path.join(tmpdir, "nonexistent.txt")

        write_newer_file(file_path, nonexistent, content)

        assert os.path.exists(file_path)
        with open(file_path) as f:
            assert f.read() == content
```

**Failing input**: Any content string when `newer_than` file doesn't exist

## Reproducing the Bug

```python
import sys
import os
import tempfile

sys.path.insert(0, '/path/to/cython/site-packages')
from Cython.TestUtils import write_newer_file

with tempfile.TemporaryDirectory() as tmpdir:
    file_path = os.path.join(tmpdir, "test.txt")
    nonexistent = os.path.join(tmpdir, "nonexistent.txt")

    write_newer_file(file_path, nonexistent, "test content")
```

**Result**: Function hangs indefinitely, never returns.

## Why This Is A Bug

The docstring states the function should "make sure [the file] is newer than the file `newer_than`". The comment on line 385 says "Support writing a fresh file (which is always newer than a non-existent one)".

However, when `newer_than` doesn't exist:
1. Line 386 sets `other_time = None`
2. Line 388's while condition: `other_time is None or other_time >= os.path.getmtime(file_path)`
3. Since `other_time is None` is True, the loop always executes
4. `other_time` is never updated in the loop
5. Infinite loop

A newly created file is always newer than a non-existent file, so the function should immediately return after the first write (line 380).

## Fix

```diff
--- a/Cython/TestUtils.py
+++ b/Cython/TestUtils.py
@@ -382,10 +382,12 @@ def write_newer_file(file_path, newer_than, content, dedent=False, encoding=Non
     try:
         other_time = os.path.getmtime(newer_than)
     except OSError:
-        # Support writing a fresh file (which is always newer than a non-existent one)
-        other_time = None
+        # Support writing a fresh file (which is always newer than a non-existent one).
+        # The file was already written on line 380, so we're done.
+        return

-    while other_time is None or other_time >= os.path.getmtime(file_path):
+    # Rewrite until file_path is newer than newer_than
+    while other_time >= os.path.getmtime(file_path):
         write_file(file_path, content, dedent=dedent, encoding=encoding)
```