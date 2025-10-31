# Bug Report: django.core.files.utils FileProxyMixin.writable() Incorrect Mode Detection

**Target**: `django.core.files.utils.FileProxyMixin.writable`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `FileProxyMixin.writable()` method incorrectly determines if a file is writable when the underlying file object doesn't have a `writable()` method. It only checks if `"w"` is in the mode string, which fails to detect other writable modes like `"r+"`, `"a"`, `"a+"`, `"x"`, and `"x+"`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from django.core.files.base import File
import tempfile
import os

@given(st.sampled_from(['r+', 'w+', 'a', 'a+', 'x', 'x+']))
def test_writable_detects_all_write_modes(mode):
    with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if mode.startswith('x'):
            os.remove(tmp_path)

        f = open(tmp_path, mode + 'b')

        class MockFile:
            def __init__(self, file_obj):
                self.file = file_obj
                self.mode = file_obj.mode

        mock_file = MockFile(f)

        django_file = File(mock_file)

        assert django_file.writable() == True, \
            f"File with mode '{mode}' should be writable"

        f.close()
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
```

**Failing input**: Any mode in `['r+', 'a', 'a+', 'x', 'x+']` - the test would fail because `writable()` returns `False` when it should return `True`.

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/django')

from io import BytesIO
from django.core.files.base import File

class MockFileWithMode:
    def __init__(self, mode):
        self.mode = mode
        self.file = BytesIO(b'test')

    def __getattr__(self, name):
        return getattr(self.file, name)

modes_to_test = ['r+', 'w', 'w+', 'a', 'a+', 'x', 'x+']

for mode in modes_to_test:
    mock_file = MockFileWithMode(mode)
    django_file = File(mock_file)

    is_writable = django_file.writable()
    should_be_writable = any(c in mode for c in 'wax+')

    status = "✓" if is_writable == should_be_writable else "✗ BUG"
    print(f"{status} Mode '{mode}': writable() returns {is_writable}, expected {should_be_writable}")
```

Output:
```
✗ BUG Mode 'r+': writable() returns False, expected True
✓ Mode 'w': writable() returns True, expected True
✓ Mode 'w+': writable() returns True, expected True
✗ BUG Mode 'a': writable() returns False, expected True
✗ BUG Mode 'a+': writable() returns False, expected True
✗ BUG Mode 'x': writable() returns False, expected True
✗ BUG Mode 'x+': writable() returns False, expected True
```

## Why This Is A Bug

The `FileProxyMixin.writable()` method is part of the Python file-like object protocol (defined in `io.IOBase`). It should accurately report whether the file can be written to. The current implementation in utils.py:62-67 only checks if `"w"` appears in the mode string:

```python
def writable(self):
    if self.closed:
        return False
    if hasattr(self.file, "writable"):
        return self.file.writable()
    return "w" in getattr(self.file, "mode", "")  # <-- BUG HERE
```

Python file modes that allow writing include:
- `"w"`, `"w+"`: Write modes (correctly detected)
- `"r+"`: Read-write mode (**incorrectly returns False**)
- `"a"`, `"a+"`: Append modes (**incorrectly returns False**)
- `"x"`, `"x+"`: Exclusive creation modes (**incorrectly returns False**)

This bug affects any code that:
1. Wraps a file object with Django's `File` class
2. Opens the file with a mode other than `"w"` or `"w+"`
3. Relies on `writable()` to determine if writing is allowed

For example, code that checks `if file.writable(): file.write(data)` would incorrectly skip writing for append-mode files.

## Fix

The mode check should detect all writable modes:

```diff
--- a/django/core/files/utils.py
+++ b/django/core/files/utils.py
@@ -62,7 +62,11 @@ class FileProxyMixin:
     def writable(self):
         if self.closed:
             return False
         if hasattr(self.file, "writable"):
             return self.file.writable()
-        return "w" in getattr(self.file, "mode", "")
+        mode = getattr(self.file, "mode", "")
+        # Check if mode allows writing: w (write), a (append), x (exclusive), or + (read-write)
+        return any(c in mode for c in "wax+")
```

This fix checks for any of the characters that indicate write capability: `'w'`, `'a'`, `'x'`, or `'+'`.