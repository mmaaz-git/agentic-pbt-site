# Bug Report: django.core.files.utils FileProxyMixin.writable() Fails to Detect Non-'w' Writable Modes

**Target**: `django.core.files.utils.FileProxyMixin.writable`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `FileProxyMixin.writable()` method incorrectly returns `False` for writable file modes that don't contain the letter 'w', including append modes ('a', 'a+'), read-write mode ('r+'), and exclusive creation modes ('x', 'x+').

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
from django.core.files.base import File
import tempfile
import os

@given(st.sampled_from(['r+', 'w+', 'a', 'a+', 'x', 'x+']))
def test_writable_detects_all_write_modes(mode):
    """Test that FileProxyMixin.writable() correctly detects all writable file modes."""

    # Create a mock file object without a writable() method to force fallback to mode checking
    class MockFileWithMode:
        def __init__(self, mode):
            self.mode = mode
            self.closed = False
            # Provide minimal file-like interface but no writable() method
            self.read = lambda: b''
            self.write = lambda x: None
            self.seek = lambda x: None
            self.tell = lambda: 0

    mock_file = MockFileWithMode(mode)
    django_file = File(mock_file)

    # All these modes should be writable according to Python's file mode specification
    assert django_file.writable() == True, \
        f"File with mode '{mode}' should be writable"

if __name__ == "__main__":
    test_writable_detects_all_write_modes()
```

<details>

<summary>
**Failing input**: `mode='r+'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 32, in <module>
    test_writable_detects_all_write_modes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 10, in test_writable_detects_all_write_modes
    def test_writable_detects_all_write_modes(mode):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 28, in test_writable_detects_all_write_modes
    assert django_file.writable() == True, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: File with mode 'r+' should be writable
Falsifying example: test_writable_detects_all_write_modes(
    mode='r+',
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from io import BytesIO
from django.core.files.base import File

class MockFileWithMode:
    def __init__(self, mode):
        self.mode = mode
        self._io = BytesIO(b'test')
        # Store necessary attributes from BytesIO but not writable()
        self.read = self._io.read
        self.write = self._io.write
        self.seek = self._io.seek
        self.tell = self._io.tell
        self.closed = False

modes_to_test = ['r', 'r+', 'w', 'w+', 'a', 'a+', 'x', 'x+', 'rb', 'r+b', 'wb', 'w+b', 'ab', 'a+b', 'xb', 'x+b']

print("Testing Django's FileProxyMixin.writable() method:")
print("=" * 60)

for mode in modes_to_test:
    mock_file = MockFileWithMode(mode)
    django_file = File(mock_file)

    is_writable = django_file.writable()

    # Determine expected result based on Python's file mode specification
    # A file is writable if it has any of: 'w', 'a', 'x', or '+' in its mode
    should_be_writable = any(c in mode for c in 'wax+')

    if is_writable == should_be_writable:
        status = "✓ PASS"
    else:
        status = "✗ FAIL"

    print(f"{status} Mode '{mode:4s}': writable()={str(is_writable):5s}, expected={str(should_be_writable):5s}")

print("\n" + "=" * 60)
print("Summary of failures:")
print("-" * 60)

failure_count = 0
for mode in modes_to_test:
    mock_file = MockFileWithMode(mode)
    django_file = File(mock_file)
    is_writable = django_file.writable()
    should_be_writable = any(c in mode for c in 'wax+')

    if is_writable != should_be_writable:
        failure_count += 1
        print(f"Mode '{mode}': Expected writable()={should_be_writable}, but got {is_writable}")

if failure_count == 0:
    print("No failures detected!")
else:
    print(f"\nTotal failures: {failure_count}")
```

<details>

<summary>
10 file modes incorrectly identified as non-writable
</summary>
```
Testing Django's FileProxyMixin.writable() method:
============================================================
✓ PASS Mode 'r   ': writable()=False, expected=False
✗ FAIL Mode 'r+  ': writable()=False, expected=True
✓ PASS Mode 'w   ': writable()=True , expected=True
✓ PASS Mode 'w+  ': writable()=True , expected=True
✗ FAIL Mode 'a   ': writable()=False, expected=True
✗ FAIL Mode 'a+  ': writable()=False, expected=True
✗ FAIL Mode 'x   ': writable()=False, expected=True
✗ FAIL Mode 'x+  ': writable()=False, expected=True
✓ PASS Mode 'rb  ': writable()=False, expected=False
✗ FAIL Mode 'r+b ': writable()=False, expected=True
✓ PASS Mode 'wb  ': writable()=True , expected=True
✓ PASS Mode 'w+b ': writable()=True , expected=True
✗ FAIL Mode 'ab  ': writable()=False, expected=True
✗ FAIL Mode 'a+b ': writable()=False, expected=True
✗ FAIL Mode 'xb  ': writable()=False, expected=True
✗ FAIL Mode 'x+b ': writable()=False, expected=True

============================================================
Summary of failures:
------------------------------------------------------------
Mode 'r+': Expected writable()=True, but got False
Mode 'a': Expected writable()=True, but got False
Mode 'a+': Expected writable()=True, but got False
Mode 'x': Expected writable()=True, but got False
Mode 'x+': Expected writable()=True, but got False
Mode 'r+b': Expected writable()=True, but got False
Mode 'ab': Expected writable()=True, but got False
Mode 'a+b': Expected writable()=True, but got False
Mode 'xb': Expected writable()=True, but got False
Mode 'x+b': Expected writable()=True, but got False

Total failures: 10
```
</details>

## Why This Is A Bug

The `FileProxyMixin.writable()` method is part of the Python file-like object protocol defined in `io.IOBase`. According to Python's documentation, `writable()` should "return whether object was opened for writing" and if it returns `False`, then `write()` will raise `OSError`.

The current implementation in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/files/utils.py:62-67` has a fallback logic that only checks if `"w"` is in the mode string:

```python
def writable(self):
    if self.closed:
        return False
    if hasattr(self.file, "writable"):
        return self.file.writable()
    return "w" in getattr(self.file, "mode", "")  # <-- BUG: Only checks for 'w'
```

However, Python's file modes that allow writing include:
- **'w', 'w+'**: Write modes (correctly detected ✓)
- **'r+'**: Read and write mode without truncation (incorrectly returns False ✗)
- **'a', 'a+'**: Append modes that write at end of file (incorrectly returns False ✗)
- **'x', 'x+'**: Exclusive creation modes (incorrectly returns False ✗)

The '+' modifier adds the complementary access mode, making 'r+' writable. This violates the contract of `io.IOBase.writable()` and breaks the principle of least surprise - users expect Django's file proxy to behave identically to Python's standard file objects.

## Relevant Context

This bug manifests when:
1. A file object is wrapped with Django's `File` class (which inherits from `FileProxyMixin`)
2. The wrapped file object doesn't have its own `writable()` method
3. The file was opened with a writable mode other than 'w' or 'w+'
4. Code checks `writable()` before attempting to write

For example, code like this would fail incorrectly:
```python
# Open a file for appending
with open('log.txt', 'a') as f:
    django_file = File(f)
    if django_file.writable():  # Returns False incorrectly!
        django_file.write(b'data')
```

The bug is in the fallback logic at line 67 of `django/core/files/utils.py`. While most modern Python file objects have a `writable()` method, the fallback is still important for custom file-like objects and should correctly implement the Python file protocol.

## Proposed Fix

```diff
--- a/django/core/files/utils.py
+++ b/django/core/files/utils.py
@@ -64,7 +64,8 @@ class FileProxyMixin:
         return False
     if hasattr(self.file, "writable"):
         return self.file.writable()
-    return "w" in getattr(self.file, "mode", "")
+    mode = getattr(self.file, "mode", "")
+    return any(c in mode for c in "wax+")

 def seekable(self):
     if self.closed:
```