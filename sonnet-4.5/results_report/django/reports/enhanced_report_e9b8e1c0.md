# Bug Report: django.core.files.base.File.open AttributeError When Reopening Files Without Mode Attribute

**Target**: `django.core.files.base.File.open`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When a Django `File` object wraps a file-like object that lacks a `mode` attribute (such as `BytesIO` or `StringIO`), calling `open()` without an explicit mode parameter crashes with `AttributeError: 'File' object has no attribute 'mode'` instead of handling the missing mode gracefully.

## Property-Based Test

```python
from django.core.files.base import File
from io import BytesIO
from hypothesis import given, strategies as st, example
import tempfile
import os


@given(st.binary(min_size=0, max_size=1000))
@example(b'')  # Test with empty content
@example(b'test')  # Test with simple content
def test_file_reopen_without_mode(content):
    """Test that File.open() works with file-like objects that don't have a mode attribute."""
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(content)
        temp_path = tf.name

    try:
        # Create a File object with BytesIO (which doesn't have a 'mode' attribute)
        f = File(BytesIO(content), name=temp_path)
        f.close()

        # This should work but raises AttributeError
        f.open()

        # If we get here, the file was reopened successfully
        assert f.closed is False
        f.close()

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    test_file_reopen_without_mode()
```

<details>

<summary>
**Failing input**: `b''` and `b'test'` (any binary content)
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/33
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_file_reopen_without_mode FAILED                            [100%]

=================================== FAILURES ===================================
________________________ test_file_reopen_without_mode _________________________
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 9, in test_file_reopen_without_mode
  |     @example(b'')  # Test with empty content
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures in explicit examples. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 23, in test_file_reopen_without_mode
    |     f.open()
    |     ~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/files/base.py", line 112, in open
    |     self.file = open(self.name, mode or self.mode, *args, **kwargs)
    |                                         ^^^^^^^^^
    | AttributeError: 'File' object has no attribute 'mode'
    | Falsifying explicit example: test_file_reopen_without_mode(
    |     content=b'',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 23, in test_file_reopen_without_mode
    |     f.open()
    |     ~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/files/base.py", line 112, in open
    |     self.file = open(self.name, mode or self.mode, *args, **kwargs)
    |                                         ^^^^^^^^^
    | AttributeError: 'File' object has no attribute 'mode'
    | Falsifying explicit example: test_file_reopen_without_mode(
    |     content=b'test',
    | )
    +------------------------------------
=========================== short test summary info ============================
FAILED hypo.py::test_file_reopen_without_mode - ExceptionGroup: Hypothesis fo...
============================== 1 failed in 0.13s ===============================
```
</details>

## Reproducing the Bug

```python
from django.core.files.base import File
from io import BytesIO
import tempfile
import os

# Create a temporary file to have a valid file path
with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tf:
    tf.write(b'test content')
    temp_path = tf.name

try:
    # Create a File object with BytesIO (which doesn't have a 'mode' attribute)
    f = File(BytesIO(b'initial'), name=temp_path)

    # Close the file
    f.close()

    # Try to reopen without specifying mode - this should trigger the bug
    f.open()

    print("File reopened successfully")

finally:
    # Clean up the temporary file
    if os.path.exists(temp_path):
        os.unlink(temp_path)
```

<details>

<summary>
AttributeError: 'File' object has no attribute 'mode'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/repo.py", line 19, in <module>
    f.open()
    ~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/files/base.py", line 112, in open
    self.file = open(self.name, mode or self.mode, *args, **kwargs)
                                        ^^^^^^^^^
AttributeError: 'File' object has no attribute 'mode'
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Incomplete attribute initialization**: The `File.__init__` method (lines 16-17 in base.py) conditionally sets `self.mode` only if the wrapped file object has a `mode` attribute. However, the `open()` method (line 112) unconditionally accesses `self.mode` when `mode` parameter is `None`, causing an `AttributeError`.

2. **Documentation mismatch**: The Django documentation states that `open(mode=None)` means "reopen with the original mode", but doesn't specify what happens when there is no original mode to use. The documentation doesn't warn that File objects require the underlying file to have a mode attribute.

3. **Poor error messaging**: The error `AttributeError: 'File' object has no attribute 'mode'` doesn't help users understand that they need to provide an explicit mode parameter when working with mode-less file objects.

4. **Common use case failure**: Standard Python file-like objects such as `BytesIO` and `StringIO` don't have `mode` attributes. These are commonly used for in-memory operations, testing, and data processing. Django's File wrapper should handle these gracefully.

5. **Inconsistent behavior**: The File class accepts these file-like objects in its constructor without complaint, but then fails when trying to use core functionality like `open()`.

## Relevant Context

The problematic code is in `/django/core/files/base.py`:

- **Lines 16-17**: Conditional mode setting in `__init__`:
  ```python
  if hasattr(file, "mode"):
      self.mode = file.mode
  ```

- **Line 112**: Unconditional mode access in `open()`:
  ```python
  self.file = open(self.name, mode or self.mode, *args, **kwargs)
  ```

This issue is similar to historical Django tickets:
- Ticket #26469: FieldFile.open() not properly setting mode when opening file
- Ticket #13809: FileField open method issues with modes

Django documentation for File.open(): https://docs.djangoproject.com/en/stable/ref/files/file/#django.core.files.File.open

The workaround is to always provide an explicit mode when calling `open()` on File objects created from mode-less file objects: `f.open('rb')` instead of `f.open()`.

## Proposed Fix

```diff
--- a/django/core/files/base.py
+++ b/django/core/files/base.py
@@ -109,7 +109,11 @@ class File(FileProxyMixin):
         if not self.closed:
             self.seek(0)
         elif self.name and os.path.exists(self.name):
-            self.file = open(self.name, mode or self.mode, *args, **kwargs)
+            if mode is None:
+                mode = getattr(self, 'mode', 'rb')
+            self.file = open(self.name, mode, *args, **kwargs)
         else:
             raise ValueError("The file cannot be reopened.")
         return self
```