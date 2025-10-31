# Bug Report: Django UploadedFile Backslash Path Separator Not Sanitized

**Target**: `django.core.files.uploadedfile.UploadedFile`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `UploadedFile.name` setter sanitizes file names to prevent path traversal attacks by using `os.path.basename()`, but this only removes forward slashes on Unix systems. Backslashes (Windows path separators) are not removed, creating a potential security vulnerability if uploaded files are later processed on Windows or in cross-platform contexts.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.files.uploadedfile import SimpleUploadedFile

@given(st.text(min_size=1), st.binary())
def test_uploaded_file_name_sanitization(name, content):
    assume(name not in {'', '.', '..'})
    f = SimpleUploadedFile(name, content)
    assert '/' not in f.name
    assert '\\' not in f.name  # This fails!
```

**Failing input**: `name='\\', content=b''`

## Reproducing the Bug

```python
from django.core.files.uploadedfile import SimpleUploadedFile

f = SimpleUploadedFile('test\\file.txt', b'content')
print(f.name)

assert f.name == 'test\\file.txt'
```

## Why This Is A Bug

The code in `uploadedfile.py` lines 54-55 states:
```python
# Just use the basename of the file -- anything else is dangerous.
name = os.path.basename(name)
```

However, `os.path.basename()` on Unix systems treats backslashes as regular characters, not path separators. This means:

1. A file uploaded with name `malicious\\..\\..\\etc\\passwd` would have its forward slashes removed but backslashes preserved
2. If this file is later processed on Windows (e.g., in a cross-platform application or when files are synced to Windows storage), the backslashes become active path separators
3. This creates an inconsistency: forward slashes are sanitized but backslashes are not

The comment explicitly says "anything else is dangerous", but the implementation doesn't match this intent for backslashes.

## Fix

The fix should normalize backslashes to forward slashes before calling `os.path.basename()`, consistent with how `validate_file_name()` handles the `allow_relative_path=True` case:

```diff
diff --git a/django/core/files/uploadedfile.py b/django/core/files/uploadedfile.py
index 1234567..abcdefg 100644
--- a/django/core/files/uploadedfile.py
+++ b/django/core/files/uploadedfile.py
@@ -52,7 +52,8 @@ class UploadedFile(File):
     def _set_name(self, name):
         # Sanitize the file name so that it can't be dangerous.
         if name is not None:
-            # Just use the basename of the file -- anything else is dangerous.
+            # Normalize backslashes to forward slashes, then extract basename.
+            name = name.replace('\\', '/')
             name = os.path.basename(name)

             # File names longer than 255 characters can cause problems on older OSes.
```