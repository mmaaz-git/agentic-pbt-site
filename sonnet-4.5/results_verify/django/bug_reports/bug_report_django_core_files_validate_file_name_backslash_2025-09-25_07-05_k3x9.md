# Bug Report: django.core.files.utils.validate_file_name Cross-Platform Path Traversal

**Target**: `django.core.files.utils.validate_file_name`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`validate_file_name()` fails to sanitize backslash characters on non-Windows platforms, allowing potential path traversal attacks when uploaded files are transferred to or accessed on Windows systems.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from django.core.files.utils import validate_file_name
from django.core.exceptions import SuspiciousFileOperation
import pytest

@given(st.text(min_size=1))
def test_validate_file_name_rejects_backslash_path_elements(name):
    assume("\\" in name)

    with pytest.raises(SuspiciousFileOperation):
        validate_file_name(name, allow_relative_path=False)
```

**Failing input**: `'\\'` (and any string containing backslash)

## Reproducing the Bug

```python
import os
from django.core.files.utils import validate_file_name
from django.core.files.uploadedfile import UploadedFile
from io import BytesIO

result = validate_file_name('uploads\\..\\..\\etc\\passwd', allow_relative_path=False)
print(f"Sanitized name: {repr(result)}")

uf = UploadedFile(file=BytesIO(b"malicious"), name='uploads\\..\\..\\etc\\passwd')
print(f"UploadedFile.name: {repr(uf.name)}")
print(f"Contains path traversal: {'..' in uf.name}")
```

Output on Linux:
```
Sanitized name: 'uploads\\..\\..\\etc\\passwd'
UploadedFile.name: 'uploads\\..\\..\\etc\\passwd'
Contains path traversal: True
```

## Why This Is A Bug

Django's file upload validation is designed to prevent path traversal attacks by rejecting file names containing path separators. However, the current implementation only checks for the platform-specific path separator (e.g., `/` on Linux) and relies on `os.path.basename()` for sanitization.

On Linux/Unix, `os.path.basename('a\\b')` returns `'a\\b'` unchanged because `\` is not a path separator on those platforms. However, if such a file name is later accessed on Windows (e.g., in a cross-platform deployment or when files are synced), the backslash becomes a valid path separator, enabling path traversal.

Lines 20-21 in `/django/core/files/utils.py`:
```python
elif name != os.path.basename(name):
    raise SuspiciousFileOperation("File name '%s' includes path elements" % name)
```

This check is platform-dependent and fails to account for Windows path separators on non-Windows systems.

## Fix

```diff
--- a/django/core/files/utils.py
+++ b/django/core/files/utils.py
@@ -7,7 +7,10 @@ from django.core.exceptions import SuspiciousFileOperation
 def validate_file_name(name, allow_relative_path=False):
     # Remove potentially dangerous names
     if os.path.basename(name) in {"", ".", ".."}:
         raise SuspiciousFileOperation("Could not derive file name from '%s'" % name)
+
+    # Check for backslash path separators (Windows) on all platforms
+    if not allow_relative_path and "\\" in name:
+        raise SuspiciousFileOperation("File name '%s' includes path elements" % name)

     if allow_relative_path:
         # Ensure that name can be treated as a pure posix path, i.e. Unix
```