# Bug Report: django.core.files.utils.validate_file_name Backslash Path Separator Bypass

**Target**: `django.core.files.utils.validate_file_name`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `validate_file_name` function fails to properly validate file names containing backslash path separators when `allow_relative_path=False`. On Linux/Unix systems, backslashes are not treated as path separators by `os.path.basename()`, allowing names like `"uploads\\..\\passwords.txt"` to pass validation. This creates a cross-platform security vulnerability where files uploaded on one OS could enable directory traversal attacks when accessed on another OS (particularly Windows).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.core.files.utils import validate_file_name
from django.core.exceptions import SuspiciousFileOperation
import pytest


@given(st.text(min_size=1), st.sampled_from(['/', '\\']))
@settings(max_examples=500)
def test_validate_file_name_rejects_path_separators(base_name, separator):
    name = f"{base_name}{separator}file"

    with pytest.raises(SuspiciousFileOperation):
        validate_file_name(name, allow_relative_path=False)
```

**Failing input**: `base_name='0'`, `separator='\\'`

## Reproducing the Bug

```python
from django.core.files.utils import validate_file_name
from django.core.exceptions import SuspiciousFileOperation


dangerous_name = "uploads\\..\\passwords.txt"

try:
    result = validate_file_name(dangerous_name, allow_relative_path=False)
    print(f"BUG: Allowed dangerous name: {result!r}")
except SuspiciousFileOperation as e:
    print(f"Correctly rejected: {e}")


safe_check = "uploads/../passwords.txt"

try:
    result = validate_file_name(safe_check, allow_relative_path=False)
    print(f"Allowed: {result!r}")
except SuspiciousFileOperation as e:
    print(f"Correctly rejected: {e}")
```

**Output on Linux:**
```
BUG: Allowed dangerous name: 'uploads\\..\\passwords.txt'
Correctly rejected: File name 'uploads/../passwords.txt' includes path elements
```

## Why This Is A Bug

The function has inconsistent behavior across the two code paths:

1. When `allow_relative_path=True`, it correctly normalizes backslashes to forward slashes before validation:
   ```python
   path = pathlib.PurePosixPath(str(name).replace("\\", "/"))
   ```

2. When `allow_relative_path=False`, it relies on `os.path.basename(name)`, which is platform-dependent:
   - On Linux/Unix: `os.path.basename("a\\b")` returns `"a\\b"` (backslash is not a separator)
   - On Windows: `os.path.basename("a\\b")` returns `"b"` (backslash IS a separator)

This creates a security vulnerability because:
- A file uploaded on Linux with name `"..\\sensitive.txt"` passes validation
- On Windows, this same name would traverse directories
- Django is meant to be cross-platform, so validation should be consistent
- Even if the application only runs on Linux, allowing backslashes in filenames can cause issues when files are transferred to Windows systems

## Fix

The function should normalize backslashes to forward slashes before the `os.path.basename()` check when `allow_relative_path=False`. This ensures consistent, platform-independent validation:

```diff
def validate_file_name(name, allow_relative_path=False):
+   # Normalize path separators for cross-platform consistency
+   normalized_name = str(name).replace("\\", "/")
+
    # Remove potentially dangerous names
-   if os.path.basename(name) in {"", ".", ".."}:
+   if os.path.basename(normalized_name) in {"", ".", ".."}:
        raise SuspiciousFileOperation("Could not derive file name from '%s'" % name)

    if allow_relative_path:
        # Ensure that name can be treated as a pure posix path, i.e. Unix
        # style (with forward slashes).
-       path = pathlib.PurePosixPath(str(name).replace("\\", "/"))
+       path = pathlib.PurePosixPath(normalized_name)
        if path.is_absolute() or ".." in path.parts:
            raise SuspiciousFileOperation(
                "Detected path traversal attempt in '%s'" % name
            )
-   elif name != os.path.basename(name):
+   elif normalized_name != os.path.basename(normalized_name):
        raise SuspiciousFileOperation("File name '%s' includes path elements" % name)

    return name
```