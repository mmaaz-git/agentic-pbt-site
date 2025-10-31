# Bug Report: django.core.files.utils.validate_file_name Backslash Handling Inconsistency

**Target**: `django.core.files.utils.validate_file_name`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `validate_file_name` function has inconsistent backslash handling between `allow_relative_path=True` and `allow_relative_path=False` modes on Unix systems, allowing backslashes in filenames when they should be rejected as path separators.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from django.core.files.utils import validate_file_name
from django.core.exceptions import SuspiciousFileOperation

@given(st.text(alphabet=st.characters(blacklist_categories=['Cc', 'Cs']), min_size=1, max_size=100))
@settings(max_examples=1000)
def test_validate_rejects_backslash_as_separator(name):
    if '\\' in name and os.path.basename(name) not in {"", ".", ".."}:
        try:
            validate_file_name(name, allow_relative_path=False)
            assert False, f"Should reject backslash in filename: {name!r}"
        except SuspiciousFileOperation:
            pass
```

**Failing input**: `'\\'` and `'file\\name'`

## Reproducing the Bug

```python
from django.core.files.utils import validate_file_name

filename_with_backslash = 'file\\name'
result = validate_file_name(filename_with_backslash, allow_relative_path=False)

print(f"Result: {result!r}")
```

On Unix systems, this prints `'file\\name'` instead of raising `SuspiciousFileOperation`.

## Why This Is A Bug

1. **Cross-platform inconsistency**: When `allow_relative_path=True`, Django explicitly converts backslashes to forward slashes (`str(name).replace("\\", "/")`), showing the intent to treat backslashes as path separators on all platforms.

2. **Unix-specific vulnerability**: When `allow_relative_path=False`, the function checks `name != os.path.basename(name)`. On Unix, `os.path.basename('file\\name')` returns `'file\\name'` (backslash is not a separator), so the validation passes.

3. **Security context**: This function is used in `UploadedFile` to validate uploaded filenames. Accepting backslashes on Unix but not Windows creates platform-dependent behavior in a security-critical function.

4. **Inconsistent semantics**: The two branches handle the same character (backslash) differently, which is confusing and error-prone.

## Fix

```diff
--- a/django/core/files/utils.py
+++ b/django/core/files/utils.py
@@ -17,7 +17,10 @@ def validate_file_name(name, allow_relative_path=False):
             raise SuspiciousFileOperation(
                 "Detected path traversal attempt in '%s'" % name
             )
-    elif name != os.path.basename(name):
+    elif name != os.path.basename(name) or "\\" in name:
         raise SuspiciousFileOperation("File name '%s' includes path elements" % name)

     return name
```

This ensures that backslashes are consistently rejected as path separators in both modes, maintaining cross-platform consistency.