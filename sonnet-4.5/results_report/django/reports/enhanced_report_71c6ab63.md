# Bug Report: Django UploadedFile Backslash Path Separator Security Vulnerability

**Target**: `django.core.files.uploadedfile.UploadedFile._set_name`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `UploadedFile._set_name` method fails to sanitize backslash characters when processing uploaded filenames on Unix systems, creating a potential path traversal vulnerability when files are later processed on Windows systems or in cross-platform environments.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for Django UploadedFile backslash vulnerability"""

import os
import sys

# Add Django to path if needed
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Setup minimal Django settings
import django
from django.conf import settings
settings.configure(DEBUG=True, SECRET_KEY='test')
django.setup()

from hypothesis import given, strategies as st, assume
from django.core.files.uploadedfile import SimpleUploadedFile

@given(st.text(min_size=1), st.binary())
def test_uploaded_file_name_sanitization(name, content):
    assume(name not in {'', '.', '..'})
    f = SimpleUploadedFile(name, content)
    assert '/' not in f.name, f"Forward slash found in sanitized name: {f.name!r}"
    assert '\\' not in f.name, f"Backslash found in sanitized name: {f.name!r}"

if __name__ == "__main__":
    # Run the test
    test_uploaded_file_name_sanitization()
```

<details>

<summary>
**Failing input**: `name='\\', content=b''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 28, in <module>
    test_uploaded_file_name_sanitization()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 20, in test_uploaded_file_name_sanitization
    def test_uploaded_file_name_sanitization(name, content):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 24, in test_uploaded_file_name_sanitization
    assert '\\' not in f.name, f"Backslash found in sanitized name: {f.name!r}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Backslash found in sanitized name: '\\'
Falsifying example: test_uploaded_file_name_sanitization(
    name='\\',
    content=b'',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of Django UploadedFile backslash vulnerability"""

import os
import sys

# Add Django to path if needed
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Setup minimal Django settings
import django
from django.conf import settings
settings.configure(DEBUG=True, SECRET_KEY='test')
django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile

# Test case 1: Basic backslash in filename
print("Test 1: Basic backslash in filename")
f = SimpleUploadedFile('test\\file.txt', b'content')
print(f"  Input: 'test\\\\file.txt'")
print(f"  Output: '{f.name}'")
print(f"  Contains backslash: {'\\' in f.name}")
print()

# Test case 2: Multiple backslashes (potential path traversal)
print("Test 2: Multiple backslashes (path traversal attempt)")
f2 = SimpleUploadedFile('..\\..\\etc\\passwd', b'malicious')
print(f"  Input: '..\\\\..\\\\etc\\\\passwd'")
print(f"  Output: '{f2.name}'")
print(f"  Contains backslash: {'\\' in f2.name}")
print()

# Test case 3: Mixed forward and backslashes
print("Test 3: Mixed forward and backslashes")
f3 = SimpleUploadedFile('../test\\..\\file.txt', b'mixed')
print(f"  Input: '../test\\\\..\\\\file.txt'")
print(f"  Output: '{f3.name}'")
print(f"  Contains backslash: {'\\' in f3.name}")
print(f"  Contains forward slash: {'/' in f3.name}")
print()

# Test case 4: Just a backslash
print("Test 4: Single backslash character")
f4 = SimpleUploadedFile('\\', b'')
print(f"  Input: '\\\\'")
print(f"  Output: '{f4.name}'")
print(f"  Name equals backslash: {f4.name == '\\\\'}")
print()

# Show os.path.basename behavior on current system
print("System information:")
print(f"  Operating System: {os.name}")
print(f"  os.path.basename('test\\\\file.txt') = '{os.path.basename('test\\\\file.txt')}'")
print(f"  os.path.basename('test/file.txt') = '{os.path.basename('test/file.txt')}'")
```

<details>

<summary>
Demonstrates backslash characters persisting after sanitization
</summary>
```
Test 1: Basic backslash in filename
  Input: 'test\\file.txt'
  Output: 'test\file.txt'
  Contains backslash: True

Test 2: Multiple backslashes (path traversal attempt)
  Input: '..\\..\\etc\\passwd'
  Output: '..\..\etc\passwd'
  Contains backslash: True

Test 3: Mixed forward and backslashes
  Input: '../test\\..\\file.txt'
  Output: 'test\..\file.txt'
  Contains backslash: True
  Contains forward slash: False

Test 4: Single backslash character
  Input: '\\'
  Output: '\'
  Name equals backslash: False

System information:
  Operating System: posix
  os.path.basename('test\\file.txt') = 'test\\file.txt'
  os.path.basename('test/file.txt') = 'file.txt'
```
</details>

## Why This Is A Bug

This violates the security principle explicitly stated in the Django source code comment at `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/files/uploadedfile.py:54`: "Just use the basename of the file -- anything else is dangerous."

The critical issue is that `os.path.basename()` behavior is platform-dependent:
- On Unix/Linux systems: Only forward slashes (/) are treated as path separators
- On Windows systems: Both forward slashes (/) and backslashes (\) are path separators

This creates an asymmetric vulnerability where:
1. A malicious filename like `..\\..\\etc\\passwd` uploaded to a Django application on Linux retains its backslashes
2. The file passes through Django's sanitization with backslashes intact
3. If this file is later accessed on Windows (e.g., shared storage, cross-platform deployment, file syncing), the backslashes become active path separators
4. This could enable directory traversal attacks in cross-platform scenarios

The inconsistency is evident when comparing with Django's own `validate_file_name()` function in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/files/utils.py:15`, which explicitly handles backslashes when `allow_relative_path=True` by converting them to forward slashes before validation.

## Relevant Context

The vulnerability affects all Django applications that:
- Accept file uploads from untrusted sources
- Store or process uploaded files in cross-platform environments
- Share file storage between Unix and Windows systems
- Use cloud storage or network file systems accessed by multiple platforms
- Deploy to both Unix and Windows environments

Django's documentation on [File Uploads](https://docs.djangoproject.com/en/stable/topics/http/file-uploads/) emphasizes security but doesn't explicitly warn about this cross-platform path traversal risk.

The issue has been present since the introduction of the `UploadedFile` class and affects all Django versions that use `os.path.basename()` for sanitization without normalizing backslashes first.

## Proposed Fix

```diff
--- a/django/core/files/uploadedfile.py
+++ b/django/core/files/uploadedfile.py
@@ -51,8 +51,10 @@ class UploadedFile(File):
     def _set_name(self, name):
         # Sanitize the file name so that it can't be dangerous.
         if name is not None:
-            # Just use the basename of the file -- anything else is dangerous.
+            # Normalize backslashes to forward slashes for cross-platform safety,
+            # then use the basename of the file -- anything else is dangerous.
+            name = name.replace('\\', '/')
             name = os.path.basename(name)

             # File names longer than 255 characters can cause problems on older OSes.
```