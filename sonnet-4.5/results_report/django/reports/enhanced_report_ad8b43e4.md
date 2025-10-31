# Bug Report: django.core.files.utils.validate_file_name Cross-Platform Path Separator Vulnerability

**Target**: `django.core.files.utils.validate_file_name`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `validate_file_name` function incorrectly allows backslash characters in file names when `allow_relative_path=False` on Unix/Linux systems, creating a cross-platform security vulnerability that could enable directory traversal attacks when files are transferred between operating systems.

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

# Run the test
if __name__ == "__main__":
    test_validate_file_name_rejects_path_separators()
```

<details>

<summary>
**Failing input**: `base_name='0'`, `separator='\\'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 17, in <module>
    test_validate_file_name_rejects_path_separators()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 8, in test_validate_file_name_rejects_path_separators
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 12, in test_validate_file_name_rejects_path_separators
    with pytest.raises(SuspiciousFileOperation):
         ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: DID NOT RAISE <class 'django.core.exceptions.SuspiciousFileOperation'>
Falsifying example: test_validate_file_name_rejects_path_separators(
    base_name='0',
    separator='\\',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/django/core/files/utils.py:23
```
</details>

## Reproducing the Bug

```python
from django.core.files.utils import validate_file_name
from django.core.exceptions import SuspiciousFileOperation

# Test case 1: Backslash path separator bypass
dangerous_name = "uploads\\..\\passwords.txt"
print(f"Testing dangerous name: {dangerous_name!r}")
try:
    result = validate_file_name(dangerous_name, allow_relative_path=False)
    print(f"BUG: Allowed dangerous name: {result!r}")
except SuspiciousFileOperation as e:
    print(f"Correctly rejected: {e}")

print()

# Test case 2: Forward slash is correctly blocked
safe_check = "uploads/../passwords.txt"
print(f"Testing with forward slash: {safe_check!r}")
try:
    result = validate_file_name(safe_check, allow_relative_path=False)
    print(f"BUG: Allowed dangerous name: {result!r}")
except SuspiciousFileOperation as e:
    print(f"Correctly rejected: {e}")

print()

# Test case 3: Simple backslash case
simple_backslash = "0\\file"
print(f"Testing simple backslash: {simple_backslash!r}")
try:
    result = validate_file_name(simple_backslash, allow_relative_path=False)
    print(f"BUG: Allowed name with backslash: {result!r}")
except SuspiciousFileOperation as e:
    print(f"Correctly rejected: {e}")

print()

# Test case 4: Demonstrate platform-dependent behavior of os.path.basename
import os
print("Platform-dependent behavior of os.path.basename:")
print(f"os.path.basename('a\\\\b') = {os.path.basename('a\\b')!r}")
print(f"os.path.basename('a/b') = {os.path.basename('a/b')!r}")
```

<details>

<summary>
Output demonstrating the backslash bypass vulnerability on Linux
</summary>
```
Testing dangerous name: 'uploads\\..\\passwords.txt'
BUG: Allowed dangerous name: 'uploads\\..\\passwords.txt'

Testing with forward slash: 'uploads/../passwords.txt'
Correctly rejected: File name 'uploads/../passwords.txt' includes path elements

Testing simple backslash: '0\\file'
BUG: Allowed name with backslash: '0\\file'

Platform-dependent behavior of os.path.basename:
os.path.basename('a\\b') = 'a\\b'
os.path.basename('a/b') = 'b'
```
</details>

## Why This Is A Bug

This violates the expected security behavior of `validate_file_name` for several critical reasons:

1. **Inconsistent Path Separator Handling**: The function has two distinct code paths that handle backslashes differently:
   - When `allow_relative_path=True` (line 15): The function explicitly normalizes backslashes to forward slashes using `.replace("\\", "/")`
   - When `allow_relative_path=False` (line 20): The function relies on `os.path.basename()` without normalization

2. **Platform-Dependent Security Vulnerability**: The `os.path.basename()` function behaves differently on different operating systems:
   - On Linux/Unix: Treats backslash as a regular character, not a path separator (`os.path.basename("a\\b")` returns `"a\\b"`)
   - On Windows: Treats backslash as a path separator (`os.path.basename("a\\b")` returns `"b"`)

3. **Cross-Platform Attack Vector**: This creates a serious security vulnerability where:
   - A malicious file name like `"..\\..\\etc\\passwd"` uploaded on a Linux server passes validation
   - The same file, when accessed on a Windows system or transferred to Windows storage, could traverse directories
   - Even file names like `"uploads\\..\\sensitive.txt"` that appear to stay within a directory actually contain path traversal sequences

4. **Violation of Django's Cross-Platform Promise**: Django is designed to be platform-agnostic, but this bug makes file validation behavior platform-dependent, contradicting the framework's design principles.

## Relevant Context

The vulnerability exists in Django's core file utilities module at `/django/core/files/utils.py`. The function is used throughout Django's file handling system, particularly in file upload processing and storage backends.

Key observations:
- The function already knows how to handle backslashes correctly (it does so in the `allow_relative_path=True` branch)
- The inconsistency appears to be an oversight rather than intentional design
- The security check at line 20 (`name != os.path.basename(name)`) fails to detect backslash-based path traversal on Unix systems

Django documentation: https://docs.djangoproject.com/en/stable/ref/files/uploads/
Source code: https://github.com/django/django/blob/main/django/core/files/utils.py

## Proposed Fix

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