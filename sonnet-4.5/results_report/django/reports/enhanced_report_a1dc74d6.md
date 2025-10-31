# Bug Report: Django Storage Backends Inconsistently Handle Null Bytes in Filenames

**Target**: `django.core.files.storage`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

InMemoryStorage accepts filenames containing null bytes (`\x00`) while FileSystemStorage rejects them with a ValueError, breaking the abstraction that storage backends should be interchangeable.

## Property-Based Test

```python
import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        USE_TZ=True,
        MEDIA_ROOT='/tmp/test_media',
        MEDIA_URL='/media/',
        FILE_UPLOAD_PERMISSIONS=None,
        FILE_UPLOAD_DIRECTORY_PERMISSIONS=None,
    )
    django.setup()

from hypothesis import given, strategies as st, settings as hypo_settings
from django.core.files.storage import InMemoryStorage, FileSystemStorage
from django.core.files.base import ContentFile
import tempfile
import shutil

@given(
    filename=st.text(min_size=1, max_size=50).filter(lambda x: '/' not in x and '\\' not in x and '..' not in x and x not in ['.', '..']),
    content_bytes=st.binary(min_size=0, max_size=1000)
)
@hypo_settings(max_examples=100)
def test_inmemory_vs_filesystem_equivalence(filename, content_bytes):
    """Test that InMemoryStorage and FileSystemStorage handle filenames consistently."""
    temp_dir = tempfile.mkdtemp()
    try:
        mem_storage = InMemoryStorage()
        fs_storage = FileSystemStorage(location=temp_dir)

        mem_content = ContentFile(content_bytes)
        fs_content = ContentFile(content_bytes)

        # Both should either accept or reject the filename
        mem_error = None
        fs_error = None

        try:
            mem_name = mem_storage.save(filename, mem_content)
            mem_exists = mem_storage.exists(mem_name)
        except Exception as e:
            mem_error = type(e).__name__
            mem_exists = False

        try:
            fs_name = fs_storage.save(filename, fs_content)
            fs_exists = fs_storage.exists(fs_name)
        except Exception as e:
            fs_error = type(e).__name__
            fs_exists = False

        # Check if behavior is consistent
        if mem_error != fs_error:
            print(f"\nInconsistent behavior detected!")
            print(f"Filename: {repr(filename)}")
            print(f"InMemoryStorage: {mem_error or 'Accepted'}")
            print(f"FileSystemStorage: {fs_error or 'Accepted'}")
            assert False, f"Storage backends handle filename {repr(filename)} differently"

        if not mem_error:
            assert mem_exists == fs_exists
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Run the test
if __name__ == "__main__":
    print("Running property-based test to find inconsistencies...")
    print("-" * 60)
    try:
        test_inmemory_vs_filesystem_equivalence()
        print("\nAll tests passed! No inconsistencies found.")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        print("\nThis demonstrates that storage backends behave inconsistently,")
```

<details>

<summary>
**Failing input**: `filename='\x00'`
</summary>
```
Running property-based test to find inconsistencies...
------------------------------------------------------------

Inconsistent behavior detected!
Filename: '6\x00Îﺚ\U000c92ad¬¾'
InMemoryStorage: Accepted
FileSystemStorage: ValueError

Inconsistent behavior detected!
Filename: '6\x00Îﺚ\U000c92ad¬¾'
InMemoryStorage: Accepted
FileSystemStorage: ValueError

[... truncated for brevity - pattern continues showing null byte filenames are accepted by InMemoryStorage but rejected by FileSystemStorage ...]

Inconsistent behavior detected!
Filename: '\x00'
InMemoryStorage: Accepted
FileSystemStorage: ValueError

Test failed: Storage backends handle filename '\x00' differently

This demonstrates that storage backends behave inconsistently,
```
</details>

## Reproducing the Bug

```python
import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        USE_TZ=True,
        MEDIA_ROOT='/tmp/test_media',
        MEDIA_URL='/media/',
        FILE_UPLOAD_PERMISSIONS=None,
        FILE_UPLOAD_DIRECTORY_PERMISSIONS=None,
    )
    django.setup()

from django.core.files.storage import InMemoryStorage, FileSystemStorage
from django.core.files.base import ContentFile
import tempfile
import shutil

# Create storage instances
mem_storage = InMemoryStorage()
temp_dir = tempfile.mkdtemp()
fs_storage = FileSystemStorage(location=temp_dir)

# Test filename with null byte
filename = 'file\x00name.txt'
content_bytes = b'test content'

print("Testing filename with null byte: repr(%r)" % filename)
print("-" * 50)

print("\n1. InMemoryStorage:")
try:
    saved_name = mem_storage.save(filename, ContentFile(content_bytes))
    print(f"   ✓ Accepted - saved as: {repr(saved_name)}")
    print(f"   ✓ File exists: {mem_storage.exists(saved_name)}")
except Exception as e:
    print(f"   ✗ Rejected with {type(e).__name__}: {e}")

print("\n2. FileSystemStorage:")
try:
    saved_name = fs_storage.save(filename, ContentFile(content_bytes))
    print(f"   ✓ Accepted - saved as: {repr(saved_name)}")
    print(f"   ✓ File exists: {fs_storage.exists(saved_name)}")
except Exception as e:
    print(f"   ✗ Rejected with {type(e).__name__}: {e}")

# Clean up
shutil.rmtree(temp_dir)

print("\n" + "-" * 50)
print("Result: Storage backends behave INCONSISTENTLY")
print("InMemoryStorage accepts null bytes while FileSystemStorage rejects them")
```

<details>

<summary>
ValueError: open: embedded null character in path
</summary>
```
Testing filename with null byte: repr('file\x00name.txt')
--------------------------------------------------

1. InMemoryStorage:
   ✓ Accepted - saved as: 'file\x00name.txt'
   ✓ File exists: True

2. FileSystemStorage:
   ✗ Rejected with ValueError: open: embedded null character in path

--------------------------------------------------
Result: Storage backends behave INCONSISTENTLY
InMemoryStorage accepts null bytes while FileSystemStorage rejects them
```
</details>

## Why This Is A Bug

This violates Django's storage abstraction principle in several critical ways:

1. **Breaking Test/Production Parity**: Django's documentation states that InMemoryStorage is "useful for speeding up tests by avoiding disk access." This implies tests using InMemoryStorage should behave identically to production using FileSystemStorage. Code that passes all tests with InMemoryStorage will crash in production when FileSystemStorage encounters a null byte.

2. **Inconsistent Validation**: The `validate_file_name()` function in `/django/core/files/utils.py:7-23` already validates for dangerous patterns like path traversal (`..`), empty names, and absolute paths. However, it misses null bytes, which are universally invalid in filesystem paths across all major operating systems (Unix/Linux, Windows, macOS).

3. **Silent Acceptance vs Hard Failure**: InMemoryStorage silently accepts invalid filenames that would cause immediate crashes with FileSystemStorage. This creates a false sense of security during development and testing.

4. **Security Implications**: Null bytes in filenames can lead to security vulnerabilities through null byte injection attacks, where the null byte causes string truncation in C-based system calls, potentially bypassing security checks.

## Relevant Context

The issue occurs because:

- **FileSystemStorage** (at `/django/core/files/storage/filesystem.py:81`) ultimately calls Python's `open()` function which delegates to the OS, where null bytes are invalid
- **InMemoryStorage** (at `/django/core/files/storage/memory.py:238-253`) stores files in memory using Python dictionaries and never validates against filesystem constraints
- **Base Storage.save()** (at `/django/core/files/storage/base.py:41,46,51`) calls `validate_file_name()` three times, but this validation function doesn't check for null bytes

The Django documentation for Storage backends states:
> "Django's default file storage is given by the DEFAULT_FILE_STORAGE setting; if you don't explicitly provide a storage system, this is the one that will be used."

This reinforces that storage backends should be interchangeable without changing application behavior.

Documentation links:
- Storage API: https://docs.djangoproject.com/en/stable/ref/files/storage/
- InMemoryStorage: https://docs.djangoproject.com/en/stable/ref/files/storage/#django.core.files.storage.InMemoryStorage
- validate_file_name source: https://github.com/django/django/blob/main/django/core/files/utils.py

## Proposed Fix

```diff
--- a/django/core/files/utils.py
+++ b/django/core/files/utils.py
@@ -6,6 +6,10 @@ from django.core.exceptions import SuspiciousFileOperation


 def validate_file_name(name, allow_relative_path=False):
+    # Reject null bytes which are invalid in filenames on all filesystems
+    if '\x00' in str(name):
+        raise SuspiciousFileOperation("File name '%s' contains null bytes." % name)
+
     # Remove potentially dangerous names
     if os.path.basename(name) in {"", ".", ".."}:
         raise SuspiciousFileOperation("Could not derive file name from '%s'" % name)
```