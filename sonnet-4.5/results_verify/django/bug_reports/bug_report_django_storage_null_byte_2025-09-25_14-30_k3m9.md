# Bug Report: InMemoryStorage Accepts Null Bytes in Filenames

**Target**: `django.core.files.storage`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

InMemoryStorage and FileSystemStorage handle null bytes in filenames inconsistently. FileSystemStorage raises ValueError when attempting to save a file with a null byte in its name, while InMemoryStorage accepts it. This breaks the abstraction and can cause code that works in tests (using InMemoryStorage) to fail in production (using FileSystemStorage).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.files.storage import InMemoryStorage, FileSystemStorage
from django.core.files.base import ContentFile
import tempfile
import shutil

@given(
    st.text(min_size=1, max_size=50).filter(lambda x: '/' not in x and '\\' not in x and '..' not in x and x not in ['.', '..']),
    st.binary(min_size=0, max_size=1000)
)
def test_inmemory_vs_filesystem_equivalence(filename, content_bytes):
    temp_dir = tempfile.mkdtemp()
    try:
        mem_storage = InMemoryStorage()
        fs_storage = FileSystemStorage(location=temp_dir)

        mem_content = ContentFile(content_bytes)
        fs_content = ContentFile(content_bytes)

        mem_name = mem_storage.save(filename, mem_content)
        fs_name = fs_storage.save(filename, fs_content)

        assert mem_storage.exists(mem_name) == fs_storage.exists(fs_name)
    finally:
        shutil.rmtree(temp_dir)
```

**Failing input**: `filename='\x00'`

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

mem_storage = InMemoryStorage()
temp_dir = tempfile.mkdtemp()
fs_storage = FileSystemStorage(location=temp_dir)

filename = 'file\x00name.txt'

print("InMemoryStorage:")
mem_storage.save(filename, ContentFile(b'test'))
print("  ✓ Accepted")

print("FileSystemStorage:")
try:
    fs_storage.save(filename, ContentFile(b'test'))
    print("  ✓ Accepted")
except ValueError as e:
    print(f"  ✗ Rejected: {e}")

shutil.rmtree(temp_dir)
```

Output:
```
InMemoryStorage:
  ✓ Accepted
FileSystemStorage:
  ✗ Rejected: open: embedded null character in path
```

## Why This Is A Bug

1. **Breaks abstraction**: Storage backends should be interchangeable. Code using InMemoryStorage for testing may pass tests but fail with FileSystemStorage in production.

2. **Invalid filenames**: Null bytes are not valid in filenames on any major filesystem (Unix, Windows, etc.). InMemoryStorage should reject them for consistency.

3. **Security concern**: Accepting null bytes could lead to unexpected behavior if filenames are used in system calls or path operations.

## Fix

Add null byte validation to `validate_file_name()` in `django/core/files/utils.py`:

```diff
def validate_file_name(name, allow_relative_path=False):
+    # Reject null bytes which are invalid in filenames
+    if '\x00' in name:
+        raise SuspiciousFileOperation("File name '%s' contains null bytes" % name)
+
    # Remove potentially dangerous names
    if os.path.basename(name) in {"", ".", ".."}:
        raise SuspiciousFileOperation("Could not derive file name from '%s'" % name)

    if allow_relative_path:
        # Ensure that name can be treated as a pure posix path, i.e. Unix
        # style (with forward slashes).
        path = pathlib.PurePosixPath(str(name).replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise SuspiciousFileOperation(
                "Detected path traversal attempt in '%s'" % name
            )
    elif name != os.path.basename(name):
        raise SuspiciousFileOperation("File name '%s' includes path elements" % name)

    return name
```

This fix ensures all storage backends reject null bytes consistently, since `validate_file_name()` is called by `Storage.save()` which is used by all storage implementations.