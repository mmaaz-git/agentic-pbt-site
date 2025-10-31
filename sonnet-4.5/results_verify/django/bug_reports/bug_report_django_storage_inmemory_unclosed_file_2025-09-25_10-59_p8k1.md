# Bug Report: InMemoryStorage._save() Resource Leak - Unclosed File

**Target**: `django.core.files.storage.memory.InMemoryStorage._save()`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`InMemoryStorage._save()` opens a file descriptor in a loop but never closes it, creating a resource leak. This is inconsistent with `FileSystemStorage._save()`, which properly closes file descriptors in a `try/finally` block.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.core.files.storage import InMemoryStorage
from django.core.files.base import ContentFile

@given(
    content=st.binary(min_size=1, max_size=1000),
    filename=st.text(
        alphabet=st.characters(min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=20
    ).filter(lambda x: '/' not in x and x not in {'.', '..'})
)
@settings(max_examples=100)
def test_file_closed_after_save(content, filename):
    storage = InMemoryStorage()

    file_obj = ContentFile(content, name=filename)
    saved_name = storage.save(filename, file_obj)

    file_node = storage._resolve(saved_name)

    assert hasattr(file_node, 'file'), "File node should have a file attribute"

    # The file should ideally be closed after save completes
    # Currently it remains open, which is a resource leak
    is_closed = file_node.file.closed

    # This assertion would fail with current implementation
    # assert is_closed, f"File should be closed after save, but is open"
```

**Failing behavior**: Files remain open after `save()` completes

## Reproducing the Bug

```python
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
from django.conf import settings
import tempfile

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        USE_TZ=True,
        MEDIA_ROOT=tempfile.mkdtemp(),
        MEDIA_URL='/media/',
    )
    django.setup()

from django.core.files.storage import InMemoryStorage
from django.core.files.base import ContentFile

storage = InMemoryStorage()

content = ContentFile(b"Test content", name="test.txt")
saved_name = storage.save("test.txt", content)

file_node = storage._resolve(saved_name)
print(f"File closed: {file_node.file.closed}")
```

Output:
```
File closed: False
```

The file remains open even though the save operation has completed.

## Why This Is A Bug

In `memory.py` lines 238-253, `InMemoryStorage._save()` opens a file descriptor but never closes it:

```python
def _save(self, name, content):
    file_node = self._resolve(
        name, create_if_missing=True, leaf_cls=InMemoryFileNode
    )
    fd = None
    for chunk in content.chunks():
        if fd is None:
            mode = "wb" if isinstance(chunk, bytes) else "wt"
            fd = file_node.open(mode)
        fd.write(chunk)

    if hasattr(content, "temporary_file_path"):
        os.remove(content.temporary_file_path())

    file_node.modified_time = now()
    return self._relative_path(name).replace("\\", "/")
```

The `fd` is opened on line 246 but never closed. This is a resource leak.

In contrast, `FileSystemStorage._save()` in `filesystem.py` lines 138-151 properly handles file closure in a `try/finally` block:

```python
fd = os.open(full_path, open_flags, 0o666)
_file = None
try:
    locks.lock(fd, locks.LOCK_EX)
    for chunk in content.chunks():
        if _file is None:
            mode = "wb" if isinstance(chunk, bytes) else "wt"
            _file = os.fdopen(fd, mode)
        _file.write(chunk)
finally:
    locks.unlock(fd)
    if _file is not None:
        _file.close()
    else:
        os.close(fd)
```

While in-memory files (BytesIO/StringIO) don't consume OS file descriptors, leaving them open is still incorrect and inconsistent with the filesystem implementation. It also prevents proper resource cleanup and could cause issues in long-running processes.

## Fix

Add proper file closure in a try/finally block:

```diff
--- a/django/core/files/storage/memory.py
+++ b/django/core/files/storage/memory.py
@@ -240,14 +240,18 @@ class InMemoryStorage(Storage, StorageSettingsMixin):
             name, create_if_missing=True, leaf_cls=InMemoryFileNode
         )
         fd = None
-        for chunk in content.chunks():
-            if fd is None:
-                mode = "wb" if isinstance(chunk, bytes) else "wt"
-                fd = file_node.open(mode)
-            fd.write(chunk)
+        try:
+            for chunk in content.chunks():
+                if fd is None:
+                    mode = "wb" if isinstance(chunk, bytes) else "wt"
+                    fd = file_node.open(mode)
+                fd.write(chunk)
+        finally:
+            if fd is not None:
+                fd.close()

         if hasattr(content, "temporary_file_path"):
             os.remove(content.temporary_file_path())

         file_node.modified_time = now()
```