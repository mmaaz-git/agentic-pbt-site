# Bug Report: django.core.files.storage FileSystemStorage Line Ending Corruption

**Target**: `django.core.files.storage.FileSystemStorage`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

FileSystemStorage and InMemoryStorage behave inconsistently when handling text files with non-LF line endings. FileSystemStorage silently converts CR (`\r`) and CRLF (`\r\n`) to LF (`\n`) when opening files in text mode, while InMemoryStorage preserves the original line endings. This violates the storage backend equivalence property and can cause silent data corruption.

## Property-Based Test

```python
from hypothesis import given, settings as hypothesis_settings, strategies as st
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage, InMemoryStorage
import tempfile


@st.composite
def file_contents(draw):
    content_type = draw(st.sampled_from(['bytes', 'text']))
    if content_type == 'bytes':
        return draw(st.binary(min_size=0, max_size=10000))
    else:
        return draw(st.text(min_size=0, max_size=10000))


@given(file_contents())
@hypothesis_settings(max_examples=200)
def test_save_open_roundtrip_filesystem(content):
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileSystemStorage(location=tmpdir)
        file_obj = ContentFile(content, name='test.txt')
        saved_name = storage.save('test.txt', file_obj)

        with storage.open(saved_name, 'rb' if isinstance(content, bytes) else 'r') as f:
            retrieved_content = f.read()

        assert retrieved_content == content
```

**Failing input**: `content='\r'` (or any text containing `\r` or `\r\n`)

## Reproducing the Bug

```python
import tempfile
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage, InMemoryStorage

content = '\r'

with tempfile.TemporaryDirectory() as tmpdir:
    fs_storage = FileSystemStorage(location=tmpdir)
    mem_storage = InMemoryStorage()

    fs_file = ContentFile(content, name='test.txt')
    mem_file = ContentFile(content, name='test.txt')

    fs_saved = fs_storage.save('test.txt', fs_file)
    mem_saved = mem_storage.save('test.txt', mem_file)

    with fs_storage.open(fs_saved, 'r') as f:
        fs_result = f.read()

    with mem_storage.open(mem_saved, 'r') as f:
        mem_result = f.read()

    print(f"Original:  {repr(content)}")
    print(f"FS result: {repr(fs_result)}")
    print(f"Mem result: {repr(mem_result)}")
```

Output:
```
Original:  '\r'
FS result: '\n'
Mem result: '\r'
```

The bug affects all line ending variations:
- `'\r'` → converted to `'\n'`
- `'\r\n'` → converted to `'\n'`
- `'hello\rworld'` → converted to `'hello\nworld'`
- `'hello\r\nworld'` → converted to `'hello\nworld'`

## Why This Is A Bug

1. **Backend Inconsistency**: Two storage implementations of the same interface should behave identically. InMemoryStorage preserves line endings, but FileSystemStorage does not.

2. **Silent Data Corruption**: Users storing text files with specific line endings (e.g., Windows CRLF) would expect them to be preserved. Silent modification breaks the fundamental contract of a storage system.

3. **Round-trip Property Violation**: The content retrieved from storage should match what was saved, which is a fundamental property of any storage system.

4. **Real-world Impact**: Applications that:
   - Store configuration files with specific line endings
   - Handle Windows text files (CRLF)
   - Parse files where CR has semantic meaning
   - Need to preserve exact file content for checksums/signatures

## Fix

The root cause is in `FileSystemStorage._open()` at line 81:

```python
def _open(self, name, mode="rb"):
    return File(open(self.path(name), mode))
```

Python's `open()` in text mode applies universal newline translation by default, converting all `\r` and `\r\n` to `\n`. This can be disabled by passing `newline=''`.

```diff
--- a/django/core/files/storage/filesystem.py
+++ b/django/core/files/storage/filesystem.py
@@ -78,7 +78,11 @@ class FileSystemStorage(Storage, StorageSettingsMixin):
         )

     def _open(self, name, mode="rb"):
-        return File(open(self.path(name), mode))
+        # Disable newline translation in text mode to preserve line endings
+        # and maintain consistency with InMemoryStorage
+        if 'b' not in mode:
+            return File(open(self.path(name), mode, newline=''))
+        return File(open(self.path(name), mode))

     def _save(self, name, content):
         full_path = self.path(name)
```

This fix ensures FileSystemStorage preserves line endings like InMemoryStorage, maintaining backend equivalence and preventing silent data corruption.