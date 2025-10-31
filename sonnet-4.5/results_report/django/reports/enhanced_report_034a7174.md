# Bug Report: django.core.files.storage FileSystemStorage Line Ending Corruption

**Target**: `django.core.files.storage.filesystem.FileSystemStorage`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

FileSystemStorage silently converts carriage return characters (`\r` and `\r\n`) to line feeds (`\n`) when opening files in text mode, while InMemoryStorage preserves the original line endings, causing backend inconsistency and data corruption.

## Property-Based Test

```python
import tempfile
import django
from django.conf import settings

# Configure Django settings
settings.configure(
    USE_TZ=False,
    MEDIA_ROOT='/tmp',
    MEDIA_URL='/media/',
)
django.setup()

from hypothesis import given, settings as hypothesis_settings, strategies as st
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage, InMemoryStorage


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

        assert retrieved_content == content, f"Content mismatch: {repr(content)} != {repr(retrieved_content)}"


if __name__ == "__main__":
    # Run the test
    import traceback
    try:
        test_save_open_roundtrip_filesystem()
        print("All tests passed!")
    except Exception as e:
        print("Falsifying example: test_save_open_roundtrip_filesystem(")
        print("    content='\\r',")
        print(")")
        print()
        traceback.print_exc()
        print()
        print("This demonstrates that FileSystemStorage does not preserve line endings in text mode.")
```

<details>

<summary>
**Failing input**: `content='\r'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 45, in <module>
    test_save_open_roundtrip_filesystem()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 28, in test_save_open_roundtrip_filesystem
    @hypothesis_settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 38, in test_save_open_roundtrip_filesystem
    assert retrieved_content == content, f"Content mismatch: {repr(content)} != {repr(retrieved_content)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Content mismatch: '\r' != '\n'
Falsifying example: test_save_open_roundtrip_filesystem(
    content='\r',
)
Falsifying example: test_save_open_roundtrip_filesystem(
    content='\r',
)


This demonstrates that FileSystemStorage does not preserve line endings in text mode.
```
</details>

## Reproducing the Bug

```python
import tempfile
import django
from django.conf import settings

# Configure Django settings
settings.configure(
    USE_TZ=False,
    MEDIA_ROOT='/tmp',
    MEDIA_URL='/media/',
)
django.setup()

from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage, InMemoryStorage

# Test case 1: Single CR character
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

    print(f"Test 1: Single CR character")
    print(f"Original:          {repr(content)}")
    print(f"FileSystemStorage: {repr(fs_result)}")
    print(f"InMemoryStorage:   {repr(mem_result)}")
    print(f"Match: {fs_result == mem_result}")
    print()

# Test case 2: CRLF sequence
content = '\r\n'

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

    print(f"Test 2: CRLF sequence")
    print(f"Original:          {repr(content)}")
    print(f"FileSystemStorage: {repr(fs_result)}")
    print(f"InMemoryStorage:   {repr(mem_result)}")
    print(f"Match: {fs_result == mem_result}")
    print()

# Test case 3: Text with CR in the middle
content = 'hello\rworld'

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

    print(f"Test 3: Text with CR in the middle")
    print(f"Original:          {repr(content)}")
    print(f"FileSystemStorage: {repr(fs_result)}")
    print(f"InMemoryStorage:   {repr(mem_result)}")
    print(f"Match: {fs_result == mem_result}")
    print()

# Test case 4: Windows-style text file
content = 'line1\r\nline2\r\nline3'

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

    print(f"Test 4: Windows-style text file")
    print(f"Original:          {repr(content)}")
    print(f"FileSystemStorage: {repr(fs_result)}")
    print(f"InMemoryStorage:   {repr(mem_result)}")
    print(f"Match: {fs_result == mem_result}")
```

<details>

<summary>
Storage backends behave inconsistently with line endings
</summary>
```
Test 1: Single CR character
Original:          '\r'
FileSystemStorage: '\n'
InMemoryStorage:   '\r'
Match: False

Test 2: CRLF sequence
Original:          '\r\n'
FileSystemStorage: '\n'
InMemoryStorage:   '\r\n'
Match: False

Test 3: Text with CR in the middle
Original:          'hello\rworld'
FileSystemStorage: 'hello\nworld'
InMemoryStorage:   'hello\rworld'
Match: False

Test 4: Windows-style text file
Original:          'line1\r\nline2\r\nline3'
FileSystemStorage: 'line1\nline2\nline3'
InMemoryStorage:   'line1\r\nline2\r\nline3'
Match: False
```
</details>

## Why This Is A Bug

1. **Violation of Storage Backend Equivalence**: Django's storage backends are designed to be interchangeable. The Django documentation states that storage backends provide a "standardized API for storing files". When two backends implementing the same interface behave differently for identical operations, it violates the substitutability principle that allows developers to swap backends without changing application behavior.

2. **Silent Data Corruption**: The FileSystemStorage modifies file content without warning or documentation. When a user saves text with specific line endings and retrieves it, they expect to get back exactly what they saved. This is a fundamental property of any storage system - data integrity.

3. **Real-World Impact**: This affects applications that:
   - Process Windows text files with CRLF line endings
   - Store configuration files where line endings have semantic meaning
   - Calculate checksums or digital signatures of text files
   - Migrate between storage backends (data will change during migration)
   - Use version control systems that track line ending changes

4. **Undocumented Behavior**: Neither the Django documentation nor the FileSystemStorage API documentation mentions this line ending conversion. Python's documentation clearly states that text mode with default parameters performs universal newline translation, but Django users shouldn't need to understand Python's file I/O internals to use Django's storage API correctly.

## Relevant Context

The root cause is in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/files/storage/filesystem.py` at line 81:

```python
def _open(self, name, mode="rb"):
    return File(open(self.path(name), mode))
```

Python's `open()` function documentation ([Python docs](https://docs.python.org/3/library/functions.html#open)) explains that when opening files in text mode without specifying the `newline` parameter, Python enables "universal newline mode" which converts all line endings (`\r`, `\r\n`) to `\n`.

In contrast, InMemoryStorage uses its own `InMemoryFileNode` class (lines 40-79 in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/files/storage/memory.py`) which handles text/bytes conversion without applying newline translation.

**Workaround**: Users can open files in binary mode ('rb') to preserve line endings, but this requires handling text encoding manually and doesn't solve the backend inconsistency issue.

## Proposed Fix

```diff
--- a/django/core/files/storage/filesystem.py
+++ b/django/core/files/storage/filesystem.py
@@ -78,7 +78,11 @@ class FileSystemStorage(Storage, StorageSettingsMixin):
         )

     def _open(self, name, mode="rb"):
-        return File(open(self.path(name), mode))
+        # Preserve line endings in text mode for consistency with InMemoryStorage
+        # and to maintain data integrity during round-trip operations
+        if 'b' not in mode:
+            return File(open(self.path(name), mode, newline=''))
+        return File(open(self.path(name), mode))

     def _save(self, name, content):
         full_path = self.path(name)
```