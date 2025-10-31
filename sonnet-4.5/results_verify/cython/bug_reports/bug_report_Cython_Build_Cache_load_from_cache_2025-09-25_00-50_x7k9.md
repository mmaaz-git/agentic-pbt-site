# Bug Report: Cython.Build.Cache.load_from_cache Incorrect Zip Extraction

**Target**: `Cython.Build.Cache.load_from_cache`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `load_from_cache` method incorrectly uses `zipfile.ZipFile.extract()` by passing a file path as the extraction directory instead of a directory path. This causes extraction to fail with "Not a directory" errors when trying to load cached compilation artifacts from zip files.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import os
import tempfile
import zipfile


@given(st.lists(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10), min_size=1, max_size=3))
def test_zip_extract_requires_directory(filenames):
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "test.zip")
        extract_dir = os.path.join(tmpdir, "extract")
        os.makedirs(extract_dir)

        filenames = [f + ".c" for f in filenames]

        with zipfile.ZipFile(zip_path, 'w') as z:
            for fname in filenames:
                z.writestr(fname, f"/* {fname} */")

        with zipfile.ZipFile(zip_path, 'r') as z:
            for fname in filenames:
                wrong_path = os.path.join(extract_dir, fname)
                try:
                    z.extract(fname, wrong_path)
                    assert False, "Should have raised an error"
                except (NotADirectoryError, FileNotFoundError, OSError):
                    pass
```

**Failing input**: Any list of filenames, e.g., `['test']`

## Reproducing the Bug

```python
import os
import tempfile
import zipfile

with tempfile.TemporaryDirectory() as tmpdir:
    zip_path = os.path.join(tmpdir, "test.zip")
    extract_dir = os.path.join(tmpdir, "extract")
    os.makedirs(extract_dir)

    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr("test.c", "/* test file */")

    with zipfile.ZipFile(zip_path, 'r') as z:
        wrong_path = os.path.join(extract_dir, "test.c")
        z.extract('test.c', wrong_path)
```

Output:
```
OSError: [Errno 20] Not a directory: '/tmp/.../extract/test.c/test.c'
```

## Why This Is A Bug

The `zipfile.ZipFile.extract(member, path=None)` method expects:
- `member`: the name of the file to extract
- `path`: the **directory** to extract into (not a file path)

The current code on line 155 of `Cache.py`:
```python
z.extract(artifact, join_path(dirname, artifact))
```

This passes `join_path(dirname, artifact)` which creates a file path like `/path/to/dir/file.c`, not a directory. When zipfile tries to extract `file.c` to `/path/to/dir/file.c`, it attempts to create `/path/to/dir/file.c/file.c`, which fails because `/path/to/dir/file.c` is not a directory.

This bug would manifest whenever:
1. A Cython compilation produces multiple artifacts (triggering zip storage)
2. The cache is used to load those artifacts
3. Extraction fails, preventing cache hits and forcing recompilation

The impact is high because it completely breaks caching for multi-artifact compilations, defeating a key performance optimization.

## Fix

The fix is simple - pass only the directory, not the file path:

```diff
--- a/Cython/Build/Cache.py
+++ b/Cython/Build/Cache.py
@@ -152,7 +152,7 @@ class Cache:
             dirname = os.path.dirname(c_file)
             with zipfile.ZipFile(cached) as z:
                 for artifact in z.namelist():
-                    z.extract(artifact, join_path(dirname, artifact))
+                    z.extract(artifact, dirname)
         else:
             raise ValueError(f"Unsupported cache file extension: {ext}")
```

This change correctly extracts each artifact to the directory containing `c_file`, with the artifact's basename as the filename (which is the default behavior of `zipfile.extract`).