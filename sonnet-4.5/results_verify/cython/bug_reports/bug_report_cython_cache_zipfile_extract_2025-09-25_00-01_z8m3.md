# Bug Report: Cython.Build.Cache.load_from_cache Incorrect Zipfile Extraction Path

**Target**: `Cython.Build.Cache.load_from_cache`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `load_from_cache` method incorrectly extracts files from zip archives to a nested subdirectory instead of the intended directory, causing cached compilation artifacts to be placed in the wrong location.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Build.Cache import Cache
import tempfile
import zipfile
import os

@given(st.text(alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')), min_size=1, max_size=20))
@settings(max_examples=100)
def test_cache_load_extracts_to_correct_location(artifact_name):
    artifact_name = artifact_name + '.c'
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Cache(tmpdir)
        zip_path = os.path.join(tmpdir, 'test.zip')
        with zipfile.ZipFile(zip_path, 'w') as z:
            z.writestr(artifact_name, 'content')

        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(output_dir)
        c_file = os.path.join(output_dir, 'test.c')

        cache.load_from_cache(c_file, zip_path)

        expected = os.path.join(output_dir, artifact_name)
        assert os.path.exists(expected), f'File should be at {expected}'
```

**Failing input**: Any artifact name (e.g., `'output.c'`)

## Reproducing the Bug

```python
from Cython.Build.Cache import Cache
import tempfile
import zipfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    cache = Cache(tmpdir)

    zip_path = os.path.join(tmpdir, 'cached.zip')
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr('output.c', 'int main() { return 0; }')

    output_dir = os.path.join(tmpdir, 'build')
    os.makedirs(output_dir)
    c_file = os.path.join(output_dir, 'test.c')

    cache.load_from_cache(c_file, zip_path)

    expected = os.path.join(output_dir, 'output.c')
    actual = os.path.join(output_dir, 'output.c', 'output.c')

    print(f'Expected location: {expected}')
    print(f'Exists: {os.path.exists(expected)}')
    print(f'\nActual location: {actual}')
    print(f'Exists: {os.path.exists(actual)}')
```

Output:
```
Expected location: /tmp/.../build/output.c
Exists: False

Actual location: /tmp/.../build/output.c/output.c
Exists: True
```

## Why This Is A Bug

The `zipfile.ZipFile.extract(member, path)` method extracts `member` to the directory `path`, creating `path/member`. However, the code passes `join_path(dirname, artifact)` as the path, resulting in files being extracted to `dirname/artifact/artifact` instead of `dirname/artifact`.

This breaks the caching mechanism - when compilation results are loaded from the cache, they won't be found at their expected locations, causing compilation failures or forcing unnecessary recompilation.

## Fix

```diff
--- a/Cache.py
+++ b/Cache.py
@@ -152,7 +152,7 @@ class Cache:
             dirname = os.path.dirname(c_file)
             with zipfile.ZipFile(cached) as z:
                 for artifact in z.namelist():
-                    z.extract(artifact, join_path(dirname, artifact))
+                    z.extract(artifact, dirname)
         else:
             raise ValueError(f"Unsupported cache file extension: {ext}")
```