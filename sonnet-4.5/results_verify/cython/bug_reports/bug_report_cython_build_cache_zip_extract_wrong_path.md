# Bug Report: Cython.Build.Cache load_from_cache Extracts Zip Files to Wrong Path

**Target**: `Cython.Build.Cache.load_from_cache`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `load_from_cache()` method incorrectly passes `join_path(dirname, artifact)` as the extraction path to `zipfile.ZipFile.extract()`, which expects only a directory path, not a full file path. This causes files to be extracted to the wrong location.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Build.Cache import Cache
import os
import tempfile
import zipfile


@given(st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=3, max_size=10))
def test_load_from_cache_zip_extracts_to_correct_location(artifact_name):
    artifact_name = artifact_name + ".c"
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Cache(tmpdir)

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        c_file = os.path.join(output_dir, "main.c")

        zip_path = os.path.join(tmpdir, "cached.zip")
        with zipfile.ZipFile(zip_path, 'w') as z:
            z.writestr(artifact_name, "// content")

        cache.load_from_cache(c_file, zip_path)

        expected_path = os.path.join(output_dir, artifact_name)
        assert os.path.exists(expected_path), f"File should be extracted to {expected_path}"
```

**Failing input**: Any zip file with artifacts will extract to wrong paths

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Cache import Cache
import os
import tempfile
import zipfile

with tempfile.TemporaryDirectory() as tmpdir:
    cache = Cache(tmpdir)

    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(output_dir)
    c_file = os.path.join(output_dir, "main.c")

    zip_path = os.path.join(tmpdir, "cached.zip")
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr("main.c", "// main content")
        z.writestr("main.h", "// header content")

    cache.load_from_cache(c_file, zip_path)

    expected_c = os.path.join(output_dir, "main.c")
    expected_h = os.path.join(output_dir, "main.h")

    print(f"Expected main.c at: {expected_c}")
    print(f"Exists: {os.path.exists(expected_c)}")
    print(f"Expected main.h at: {expected_h}")
    print(f"Exists: {os.path.exists(expected_h)}")

    wrong_c = os.path.join(output_dir, "main.c", "main.c")
    wrong_h = os.path.join(output_dir, "main.h", "main.h")
    print(f"\nWrong location main.c at: {wrong_c}")
    print(f"Exists: {os.path.exists(wrong_c)}")
```

## Why This Is A Bug

The `zipfile.ZipFile.extract(member, path=None)` method expects `path` to be a directory where the file will be extracted, not the full target file path. By passing `join_path(dirname, artifact)` instead of just `dirname`, the code attempts to create nested directories with the artifact name as a directory, leading to incorrect extraction paths and potential filesystem errors.

## Fix

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