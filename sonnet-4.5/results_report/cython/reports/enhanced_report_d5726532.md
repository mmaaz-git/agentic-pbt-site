# Bug Report: Cython.Build.Cache.load_from_cache Creates Nested Subdirectories When Extracting Zip Archives

**Target**: `Cython.Build.Cache.load_from_cache`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `load_from_cache` method incorrectly extracts cached files from zip archives into nested subdirectories (e.g., `/path/output.c/output.c`) instead of the intended directory (`/path/output.c`), breaking the caching mechanism for multi-file Cython modules.

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
        # Check if it's a file, not a directory
        assert os.path.isfile(expected), f'File should be at {expected}, but found directory={os.path.isdir(expected)}, file={os.path.isfile(expected)}'

if __name__ == "__main__":
    test_cache_load_extracts_to_correct_location()
```

<details>

<summary>
**Failing input**: `artifact_name='a'` (or any other generated value)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 28, in <module>
    test_cache_load_extracts_to_correct_location()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 8, in test_cache_load_extracts_to_correct_location
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 25, in test_cache_load_extracts_to_correct_location
    assert os.path.isfile(expected), f'File should be at {expected}, but found directory={os.path.isdir(expected)}, file={os.path.isfile(expected)}'
           ~~~~~~~~~~~~~~^^^^^^^^^^
AssertionError: File should be at /tmp/tmpr0k2tev3/output/a.c, but found directory=True, file=False
Falsifying example: test_cache_load_extracts_to_correct_location(
    artifact_name='a',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from Cython.Build.Cache import Cache
import tempfile
import zipfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    cache = Cache(tmpdir)

    # Create a cached zip file with an artifact
    zip_path = os.path.join(tmpdir, 'cached.zip')
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr('output.c', 'int main() { return 0; }')

    # Create output directory structure
    output_dir = os.path.join(tmpdir, 'build')
    os.makedirs(output_dir)
    c_file = os.path.join(output_dir, 'test.c')

    # Load from cache
    cache.load_from_cache(c_file, zip_path)

    # Check where files actually ended up
    expected = os.path.join(output_dir, 'output.c')
    actual = os.path.join(output_dir, 'output.c', 'output.c')

    print(f'Expected location: {expected}')
    print(f'Exists as file: {os.path.isfile(expected)}')
    print(f'Exists as directory: {os.path.isdir(expected)}')
    print(f'\nActual location: {actual}')
    print(f'Exists as file: {os.path.isfile(actual)}')

    # Show directory structure
    print(f'\nDirectory structure in {output_dir}:')
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        sub_indent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{sub_indent}{file}')
```

<details>

<summary>
Files extracted to incorrect nested subdirectory
</summary>
```
Expected location: /tmp/tmp_d2heyam/build/output.c
Exists as file: False
Exists as directory: True

Actual location: /tmp/tmp_d2heyam/build/output.c/output.c
Exists as file: True

Directory structure in /tmp/tmp_d2heyam/build:
build/
  output.c/
    output.c
```
</details>

## Why This Is A Bug

This violates the expected behavior of Cython's caching mechanism. The bug occurs because `zipfile.ZipFile.extract(member, path)` extracts the member to `path/member`, but the code incorrectly passes `join_path(dirname, artifact)` as the path parameter. Since `artifact` is already the filename (e.g., 'output.c'), this results in double nesting: the file gets extracted to `dirname/output.c/output.c` instead of `dirname/output.c`.

The Python documentation for `zipfile.extract()` states: "Extract a member from the archive to the current working directory; member must be its full name or a ZipInfo object. Its file information is extracted as accurately as possible. path specifies a different directory to extract to." The method appends the member name to the path, so passing a path that already includes the filename causes the unintended nesting.

This breaks the caching optimization for Cython modules that generate multiple output files (those with public/api declarations), forcing unnecessary recompilation and defeating the purpose of the cache.

## Relevant Context

The bug is in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Build/Cache.py:155`. The caching system works correctly for single-file outputs (using gzip), but fails for multi-file outputs (using zip).

The `store_to_cache` method correctly stores files with just their basenames in the zip archive (line 172: `zip.write(artifact, os.path.basename(artifact))`), but `load_from_cache` tries to extract them to the wrong location due to the misuse of `join_path`.

Multi-file outputs occur when Cython modules use public/api declarations, generating additional header files (.h, _api.h) alongside the main .c file. These are legitimate and common use cases in Cython development.

## Proposed Fix

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