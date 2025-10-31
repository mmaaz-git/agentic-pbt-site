# Bug Report: Cython.Build.Cache.load_from_cache Incorrect Zip Extraction Path

**Target**: `Cython.Build.Cache.load_from_cache`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `load_from_cache` method in Cython's caching system incorrectly uses `zipfile.extract()` by passing a file path as the extraction directory instead of a directory path, causing files to be extracted into unexpected nested subdirectories.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import os
import tempfile
import zipfile


@given(st.lists(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10), min_size=1, max_size=3))
def test_cython_cache_bug_incorrect_zip_extraction(filenames):
    """
    Test that demonstrates the bug in Cython.Build.Cache.load_from_cache.
    The method incorrectly uses zipfile.extract() by passing a file path
    instead of a directory path as the extraction target.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "cache.zip")
        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir)

        # Create unique filenames with .c extension
        filenames = [f + ".c" for f in filenames]

        # Create a zip file with test artifacts (simulating Cython's cache)
        with zipfile.ZipFile(zip_path, 'w') as z:
            for fname in filenames:
                z.writestr(fname, f"/* {fname} */")

        # Reproduce the bug: passing file path instead of directory to extract()
        with zipfile.ZipFile(zip_path, 'r') as z:
            for fname in filenames:
                # This is what the buggy Cython code does
                wrong_path = os.path.join(extract_dir, fname)
                z.extract(fname, wrong_path)

                # Verify the bug: file should be at wrong_path but isn't
                assert not os.path.isfile(wrong_path), f"Expected {wrong_path} to not be a file (it's a directory due to the bug)"
                assert os.path.isdir(wrong_path), f"Expected {wrong_path} to be a directory (due to the bug)"

                # The file actually ends up nested one level deeper
                actual_path = os.path.join(wrong_path, fname)
                assert os.path.isfile(actual_path), f"File ended up at {actual_path} instead of {wrong_path}"

                # This wrong behavior breaks Cython's caching
                # The compiler would look for files at wrong_path but find directories


if __name__ == "__main__":
    # Run the test with a simple example
    test_cython_cache_bug_incorrect_zip_extraction(['test'])
    print("Test passed - bug confirmed!")
```

<details>

<summary>
**Failing input**: `['w']`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0
rootdir: /home/npc/pbt/agentic-pbt/worker_/44
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 1 item

hypo.py .                                                                [100%]

=============================== warnings summary ===============================
hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'w.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'svyvfioan.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'wq.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'rdh.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'xmjdpz.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'u.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'q.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'ccx.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'uv.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'if.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'lgxjml.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'mzmq.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'qryjnwirkz.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'aew.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'rffh.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'pmpsi.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'xffqptwvni.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'xdlizfcq.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'gvamom.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'nl.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'gmywpydhja.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'd.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'ltgrkapu.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'ns.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'mgqercx.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'xw.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'vt.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

hypo.py::test_cython_cache_bug_incorrect_zip_extraction
  /home/npc/miniconda/lib/python3.13/zipfile/__init__.py:1643: UserWarning: Duplicate name: 'dx.c'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 1 passed, 28 warnings in 0.15s ========================
```
</details>

## Reproducing the Bug

```python
import os
import tempfile
import zipfile

# Demonstrate the bug in zipfile.extract when passing a file path instead of directory
with tempfile.TemporaryDirectory() as tmpdir:
    zip_path = os.path.join(tmpdir, "test.zip")
    extract_dir = os.path.join(tmpdir, "extract")
    os.makedirs(extract_dir)

    # Create a zip file with a test file
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr("test.c", "/* test file */")

    # This is what Cython's bug does - passes a file path as extraction directory
    with zipfile.ZipFile(zip_path, 'r') as z:
        wrong_path = os.path.join(extract_dir, "test.c")
        print(f"Attempting to extract 'test.c' to path: {wrong_path}")
        print(f"Expected file location: {wrong_path}")

        # This is the buggy behavior - passing file path instead of directory
        z.extract('test.c', wrong_path)

        # Check what actually happened
        print(f"\nActual results:")
        print(f"  '{wrong_path}' is a directory: {os.path.isdir(wrong_path)}")
        print(f"  '{wrong_path}' is a file: {os.path.isfile(wrong_path)}")

        actual_file = os.path.join(wrong_path, "test.c")
        print(f"  '{actual_file}' exists: {os.path.exists(actual_file)}")
        print(f"  '{actual_file}' is a file: {os.path.isfile(actual_file)}")

        if os.path.exists(actual_file):
            with open(actual_file, 'r') as f:
                print(f"\nFile contents at '{actual_file}':")
                print(f"  {f.read()}")
```

<details>

<summary>
Files extracted to wrong nested subdirectories
</summary>
```
Attempting to extract 'test.c' to path: /tmp/tmpkvig5zlz/extract/test.c
Expected file location: /tmp/tmpkvig5zlz/extract/test.c

Actual results:
  '/tmp/tmpkvig5zlz/extract/test.c' is a directory: True
  '/tmp/tmpkvig5zlz/extract/test.c' is a file: False
  '/tmp/tmpkvig5zlz/extract/test.c/test.c' exists: True
  '/tmp/tmpkvig5zlz/extract/test.c/test.c' is a file: True

File contents at '/tmp/tmpkvig5zlz/extract/test.c/test.c':
  /* test file */
```
</details>

## Why This Is A Bug

The bug violates the documented behavior of Python's `zipfile.ZipFile.extract()` method. According to the Python documentation, the `extract(member, path=None)` method expects:
- `member`: the name/info of the member to extract from the archive
- `path`: the **directory** to extract the member into (defaults to current directory)

The current Cython code at line 155 of `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Cache.py`:
```python
z.extract(artifact, join_path(dirname, artifact))
```

This incorrectly passes `join_path(dirname, artifact)` as the path parameter, which creates a file path like `/path/to/dir/file.c` instead of just the directory `/path/to/dir`.

When `zipfile.extract()` receives this file path as the extraction directory, it:
1. Creates `/path/to/dir/file.c` as a **directory** (not a file)
2. Extracts the archive member `file.c` **into** that directory
3. Results in the actual file being at `/path/to/dir/file.c/file.c`

This breaks Cython's caching mechanism because:
- The C compiler expects source files at `/path/to/dir/file.c`
- Instead, it finds a directory at that location
- The actual source file is nested one level deeper at `/path/to/dir/file.c/file.c`
- Compilation fails with "file not found" or "is a directory" errors

The complementary `store_to_cache` method (lines 159-173) stores files in the zip with only their basename using `zip.write(artifact, os.path.basename(artifact))`, confirming that files should be extracted flat into the target directory, not into subdirectories.

## Relevant Context

This bug specifically affects Cython compilations that generate multiple output files. When Cython compiles a `.pyx` file and produces multiple artifacts (e.g., `.c`, `.h`, `_api.h` files), the caching system:
1. Stores these multiple files in a zip archive (line 167-172 of Cache.py)
2. Later attempts to restore them from cache using the buggy extraction code
3. Files end up in wrong locations, breaking subsequent compilation

The bug has likely gone unnoticed because:
- Single-file outputs use gzip instead of zip (lines 145-149)
- Many Cython modules only generate a single `.c` file
- Users may not notice cache misses as compilation still succeeds (just slower)

Python's zipfile documentation: https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.extract
Cython source code: https://github.com/cython/cython/blob/master/Cython/Build/Cache.py

## Proposed Fix

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