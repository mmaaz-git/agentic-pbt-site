# Bug Report: dask.dataframe.dask_expr.io.parquet _normalize_and_strip_protocol Converts Root Paths to Empty Strings

**Target**: `dask.dataframe.dask_expr.io.parquet._normalize_and_strip_protocol`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_normalize_and_strip_protocol` function incorrectly converts valid filesystem root paths (like "/" or "///") into empty strings, causing downstream PyArrow FileSelector operations to fail with FileNotFoundError.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test that discovers the bug in _normalize_and_strip_protocol
where root paths like "/" get converted to empty strings.
"""

from hypothesis import given, strategies as st, settings, example
from dask.dataframe.dask_expr.io.parquet import _normalize_and_strip_protocol

@given(st.text(min_size=1))
@settings(max_examples=500)
@example("/")
@example("///")
@example("s3:///")
def test_normalize_and_strip_protocol_no_empty_strings(path):
    result = _normalize_and_strip_protocol(path)

    assert len(result) == 1
    assert result[0] != "", f"Result should not be empty string for input {path!r}"

if __name__ == "__main__":
    # Run the test
    test_normalize_and_strip_protocol_no_empty_strings()
```

<details>

<summary>
**Failing input**: `/`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 23, in <module>
  |     test_normalize_and_strip_protocol_no_empty_strings()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 11, in test_normalize_and_strip_protocol_no_empty_strings
  |     @settings(max_examples=500)
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures in explicit examples. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 19, in test_normalize_and_strip_protocol_no_empty_strings
    |     assert result[0] != "", f"Result should not be empty string for input {path!r}"
    |            ^^^^^^^^^^^^^^^
    | AssertionError: Result should not be empty string for input '/'
    | Falsifying explicit example: test_normalize_and_strip_protocol_no_empty_strings(
    |     path='/',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 19, in test_normalize_and_strip_protocol_no_empty_strings
    |     assert result[0] != "", f"Result should not be empty string for input {path!r}"
    |            ^^^^^^^^^^^^^^^
    | AssertionError: Result should not be empty string for input '///'
    | Falsifying explicit example: test_normalize_and_strip_protocol_no_empty_strings(
    |     path='///',
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 19, in test_normalize_and_strip_protocol_no_empty_strings
    |     assert result[0] != "", f"Result should not be empty string for input {path!r}"
    |            ^^^^^^^^^^^^^^^
    | AssertionError: Result should not be empty string for input 's3:///'
    | Falsifying explicit example: test_normalize_and_strip_protocol_no_empty_strings(
    |     path='s3:///',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Demonstration of the bug in dask.dataframe.dask_expr.io.parquet._normalize_and_strip_protocol
This function incorrectly returns empty strings for root path inputs.
"""

from dask.dataframe.dask_expr.io.parquet import _normalize_and_strip_protocol

print("Testing _normalize_and_strip_protocol with various root path inputs:")
print("-" * 70)

# Test 1: Single forward slash (root directory)
result = _normalize_and_strip_protocol("/")
print(f"Input: '/'")
print(f"Output: {result!r}")
print(f"Result is empty string: {result == ['']}")
print()

# Test 2: Multiple forward slashes
result = _normalize_and_strip_protocol("///")
print(f"Input: '///'")
print(f"Output: {result!r}")
print(f"Result is empty string: {result == ['']}")
print()

# Test 3: Protocol with only slashes
result = _normalize_and_strip_protocol("s3:///")
print(f"Input: 's3:///'")
print(f"Output: {result!r}")
print(f"Result is empty string: {result == ['']}")
print()

# Test 4: Mixed paths including root
result = _normalize_and_strip_protocol(["s3://bucket/", "/"])
print(f"Input: ['s3://bucket/', '/']")
print(f"Output: {result!r}")
print(f"Contains empty string: {'' in result}")
print()

# Demonstrate downstream failure with PyArrow
print("-" * 70)
print("Demonstrating downstream failure with PyArrow FileSelector:")
print()

import pyarrow.fs as pa_fs
import pyarrow as pa

# This works fine with "/"
try:
    fs = pa_fs.LocalFileSystem()
    selector = pa_fs.FileSelector("/", recursive=False)
    print("FileSelector with '/' path: SUCCESS")
    # We can even list files (limiting output for brevity)
    files = fs.get_file_info(selector)[:3]
    print(f"  Found {len(fs.get_file_info(selector))} items in root directory")
except Exception as e:
    print(f"FileSelector with '/' path: FAILED - {e}")

print()

# This fails with empty string
try:
    fs = pa_fs.LocalFileSystem()
    selector = pa_fs.FileSelector("", recursive=False)
    print("FileSelector with empty string: SUCCESS")
    files = fs.get_file_info(selector)
    print(f"  Found {len(files)} items")
except Exception as e:
    print(f"FileSelector with empty string: FAILED - {e}")

print()
print("-" * 70)
print("CONCLUSION: The function transforms valid paths ('/') into invalid ones ('')")
print("This causes downstream failures in PyArrow operations.")
```

<details>

<summary>
FileNotFoundError when PyArrow tries to use empty string path
</summary>
```
Testing _normalize_and_strip_protocol with various root path inputs:
----------------------------------------------------------------------
Input: '/'
Output: ['']
Result is empty string: True

Input: '///'
Output: ['']
Result is empty string: True

Input: 's3:///'
Output: ['']
Result is empty string: True

Input: ['s3://bucket/', '/']
Output: ['bucket', '']
Contains empty string: True

----------------------------------------------------------------------
Demonstrating downstream failure with PyArrow FileSelector:

FileSelector with '/' path: SUCCESS
  Found 26 items in root directory

FileSelector with empty string: SUCCESS
FileSelector with empty string: FAILED - [Errno 2] Cannot list directory ''. Detail: [errno 2] No such file or directory

----------------------------------------------------------------------
CONCLUSION: The function transforms valid paths ('/') into invalid ones ('')
This causes downstream failures in PyArrow operations.
```
</details>

## Why This Is A Bug

The `_normalize_and_strip_protocol` function is designed to normalize file paths by removing protocol prefixes (like "s3://") and trailing slashes. However, when processing paths that consist entirely of slashes (such as "/", "///", or "s3:///"), the function's use of `rstrip("/")` at line 1834 removes all slashes, resulting in an empty string.

This behavior violates several important expectations:

1. **Valid paths become invalid**: The Unix root directory "/" is a universally recognized valid filesystem path. After normalization, it becomes an empty string "", which is not a valid path in any filesystem context.

2. **PyArrow incompatibility**: The normalized paths are passed to PyArrow's FileSelector at line 1019 of parquet.py. PyArrow FileSelector accepts "/" as valid input (representing the root directory) but rejects empty strings with a FileNotFoundError: "Cannot list directory ''".

3. **Function contract violation**: A path normalization function should preserve the validity of paths. Converting a valid path to an invalid one breaks this implicit contract.

4. **Affects multiple input patterns**: The bug manifests for various inputs including "/", "///", "s3:///", and any path that reduces to only slashes after protocol stripping.

## Relevant Context

The `_normalize_and_strip_protocol` function is an internal/private function (indicated by the underscore prefix) located at line 1822-1835 in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/parquet.py`.

The function is used by `ReadParquetPyarrowFS.normalized_path` property (line 866), which then passes the normalized paths to PyArrow's FileSelector for directory listing operations (line 1019).

While reading parquet files from the filesystem root directory is an uncommon use case, the bug represents a logical error where valid input is transformed into invalid output. The fix is straightforward and would prevent potential confusion and errors for users who might encounter this edge case.

PyArrow FileSelector documentation: https://arrow.apache.org/docs/python/generated/pyarrow.fs.FileSelector.html

## Proposed Fix

```diff
--- a/dask/dataframe/dask_expr/io/parquet.py
+++ b/dask/dataframe/dask_expr/io/parquet.py
@@ -1831,7 +1831,11 @@ def _normalize_and_strip_protocol(path):
             if len(split) > 1:
                 p = split[1]
                 break
-        result.append(p.rstrip("/"))
+        # Strip trailing slashes but preserve root path
+        normalized = p.rstrip("/")
+        if not normalized and p.startswith("/"):
+            # If stripping resulted in empty string from a slash-only path, preserve as root
+            normalized = "/"
+        result.append(normalized)
     return result
```