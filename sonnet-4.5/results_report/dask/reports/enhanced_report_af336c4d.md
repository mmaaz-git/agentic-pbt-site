# Bug Report: dask.dataframe.io.parquet _normalize_and_strip_protocol Returns Empty Strings

**Target**: `dask.dataframe.dask_expr.io.parquet._normalize_and_strip_protocol`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_normalize_and_strip_protocol` function returns empty strings when given paths consisting only of slashes (e.g., `/`, `///`) or protocol-only paths (e.g., `file:///`, `s3:///`), causing downstream `FileNotFoundError` in pyarrow operations.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis to verify that
_normalize_and_strip_protocol never returns empty strings
"""

from hypothesis import given, strategies as st, example

@given(st.lists(st.text(min_size=0), min_size=1))
@example(["/"])
@example(["///"])
@example(["file:///"])
@example(["s3:///"])
def test_no_empty_strings_in_normalized_paths(paths):
    """Test that normalized paths should never be empty strings."""
    from dask.dataframe.dask_expr.io.parquet import _normalize_and_strip_protocol
    result = _normalize_and_strip_protocol(paths)
    for r in result:
        assert r != "", f"Empty string found in result for input {paths}"

if __name__ == "__main__":
    # Run the test
    print("Running property-based test with Hypothesis...")
    print("=" * 60)
    try:
        test_no_empty_strings_in_normalized_paths()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nHypothesis will now attempt to find the minimal failing example...")

    # Run with hypothesis to get detailed output
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
```

<details>

<summary>
**Failing input**: `['/']`, `['///']`, `['file:///']`, `['s3:///']`
</summary>
```
Running property-based test with Hypothesis...
============================================================
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 26, in <module>
  |     test_no_empty_strings_in_normalized_paths()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 10, in test_no_empty_strings_in_normalized_paths
  |     @example(["/"])
  |
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 4 distinct failures in explicit examples. (4 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 19, in test_no_empty_strings_in_normalized_paths
    |     assert r != "", f"Empty string found in result for input {paths}"
    |            ^^^^^^^
    | AssertionError: Empty string found in result for input ['/']
    | Falsifying explicit example: test_no_empty_strings_in_normalized_paths(
    |     paths=['/'],
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 19, in test_no_empty_strings_in_normalized_paths
    |     assert r != "", f"Empty string found in result for input {paths}"
    |            ^^^^^^^
    | AssertionError: Empty string found in result for input ['///']
    | Falsifying explicit example: test_no_empty_strings_in_normalized_paths(
    |     paths=['///'],
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 19, in test_no_empty_strings_in_normalized_paths
    |     assert r != "", f"Empty string found in result for input {paths}"
    |            ^^^^^^^
    | AssertionError: Empty string found in result for input ['file:///']
    | Falsifying explicit example: test_no_empty_strings_in_normalized_paths(
    |     paths=['file:///'],
    | )
    +---------------- 4 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 19, in test_no_empty_strings_in_normalized_paths
    |     assert r != "", f"Empty string found in result for input {paths}"
    |            ^^^^^^^
    | AssertionError: Empty string found in result for input ['s3:///']
    | Falsifying explicit example: test_no_empty_strings_in_normalized_paths(
    |     paths=['s3:///'],
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction script demonstrating the bug in
dask.dataframe.dask_expr.io.parquet._normalize_and_strip_protocol
"""

from dask.dataframe.dask_expr.io.parquet import _normalize_and_strip_protocol

print("Testing _normalize_and_strip_protocol with problematic inputs...")
print()

# Test case 1: Single slash
print("Input: '/'")
result = _normalize_and_strip_protocol("/")
print(f"Output: {result}")
print(f"Contains empty string: {'' in result}")
print()

# Test case 2: Multiple slashes
print("Input: '///'")
result = _normalize_and_strip_protocol("///")
print(f"Output: {result}")
print(f"Contains empty string: {'' in result}")
print()

# Test case 3: file:/// protocol
print("Input: 'file:///'")
result = _normalize_and_strip_protocol("file:///")
print(f"Output: {result}")
print(f"Contains empty string: {'' in result}")
print()

# Test case 4: s3:/// protocol
print("Input: 's3:///'")
result = _normalize_and_strip_protocol("s3:///")
print(f"Output: {result}")
print(f"Contains empty string: {'' in result}")
print()

# Test case 5: List of problematic inputs
print("Input: ['/', '///', 'file:///']")
result = _normalize_and_strip_protocol(["/", "///", "file:///"])
print(f"Output: {result}")
print(f"Contains empty strings: {any(s == '' for s in result)}")
print()

# Demonstrate the downstream error when empty string is passed to pyarrow
print("=" * 60)
print("Demonstrating downstream error with pyarrow.fs.FileSelector:")
print("=" * 60)

import pyarrow as pa
import pyarrow.fs as pa_fs

try:
    # This is what happens downstream when an empty string is returned
    print("Attempting: pa_fs.FileSelector('', recursive=True)")
    selector = pa_fs.FileSelector("", recursive=True)
    fs = pa_fs.LocalFileSystem()
    files = fs.get_file_info(selector)
    print(f"Success: Found {len(files)} files")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
pyarrow.fs.FileSelector raises FileNotFoundError when given empty string
</summary>
```
Testing _normalize_and_strip_protocol with problematic inputs...

Input: '/'
Output: ['']
Contains empty string: True

Input: '///'
Output: ['']
Contains empty string: True

Input: 'file:///'
Output: ['']
Contains empty string: True

Input: 's3:///'
Output: ['']
Contains empty string: True

Input: ['/', '///', 'file:///']
Output: ['', '', '']
Contains empty strings: True

============================================================
Demonstrating downstream error with pyarrow.fs.FileSelector:
============================================================
Attempting: pa_fs.FileSelector('', recursive=True)
Error: FileNotFoundError: [Errno 2] Cannot list directory ''. Detail: [errno 2] No such file or directory
```
</details>

## Why This Is A Bug

This bug violates expected behavior in multiple ways:

1. **Empty strings are invalid filesystem paths**: No filesystem API accepts empty strings as valid paths. The Unix root directory "/" is a valid path that should be preserved, not stripped to "".

2. **Causes downstream failures in pyarrow**: The function's output is directly used in `ReadParquet.dataset_info` (line 1019) where it's passed to `pyarrow.fs.FileSelector`. When an empty string is passed, pyarrow raises `FileNotFoundError: [Errno 2] Cannot list directory ''`.

3. **Violates principle of least surprise**: A function named `_normalize_and_strip_protocol` should normalize paths to a canonical form, not produce invalid paths. The root directory "/" is already in its most normalized form.

4. **Breaks valid use cases**: Users may legitimately want to read parquet files from the root directory or use protocol-only paths as base directories for cloud storage operations.

## Relevant Context

The bug occurs in the internal function `_normalize_and_strip_protocol` at line 1834 of `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/parquet.py`. The function:

1. Strips protocol prefixes like "file://", "s3://", etc.
2. Then calls `rstrip("/")` to remove trailing slashes
3. When the path is only slashes or a protocol followed by only slashes, `rstrip("/")` removes everything, leaving an empty string

The function is used by the `ReadParquet` class's `normalized_path` property, which feeds into dataset discovery operations. The PyArrow documentation explicitly states that FileSelector requires a non-empty directory path.

Key code locations:
- Bug location: `dask/dataframe/dask_expr/io/parquet.py:1834`
- Usage in ReadParquet: `dask/dataframe/dask_expr/io/parquet.py:866`
- Downstream failure point: `dask/dataframe/dask_expr/io/parquet.py:1019`

## Proposed Fix

The function should preserve "/" as the minimum valid path when stripping would result in an empty string:

```diff
--- a/dask/dataframe/dask_expr/io/parquet.py
+++ b/dask/dataframe/dask_expr/io/parquet.py
@@ -1831,7 +1831,10 @@ def _normalize_and_strip_protocol(path):
             if len(split) > 1:
                 p = split[1]
                 break
-        result.append(p.rstrip("/"))
+        normalized = p.rstrip("/")
+        if not normalized and p:  # If stripping resulted in empty but input wasn't empty
+            normalized = "/"  # Preserve root directory
+        result.append(normalized)
     return result
```

This fix ensures that:
- The root directory "/" is preserved as "/" (not empty string)
- Protocol-only paths like "s3:///" become "/" (valid root reference)
- Normal paths like "/foo/bar/" become "/foo/bar" (unchanged behavior)
- Empty input remains empty (no change for already-empty strings)