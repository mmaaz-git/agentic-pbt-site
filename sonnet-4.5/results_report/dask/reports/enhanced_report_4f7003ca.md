# Bug Report: fsspec.utils.read_block AssertionError When length=None Without Delimiter

**Target**: `fsspec.utils.read_block`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `read_block` function raises an `AssertionError` when called with `length=None` and `delimiter=None`, directly contradicting its docstring which states that `length=None` should "read through end of file".

## Property-Based Test

```python
from io import BytesIO
from hypothesis import given, strategies as st, assume
from fsspec.utils import read_block

@given(st.binary(min_size=1, max_size=1000), st.integers(min_value=0, max_value=100))
def test_read_block_length_none_reads_to_end(data, offset):
    assume(offset < len(data))
    f = BytesIO(data)
    result = read_block(f, offset, None, delimiter=None)
    expected = data[offset:]
    assert result == expected

if __name__ == "__main__":
    test_read_block_length_none_reads_to_end()
```

<details>

<summary>
**Failing input**: `data=b'\x00', offset=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 14, in <module>
    test_read_block_length_none_reads_to_end()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 6, in test_read_block_length_none_reads_to_end
    def test_read_block_length_none_reads_to_end(data, offset):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 9, in test_read_block_length_none_reads_to_end
    result = read_block(f, offset, None, delimiter=None)
  File "/home/npc/miniconda/lib/python3.13/site-packages/fsspec/utils.py", line 303, in read_block
    assert length is not None
           ^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_read_block_length_none_reads_to_end(
    data=b'\x00',  # or any other generated value
    offset=0,
)
```
</details>

## Reproducing the Bug

```python
from io import BytesIO
from fsspec.utils import read_block

# Test case demonstrating the bug
data = b"Hello World!"
f = BytesIO(data)

print("Attempting to call read_block with length=None and delimiter=None...")
try:
    result = read_block(f, 0, None, delimiter=None)
    print(f"Success! Result: {result}")
except AssertionError as e:
    print(f"AssertionError raised!")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
AssertionError raised when attempting to read with length=None
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/repo.py", line 10, in <module>
    result = read_block(f, 0, None, delimiter=None)
  File "/home/npc/miniconda/lib/python3.13/site-packages/fsspec/utils.py", line 303, in read_block
    assert length is not None
           ^^^^^^^^^^^^^^^^^^
AssertionError
Attempting to call read_block with length=None and delimiter=None...
AssertionError raised!
```
</details>

## Why This Is A Bug

This is a clear contract violation where the implementation directly contradicts the documented API. The function's docstring at line 249 explicitly states:

```
length: int
    Number of bytes to read, read through end of file if None
```

However, the implementation contains an assertion at line 303 that explicitly prevents this documented behavior:

```python
# TODO: allow length to be None and read to the end of the file?
assert length is not None
```

The TODO comment itself acknowledges that this is a gap between the documentation and implementation. This creates several problems:

1. **API Inconsistency**: The function behaves differently depending on whether a delimiter is provided. When `delimiter` is provided with `length=None`, the function works correctly and reads to the end of file (the code returns early at line 281). However, without a delimiter, it raises an AssertionError.

2. **User Expectations**: Users following the documentation will encounter an unexpected AssertionError when using the documented feature of passing `length=None` to read to end of file.

3. **Common Use Case**: Reading to end of file is a fundamental I/O operation that users reasonably expect to work based on the documentation.

## Relevant Context

- The `read_block` function is part of the `fsspec` library (filesystem spec), which is a foundational library for file system operations used by Dask and many other data processing libraries.
- The function is imported and re-exported by `dask.bytes.core`, making this issue visible to Dask users as well.
- Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/fsspec/utils.py` lines 234-305
- The assertion that causes the failure is at line 303
- When a delimiter is provided, the code takes a different path (lines 277-281) that bypasses the assertion, allowing `length=None` to work correctly

Testing confirms the inconsistent behavior:
- `read_block(f, 0, None, delimiter=None)` - Raises AssertionError
- `read_block(f, 0, None, delimiter=b'\n')` - Works correctly, returns entire file content

## Proposed Fix

```diff
--- a/fsspec/utils.py
+++ b/fsspec/utils.py
@@ -299,8 +299,10 @@ def read_block(

     f.seek(offset)

-    # TODO: allow length to be None and read to the end of the file?
-    assert length is not None
-    b = f.read(length)
+    if length is None:
+        b = f.read()
+    else:
+        b = f.read(length)
+
     return b
```