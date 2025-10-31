# Bug Report: dask.bytes.core.read_block Length None Without Delimiter

**Target**: `dask.bytes.core.read_block`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `read_block` function raises `AssertionError` when `length=None` and `delimiter=None`, despite the docstring explicitly stating that `length=None` should "read through end of file if None".

## Property-Based Test

```python
from io import BytesIO
from hypothesis import given, strategies as st, assume
from dask.bytes.core import read_block

@given(st.binary(min_size=1, max_size=1000), st.integers(min_value=0, max_value=100))
def test_read_block_length_none_reads_to_end(data, offset):
    assume(offset < len(data))
    f = BytesIO(data)
    result = read_block(f, offset, None, delimiter=None)
    expected = data[offset:]
    assert result == expected
```

**Failing input**: `data=b'\x00', offset=0`

## Reproducing the Bug

```python
from io import BytesIO
from dask.bytes.core import read_block

data = b"Hello World!"
f = BytesIO(data)

result = read_block(f, 0, None, delimiter=None)
```

Output:
```
AssertionError
```

## Why This Is A Bug

The function's docstring clearly documents that the `length` parameter can be `None`:

> length: int
>     Number of bytes to read, read through end of file if None

However, the implementation contains `assert length is not None`, which prevents this documented behavior. The code even has a TODO comment acknowledging this:

```python
# TODO: allow length to be None and read to the end of the file?
assert length is not None
```

This is a contract violation where the implementation doesn't match the documented API. Users following the documentation will encounter an unexpected `AssertionError`.

Note that `length=None` works correctly when a `delimiter` is provided, making this inconsistency even more confusing.

## Fix

```diff
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -298,8 +298,10 @@ def read_block(

     f.seek(offset)

-    # TODO: allow length to be None and read to the end of the file?
-    assert length is not None
+    if length is None:
+        b = f.read()
+    else:
+        b = f.read(length)
-    b = f.read(length)
     return b
```