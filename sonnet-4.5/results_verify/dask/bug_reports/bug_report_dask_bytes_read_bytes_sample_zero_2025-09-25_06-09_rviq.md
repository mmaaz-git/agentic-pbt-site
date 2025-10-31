# Bug Report: dask.bytes.read_bytes sample=0 Returns Integer Instead of Bytes

**Target**: `dask.bytes.read_bytes`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When `read_bytes` is called with `sample=0`, it returns the integer `0` instead of an empty bytes object `b''`, violating the documented return type.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.bytes.core import read_bytes
import tempfile
import os


@given(st.integers(min_value=0, max_value=1000))
@settings(max_examples=200)
def test_sample_return_type_with_integer(sample_size):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'test.txt')

        test_data = b'x' * 1000
        with open(test_file, 'wb') as f:
            f.write(test_data)

        sample, blocks = read_bytes(test_file, sample=sample_size, blocksize=None)

        assert isinstance(sample, bytes), \
            f"sample={sample_size} should return bytes, got {type(sample).__name__}"
```

**Failing input**: `sample_size=0`

## Reproducing the Bug

```python
import tempfile
import os
from dask.bytes.core import read_bytes

with tempfile.TemporaryDirectory() as tmpdir:
    test_file = os.path.join(tmpdir, 'test.txt')
    with open(test_file, 'wb') as f:
        f.write(b'hello world')

    sample, blocks = read_bytes(test_file, sample=0, blocksize=None)

    print(f"Type: {type(sample)}")
    print(f"Value: {sample!r}")

    sample.decode('utf-8')
```

Output:
```
Type: <class 'int'>
Value: 0
AttributeError: 'int' object has no attribute 'decode'
```

## Why This Is A Bug

The `read_bytes` docstring specifies:

```python
Parameters
----------
sample : int, string, or boolean
    Whether or not to return a header sample.
    Values can be ``False`` for "no sample requested"
    Or an integer or string value like ``2**20`` or ``"1 MiB"``

Returns
-------
sample : bytes
    The sample header
```

When `sample=0` (an integer), the function should read 0 bytes and return `b''` (empty bytes), not the integer `0`. The current behavior violates the documented return type, breaking code that expects bytes.

The issue is in the sample handling logic where `if sample:` treats 0 as falsy, skipping the read and returning the integer unchanged.

## Fix

```diff
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -160,7 +160,7 @@ def read_bytes(
             )
         out.append(values)

-    if sample:
+    if sample is not False:
         if sample is True:
             sample = "10 kiB"  # backwards compatibility
         if isinstance(sample, str):
```

This fix ensures that `sample=0` is treated as "read 0 bytes" rather than "no sample", while preserving the special case where `sample=False` means "no sample requested".