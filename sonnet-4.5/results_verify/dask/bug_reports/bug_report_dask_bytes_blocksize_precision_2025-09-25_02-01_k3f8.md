# Bug Report: dask.bytes.read_bytes Block Size Precision Error

**Target**: `dask.bytes.core.read_bytes`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_bytes` function uses float division to calculate adjusted block sizes when a file size is not evenly divisible by the requested block size. This causes precision errors in offset calculations, leading to incorrect block boundaries and potential data integrity issues.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import tempfile
import os
from dask.bytes.core import read_bytes
from dask import compute
from tlz import concat

@given(
    file_size=st.integers(min_value=100, max_value=10000),
    blocksize=st.integers(min_value=50, max_value=1000)
)
@settings(max_examples=200)
def test_blocksize_precision_property(file_size, blocksize):
    assume(file_size > blocksize)
    assume(file_size % blocksize != 0)

    content = b'x' * file_size

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(content)
        f.flush()
        temp_path = f.name

    try:
        sample, blocks = read_bytes(temp_path, blocksize=blocksize)
        computed_blocks = compute(*concat(blocks), scheduler="sync")
        total_length = sum(len(block) for block in computed_blocks)

        assert total_length == file_size, \
            f"Length mismatch: file_size={file_size}, blocksize={blocksize}, " \
            f"total_length={total_length}, difference={total_length - file_size}"
    finally:
        os.unlink(temp_path)
```

**Failing input**: `file_size=1000, blocksize=300`

## Reproducing the Bug

```python
import tempfile
import os
from dask.bytes.core import read_bytes
from dask import compute
from tlz import concat

file_size = 1000
blocksize = 300
content = b'x' * file_size

with tempfile.NamedTemporaryFile(delete=False) as f:
    f.write(content)
    f.flush()
    temp_path = f.name

try:
    sample, blocks = read_bytes(temp_path, blocksize=blocksize)
    computed_blocks = compute(*concat(blocks), scheduler="sync")
    total_length = sum(len(block) for block in computed_blocks)

    print(f"Expected: {file_size}")
    print(f"Actual: {total_length}")
    print(f"Bug: {total_length != file_size}")
finally:
    os.unlink(temp_path)
```

## Why This Is A Bug

The `read_bytes` function is responsible for splitting files into chunks for parallel processing. When a file size is not evenly divisible by the block size, the code attempts to calculate an adjusted block size to distribute bytes evenly across blocks. However, on line 125 of `dask/bytes/core.py`, it uses float division:

```python
blocksize1 = size / (size // blocksize)  # Float division
```

This float value is then used in arithmetic operations (line 135):

```python
place += blocksize1  # Accumulating float values
```

And converted to int for offsets (line 136):

```python
off.append(int(place))  # Truncating accumulated float
```

This causes precision errors in calculating block boundaries. For example, with `file_size=1000` and `blocksize=300`:
- Current code: `blocksize1 = 1000 / 3 = 333.333...` (float)
- Correct code: `blocksize1 = 1000 // 3 = 333` (int)

The accumulation and truncation of float values leads to incorrect offset calculations, violating the fundamental expectation that reading a file in chunks preserves all data.

## Fix

```diff
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -122,7 +122,7 @@ def read_bytes(
             else:
                 # shrink blocksize to give same number of parts
                 if size % blocksize and size > blocksize:
-                    blocksize1 = size / (size // blocksize)
+                    blocksize1 = size // (size // blocksize)
                 else:
                     blocksize1 = blocksize
                 place = 0
```