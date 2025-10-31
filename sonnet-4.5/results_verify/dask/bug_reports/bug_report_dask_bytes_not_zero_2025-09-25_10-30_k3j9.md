# Bug Report: dask.bytes.read_bytes not_zero Parameter Causes Empty Blocks

**Target**: `dask.bytes.core.read_bytes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `read_bytes` is called with `not_zero=True` on a file where the first block size would be 1 byte, the function produces an empty block instead of skipping just the first byte. This results in data loss and violates the expected behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import tempfile
import os
from dask.bytes.core import read_bytes

@given(
    file_size=st.integers(min_value=1, max_value=1000),
    blocksize=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=500, deadline=None)
def test_read_bytes_with_not_zero_all_blocks_nonempty(file_size, blocksize):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.txt")
        data = b'x' * file_size

        with open(filepath, 'wb') as f:
            f.write(data)

        sample, blocks = read_bytes(filepath, blocksize=blocksize, not_zero=True, sample=False)

        total_bytes = 0
        for block in blocks[0]:
            result = block.compute()
            assert len(result) > 0, f"Empty block found! file_size={file_size}, blocksize={blocksize}"
            total_bytes += len(result)

        expected_total = file_size - 1
        assert total_bytes == expected_total
```

**Failing input**: `file_size=1, blocksize=1`

## Reproducing the Bug

```python
import tempfile
import os
from dask.bytes.core import read_bytes

with tempfile.TemporaryDirectory() as tmpdir:
    filepath = os.path.join(tmpdir, "test.txt")
    with open(filepath, 'wb') as f:
        f.write(b'a')

    sample, blocks = read_bytes(filepath, blocksize=1, not_zero=True, sample=False)

    result = blocks[0][0].compute()
    print(f"Expected: non-empty block (the file has 1 byte)")
    print(f"Got: {result!r} (length={len(result)})")
```

## Why This Is A Bug

The `not_zero` parameter is documented to "Force seek of start-of-file delimiter, discarding header." This means it should skip the first byte and return the rest of the file. However, when the first block is only 1 byte long, the current implementation creates an empty block by:

1. Setting `off[0] = 1` (start at byte 1 instead of 0)
2. Setting `length[0] -= 1` (reducing the length from 1 to 0)

This results in reading 0 bytes instead of properly handling the edge case.

The invariant that "all blocks should have length > 0" is violated, which can cause downstream code to fail or produce incorrect results.

## Fix

```diff
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -136,9 +136,11 @@ def read_bytes(
                     length.append(off[-1] - off[-2])
                 length.append(size - off[-1])

-                if not_zero:
+                if not_zero and length[0] > 1:
                     off[0] = 1
                     length[0] -= 1
+                elif not_zero:
+                    off = off[1:]
+                    length = length[1:]
                 offsets.append(off)
                 lengths.append(length)
```

This fix checks if the first block is large enough (> 1 byte) before applying the not_zero adjustment. If the first block is only 1 byte, it removes that block entirely rather than creating an empty block.