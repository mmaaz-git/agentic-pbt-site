# Bug Report: dask.bytes.read_block Data Loss at End of File

**Target**: `dask.bytes.core.read_block`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_block` function silently loses data at the end of a file when reading in blocks with a delimiter. When the remaining data after a delimiter is less than the requested block size, the function returns an empty byte string instead of the remaining data, causing data loss.

## Property-Based Test

```python
from io import BytesIO
from hypothesis import given, strategies as st, assume, settings
from dask.bytes.core import read_block


@given(
    st.lists(st.binary(min_size=1, max_size=100), min_size=2, max_size=50),
    st.binary(min_size=1, max_size=5),
    st.integers(min_value=10, max_value=200),
)
@settings(max_examples=300)
def test_read_block_full_file_coverage_with_delimiter(chunks, delimiter, block_size):
    assume(delimiter not in b''.join(chunk for chunk in chunks))

    content = delimiter.join(chunks) + delimiter
    assume(len(content) > block_size)

    blocks = []
    offset = 0
    f = BytesIO(content)

    while offset < len(content):
        f.seek(0)
        block = read_block(f, offset, block_size, delimiter=delimiter)
        if not block:
            break
        blocks.append(block)
        offset += len(block)

    reconstructed = b''.join(blocks)
    assert reconstructed == content, f"Concatenated blocks should equal original content. Got {len(reconstructed)} bytes, expected {len(content)}"
```

**Failing input**:
```python
chunks=[b'\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01', b'\x01']
delimiter=b'\x00'
block_size=10
```

## Reproducing the Bug

```python
from io import BytesIO
from dask.bytes.core import read_block

content = b'\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x01\x00'

blocks = []
offset = 0
block_size = 10
delimiter = b'\x00'

while offset < len(content):
    f = BytesIO(content)
    block = read_block(f, offset, block_size, delimiter=delimiter)
    if not block:
        break
    blocks.append(block)
    offset += len(block)

reconstructed = b''.join(blocks)

print(f"Original:      {content} ({len(content)} bytes)")
print(f"Reconstructed: {reconstructed} ({len(reconstructed)} bytes)")
print(f"Lost data:     {content[len(reconstructed):]} ({len(content) - len(reconstructed)} bytes)")
```

Output:
```
Original:      b'\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x01\x00' (13 bytes)
Reconstructed: b'\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00' (11 bytes)
Lost data:     b'\x01\x00' (2 bytes)
```

## Why This Is A Bug

The function violates the fundamental property that reading a file in consecutive blocks and concatenating them should produce the original file content. The docstring states this is designed to "cleanly break data by a delimiter" for parallel processing, but data loss makes it unsuitable for this purpose.

The bug occurs when:
1. Reading at an offset where the next delimiter is found shortly after
2. After seeking past that delimiter, the file position is near EOF
3. The remaining content is less than the requested block size
4. The function seeks beyond EOF to find the end delimiter
5. Returns an empty byte string instead of the remaining data

This is a high-severity bug because it causes **silent data loss** - the function returns successfully but loses data without any error or warning.

## Fix

The issue is in the delimiter handling logic when calculating the end position. When `start + length` exceeds the file size and no end delimiter is found, the function should still return the remaining data from `start` to EOF.

The fix requires checking if we're near EOF and adjusting the logic accordingly:

```diff
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -XX,XX +XX,XX @@ def read_block(
     if delimiter:
         f.seek(offset)
         found_start_delim = seek_delimiter(f, delimiter, 2**16)
         if length is None:
             return f.read()
         start = f.tell()
         length -= start - offset
+
+        if length <= 0:
+            return f.read()

         f.seek(start + length)
         found_end_delim = seek_delimiter(f, delimiter, 2**16)
```

The key insight is that when `start - offset >= original_length`, we've already consumed the block size budget finding the start delimiter, so we should just read to EOF rather than trying to find an end delimiter.