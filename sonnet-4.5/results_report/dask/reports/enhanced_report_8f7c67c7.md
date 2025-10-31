# Bug Report: dask.bytes.read_bytes Returns Empty Blocks When not_zero=True and First Block Size is 1 Byte

**Target**: `dask.bytes.core.read_bytes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_bytes` function returns empty blocks (0 bytes) when called with `not_zero=True` on files where the first block would be exactly 1 byte, violating the expectation that all blocks should contain data.

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

if __name__ == "__main__":
    test_read_bytes_with_not_zero_all_blocks_nonempty()
```

<details>

<summary>
**Failing input**: `file_size=1, blocksize=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 31, in <module>
    test_read_bytes_with_not_zero_all_blocks_nonempty()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 7, in test_read_bytes_with_not_zero_all_blocks_nonempty
    file_size=st.integers(min_value=1, max_value=1000),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 24, in test_read_bytes_with_not_zero_all_blocks_nonempty
    assert len(result) > 0, f"Empty block found! file_size={file_size}, blocksize={blocksize}"
           ^^^^^^^^^^^^^^^
AssertionError: Empty block found! file_size=1, blocksize=1
Falsifying example: test_read_bytes_with_not_zero_all_blocks_nonempty(
    # The test sometimes passed when commented parts were varied together.
    file_size=1,  # or any other generated value
    blocksize=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import tempfile
import os
from dask.bytes.core import read_bytes

# Test case: file with 1 byte, blocksize=1, not_zero=True
with tempfile.TemporaryDirectory() as tmpdir:
    filepath = os.path.join(tmpdir, "test.txt")
    with open(filepath, 'wb') as f:
        f.write(b'a')

    print("Testing read_bytes with not_zero=True on 1-byte file with blocksize=1")
    print("=" * 60)

    sample, blocks = read_bytes(filepath, blocksize=1, not_zero=True, sample=False)

    print(f"Number of blocks: {len(blocks[0])}")

    if len(blocks[0]) > 0:
        result = blocks[0][0].compute()
        print(f"Block 0 content: {result!r}")
        print(f"Block 0 length: {len(result)}")

        if len(result) == 0:
            print("\nBUG CONFIRMED: Empty block returned!")
            print("Expected: Either no blocks (since we skip the only byte) or error")
            print("Got: Empty block with length 0")
    else:
        print("No blocks returned (which might be correct since we skip the only byte)")

print("\n" + "=" * 60)
print("Additional test cases:")
print("=" * 60)

# Test with 2-byte file
with tempfile.TemporaryDirectory() as tmpdir:
    filepath = os.path.join(tmpdir, "test2.txt")
    with open(filepath, 'wb') as f:
        f.write(b'ab')

    print("\n2-byte file with blocksize=1, not_zero=True:")
    sample, blocks = read_bytes(filepath, blocksize=1, not_zero=True, sample=False)

    for i, block in enumerate(blocks[0]):
        result = block.compute()
        print(f"  Block {i}: {result!r} (length={len(result)})")

# Test with 3-byte file
with tempfile.TemporaryDirectory() as tmpdir:
    filepath = os.path.join(tmpdir, "test3.txt")
    with open(filepath, 'wb') as f:
        f.write(b'abc')

    print("\n3-byte file with blocksize=1, not_zero=True:")
    sample, blocks = read_bytes(filepath, blocksize=1, not_zero=True, sample=False)

    for i, block in enumerate(blocks[0]):
        result = block.compute()
        print(f"  Block {i}: {result!r} (length={len(result)})")
```

<details>

<summary>
Empty blocks returned for files with 1-byte first blocks when not_zero=True
</summary>
```
Testing read_bytes with not_zero=True on 1-byte file with blocksize=1
============================================================
Number of blocks: 1
Block 0 content: b''
Block 0 length: 0

BUG CONFIRMED: Empty block returned!
Expected: Either no blocks (since we skip the only byte) or error
Got: Empty block with length 0

============================================================
Additional test cases:
============================================================

2-byte file with blocksize=1, not_zero=True:
  Block 0: b'' (length=0)
  Block 1: b'b' (length=1)

3-byte file with blocksize=1, not_zero=True:
  Block 0: b'' (length=0)
  Block 1: b'b' (length=1)
  Block 2: b'c' (length=1)
```
</details>

## Why This Is A Bug

The `not_zero` parameter is documented to "Force seek of start-of-file delimiter, discarding header." This should skip the first byte and return the remaining file content in blocks. However, the current implementation has a logic error that produces empty blocks when the first block would be exactly 1 byte.

Looking at the code in `dask/bytes/core.py` lines 139-141:

```python
if not_zero:
    off[0] = 1      # Start reading at byte 1 instead of 0
    length[0] -= 1  # Reduce first block length by 1
```

When the first block has length=1:
- `off[0] = 1` sets the start position to byte 1 (skipping byte 0)
- `length[0] = 1 - 1 = 0` sets the read length to 0 bytes

This creates a block that reads 0 bytes starting from position 1, resulting in an empty block `b''`.

This violates the reasonable expectation that:
1. All blocks returned should contain data (non-empty)
2. The function should handle edge cases gracefully
3. Empty blocks serve no purpose and can cause downstream processing failures

## Relevant Context

The bug manifests in any scenario where:
- `not_zero=True` is specified
- The first block would be exactly 1 byte (either `blocksize=1` or the last block of a file happens to be 1 byte)

This issue was discovered through property-based testing with Hypothesis, which systematically explores edge cases. While `blocksize=1` is uncommon in production, the bug could also occur with larger blocksizes when file sizes create 1-byte first blocks through the block distribution algorithm.

The underlying `fsspec.utils.read_block` function correctly returns empty bytes when asked to read 0 bytes, so the issue is purely in the dask layer's handling of the `not_zero` parameter.

Documentation link: The function is part of dask.bytes.core module, used for chunked reading of files in distributed computing scenarios.

## Proposed Fix

```diff
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -136,9 +136,14 @@ def read_bytes(
                 length.append(off[-1] - off[-2])
             length.append(size - off[-1])

-            if not_zero:
+            if not_zero and length[0] > 1:
+                # Normal case: adjust first block to skip first byte
                 off[0] = 1
                 length[0] -= 1
+            elif not_zero and length[0] == 1:
+                # Edge case: first block is only 1 byte, remove it entirely
+                off = off[1:] if len(off) > 1 else []
+                length = length[1:] if len(length) > 1 else []
             offsets.append(off)
             lengths.append(length)
```