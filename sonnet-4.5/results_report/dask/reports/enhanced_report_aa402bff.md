# Bug Report: fsspec.utils.read_block Silent Data Loss at End of File

**Target**: `fsspec.utils.read_block`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_block` function silently loses data when reading files sequentially with a delimiter, specifically when the function seeks beyond EOF while looking for an end delimiter and returns empty bytes instead of the remaining file content.

## Property-Based Test

```python
from io import BytesIO
from hypothesis import given, strategies as st, assume, settings
from fsspec.utils import read_block


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

if __name__ == "__main__":
    # Run the test
    test_read_block_full_file_coverage_with_delimiter()
```

<details>

<summary>
**Failing input**: `chunks=[b'\x01', b'\x01\x01\x01', b'\x01\x01\x01\x01', b'\x01'], delimiter=b'\x00', block_size=10`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 35, in <module>
    test_read_block_full_file_coverage_with_delimiter()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 7, in test_read_block_full_file_coverage_with_delimiter
    st.lists(st.binary(min_size=1, max_size=100), min_size=2, max_size=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 31, in test_read_block_full_file_coverage_with_delimiter
    assert reconstructed == content, f"Concatenated blocks should equal original content. Got {len(reconstructed)} bytes, expected {len(content)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Concatenated blocks should equal original content. Got 11 bytes, expected 13
Falsifying example: test_read_block_full_file_coverage_with_delimiter(
    chunks=[b'\x01', b'\x01\x01\x01', b'\x01\x01\x01\x01', b'\x01'],
    delimiter=b'\x00',
    block_size=10,
)
```
</details>

## Reproducing the Bug

```python
from io import BytesIO
from fsspec.utils import read_block

# The bug report's minimal example
content = b'\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x01\x00'
print(f"Original content: {content} ({len(content)} bytes)")
print(f"Original content in hex: {content.hex()}")

blocks = []
offset = 0
block_size = 10
delimiter = b'\x00'

print(f"\nReading blocks with block_size={block_size}, delimiter={delimiter.hex()}")

while offset < len(content):
    f = BytesIO(content)
    print(f"\nAttempting to read block at offset {offset}...")
    block = read_block(f, offset, block_size, delimiter=delimiter)
    print(f"  Got block: {block} ({len(block)} bytes)")
    if not block:
        print("  Empty block returned, stopping")
        break
    blocks.append(block)
    offset += len(block)
    print(f"  New offset: {offset}")

reconstructed = b''.join(blocks)

print(f"\n=== RESULTS ===")
print(f"Original:      {content} ({len(content)} bytes)")
print(f"Original hex:  {content.hex()}")
print(f"Reconstructed: {reconstructed} ({len(reconstructed)} bytes)")
print(f"Reconstructed hex: {reconstructed.hex()}")

if len(content) > len(reconstructed):
    lost_data = content[len(reconstructed):]
    print(f"\nLost data:     {lost_data} ({len(lost_data)} bytes)")
    print(f"Lost data hex: {lost_data.hex()}")
    print("\n⚠️  DATA LOSS DETECTED!")
else:
    print("\n✓ No data loss")
```

<details>

<summary>
Silent data loss: 2 bytes lost from 13-byte file
</summary>
```
Original content: b'\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x01\x00' (13 bytes)
Original content in hex: 01010101010101010101000100

Reading blocks with block_size=10, delimiter=00

Attempting to read block at offset 0...
  Got block: b'\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00' (11 bytes)
  New offset: 11

Attempting to read block at offset 11...
  Got block: b'' (0 bytes)
  Empty block returned, stopping

=== RESULTS ===
Original:      b'\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x01\x00' (13 bytes)
Original hex:  01010101010101010101000100
Reconstructed: b'\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00' (11 bytes)
Reconstructed hex: 0101010101010101010100

Lost data:     b'\x01\x00' (2 bytes)
Lost data hex: 0100

⚠️  DATA LOSS DETECTED!
```
</details>

## Why This Is A Bug

This violates the fundamental expectation that reading a file in consecutive blocks should preserve all data. The function is documented as being designed to "cleanly break data by a delimiter" for parallel processing, but silent data loss makes it unsuitable for its intended purpose.

The bug occurs through this specific sequence:
1. When reading the second block at offset 11, `seek_delimiter` finds the delimiter at position 11 and advances to position 13 (past the delimiter)
2. The function calculates it needs to read 8 more bytes: `length = 10 - (13 - 11) = 8`
3. It seeks to position `13 + 8 = 21`, which is 8 bytes beyond EOF (file is only 13 bytes)
4. `seek_delimiter` returns False because it's at EOF with no delimiter found
5. The function then tries to read from position 13 to 21, but since the file ends at 13, it reads 0 bytes
6. An empty byte string is returned instead of the remaining data `b'\x01\x00'`

This is a high-severity bug because:
- **Silent data corruption**: No errors or warnings are raised, making it extremely dangerous for production systems
- **Affects core functionality**: This is a fundamental file I/O operation used by dask for parallel data processing
- **Data integrity violation**: Violates the basic principle that sequential file reading should preserve all data
- **Not an edge case**: Occurs whenever remaining data after a delimiter is less than the requested block size

## Relevant Context

The `read_block` function is imported by dask from `fsspec.utils` and is used in `dask.bytes.core.read_bytes` for distributed file reading. The function is critical for dask's parallel processing capabilities as it's used to split files into chunks that can be processed independently.

Key code locations:
- Bug location: `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/fsspec/utils.py:232-303`
- Used by dask: `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/bytes/core.py:194`

The issue stems from the logic at lines 283-285 in fsspec/utils.py where after seeking past the requested block size to find an end delimiter, the function doesn't handle the case where we've seeked beyond EOF properly.

## Proposed Fix

The bug can be fixed by ensuring that when we seek beyond EOF while looking for an end delimiter, we still return the data from the start position to EOF rather than returning empty bytes:

```diff
--- a/fsspec/utils.py
+++ b/fsspec/utils.py
@@ -275,15 +275,22 @@ def read_block(
     if delimiter:
         f.seek(offset)
         found_start_delim = seek_delimiter(f, delimiter, 2**16)
         if length is None:
             return f.read()
         start = f.tell()
         length -= start - offset

         f.seek(start + length)
         found_end_delim = seek_delimiter(f, delimiter, 2**16)
         end = f.tell()
+
+        # If we seeked beyond EOF and didn't find delimiter, read to EOF
+        if not found_end_delim and end > start:
+            f.seek(0, 2)  # Seek to end of file
+            actual_end = f.tell()
+            if actual_end > start:
+                end = actual_end

         # Adjust split location to before delimiter if seek found the
         # delimiter sequence, not start or end of file.
         if found_start_delim and split_before:
             start -= len(delimiter)
```