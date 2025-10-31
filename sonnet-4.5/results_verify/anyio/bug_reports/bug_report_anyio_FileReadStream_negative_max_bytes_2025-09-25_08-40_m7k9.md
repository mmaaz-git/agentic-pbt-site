# Bug Report: anyio.streams.file.FileReadStream Negative max_bytes

**Target**: `anyio.streams.file.FileReadStream.receive`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FileReadStream.receive()` accepts negative values for `max_bytes`, which causes it to read the entire file instead of a bounded maximum, potentially causing memory issues with large files.

## Property-Based Test

```python
import tempfile
import pytest
import anyio
from anyio.streams.file import FileReadStream
from hypothesis import given, strategies as st


@pytest.mark.anyio
async def test_receive_negative_max_bytes():
    """
    Property: max_bytes should constrain the amount of data read.
    Negative values should either be rejected or treated as a sensible default,
    not read the entire file.
    """
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        test_data = b'x' * 1000000  # 1MB
        f.write(test_data)
        f.flush()
        path = f.name

    try:
        stream = await FileReadStream.from_path(path)
        data = await stream.receive(max_bytes=-1)
        assert len(data) == 1000000  # Entire file was read!
        await stream.aclose()
    finally:
        import os
        os.unlink(path)
```

**Failing input**: `max_bytes=-1` or any negative value

## Reproducing the Bug

```python
import tempfile
import anyio
from anyio.streams.file import FileReadStream


async def reproduce_bug():
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        large_data = b'x' * 10_000_000  # 10MB
        f.write(large_data)
        f.flush()
        path = f.name

    try:
        stream = await FileReadStream.from_path(path)

        data = await stream.receive(max_bytes=-1)

        print(f"Requested: max_bytes=-1")
        print(f"Received: {len(data)} bytes (entire file!)")
        print(f"Expected: Should error or read a bounded amount")

        await stream.aclose()
    finally:
        import os
        os.unlink(path)


anyio.run(reproduce_bug)
```

Output:
```
Requested: max_bytes=-1
Received: 10000000 bytes (entire file!)
Expected: Should error or read a bounded amount
```

## Why This Is A Bug

The parameter name `max_bytes` implies it sets a maximum on the number of bytes to receive. However, the implementation directly passes this value to `file.read()`:

```python
# file.py lines 75-77
async def receive(self, max_bytes: int = 65536) -> bytes:
    try:
        data = await to_thread.run_sync(self._file.read, max_bytes)
```

In Python, `file.read(-1)` is a special case that reads the **entire file**, regardless of size. This means:

1. **Violates parameter semantics**: `max_bytes=-1` reads unlimited bytes, not a maximum
2. **Memory safety**: Can cause OOM errors with large files
3. **Unexpected behavior**: Users might pass -1 accidentally (e.g., from calculation errors)
4. **No validation**: The method doesn't validate that `max_bytes > 0`

**Related issue**: `max_bytes=0` raises `EndOfStream` even when the file has data, which is confusing (though less severe).

## Fix

Add validation to ensure `max_bytes` is positive:

```diff
--- a/anyio/streams/file.py
+++ b/anyio/streams/file.py
@@ -74,6 +74,9 @@ class FileReadStream(_BaseFileStream, ByteReceiveStream):

     async def receive(self, max_bytes: int = 65536) -> bytes:
+        if max_bytes <= 0:
+            raise ValueError("max_bytes must be positive")
+
         try:
             data = await to_thread.run_sync(self._file.read, max_bytes)
         except ValueError:
```

Alternatively, treat non-positive values as the default:

```diff
--- a/anyio/streams/file.py
+++ b/anyio/streams/file.py
@@ -74,6 +74,9 @@ class FileReadStream(_BaseFileStream, ByteReceiveStream):

     async def receive(self, max_bytes: int = 65536) -> bytes:
+        if max_bytes <= 0:
+            max_bytes = 65536  # Use default
+
         try:
             data = await to_thread.run_sync(self._file.read, max_bytes)
         except ValueError: