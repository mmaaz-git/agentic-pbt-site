# Bug Report: FileReadStream.receive(0) raises EndOfStream

**Target**: `anyio.streams.file.FileReadStream.receive`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Calling `FileReadStream.receive(0)` on a stream with remaining data incorrectly raises `EndOfStream` instead of returning an empty bytes object.

## Property-Based Test

```python
import pytest
from hypothesis import given, strategies as st, settings
import tempfile
import anyio
from anyio.streams.file import FileReadStream
from anyio import EndOfStream


@given(content=st.binary(min_size=1, max_size=1000))
@settings(max_examples=100)
def test_file_read_stream_receive_zero_bytes(content):
    """
    Property: Calling receive(0) on a FileReadStream with remaining data should
    return an empty bytes object, not raise EndOfStream.
    """
    async def run_test():
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            stream = await FileReadStream.from_path(temp_path)
            result = await stream.receive(0)
            assert result == b''
            await stream.aclose()
        finally:
            import os
            os.unlink(temp_path)

    try:
        anyio.run(run_test)
    except EndOfStream:
        pytest.fail("receive(0) raised EndOfStream instead of returning b''")
```

**Failing input**: `content=b'\x00'` (or any non-empty bytes)

## Reproducing the Bug

```python
import tempfile
import anyio
from anyio.streams.file import FileReadStream


async def demonstrate_bug():
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(b'Hello, World!')
        temp_path = f.name

    try:
        stream = await FileReadStream.from_path(temp_path)
        result = await stream.receive(0)
        print(f"Result: {result!r}")
    except anyio.EndOfStream:
        print("Bug: EndOfStream raised when receiving 0 bytes")
    finally:
        import os
        os.unlink(temp_path)


anyio.run(demonstrate_bug)
```

## Why This Is A Bug

The `receive(0)` call is a valid request where the user explicitly asks for 0 bytes. The current implementation:

1. Calls `file.read(0)` which always returns `b''`
2. Treats empty bytes as end-of-stream (line 83-86 in file.py)
3. Raises `EndOfStream` even though the stream still has data

This is incorrect because:
- The stream is NOT at EOF - there is still data to read
- Requesting 0 bytes should return 0 bytes without indicating EOF
- This prevents users from using `receive(0)` as a no-op or state check

## Fix

```diff
--- a/anyio/streams/file.py
+++ b/anyio/streams/file.py
@@ -74,6 +74,10 @@ class FileReadStream(_BaseFileStream, ByteReceiveStream):

     async def receive(self, max_bytes: int = 65536) -> bytes:
+        if max_bytes == 0:
+            return b''
+
         try:
             data = await to_thread.run_sync(self._file.read, max_bytes)
         except ValueError:
```