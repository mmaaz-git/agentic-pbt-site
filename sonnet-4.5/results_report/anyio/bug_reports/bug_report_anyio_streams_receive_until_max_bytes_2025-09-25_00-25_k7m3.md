# Bug Report: anyio.streams.buffered.BufferedByteReceiveStream.receive_until max_bytes Violation

**Target**: `anyio.streams.buffered.BufferedByteReceiveStream.receive_until`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `receive_until()` method violates its documented contract by reading significantly more than `max_bytes` before raising `DelimiterNotFound`. The method can exceed the limit by up to one full chunk size (default 65536 bytes).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import anyio
from anyio.streams.buffered import BufferedByteReceiveStream

class LargeChunkReceiveStream:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self.count = 0

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self.count >= 10:
            raise anyio.EndOfStream
        self.count += 1
        return b"X" * self.chunk_size

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}

@given(
    chunk_size=st.integers(min_value=100, max_value=10000),
    max_bytes=st.integers(min_value=10, max_value=200)
)
def test_receive_until_respects_max_bytes(chunk_size, max_bytes):
    async def run_test():
        if chunk_size <= max_bytes:
            return

        stream = LargeChunkReceiveStream(chunk_size)
        buffered = BufferedByteReceiveStream(stream)

        try:
            await buffered.receive_until(b"NOTFOUND", max_bytes)
        except anyio.DelimiterNotFound:
            bytes_read = len(buffered.buffer)
            assert bytes_read <= max_bytes, f"Read {bytes_read} bytes, exceeded max_bytes={max_bytes}"

    anyio.run(run_test)
```

**Failing input**: `chunk_size=1000, max_bytes=100`

## Reproducing the Bug

```python
import anyio
from anyio.streams.buffered import BufferedByteReceiveStream


class LargeChunkReceiveStream:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self.count = 0

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self.count >= 10:
            raise anyio.EndOfStream
        self.count += 1
        return b"X" * self.chunk_size

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}


async def main():
    chunk_size = 1000
    max_bytes = 100

    stream = LargeChunkReceiveStream(chunk_size)
    buffered = BufferedByteReceiveStream(stream)

    try:
        await buffered.receive_until(b"NOTFOUND", max_bytes)
    except anyio.DelimiterNotFound:
        bytes_read = len(buffered.buffer)
        print(f"max_bytes parameter: {max_bytes}")
        print(f"Actual bytes read: {bytes_read}")
        print(f"Exceeded limit by: {bytes_read - max_bytes}")


anyio.run(main())
```

## Why This Is A Bug

The method's docstring states:

```
:param max_bytes: maximum number of bytes that will be read before raising
    :exc:`~anyio.DelimiterNotFound`
```

This is a clear contract: the method should not read more than `max_bytes`. However, the implementation checks the buffer size before reading from the stream, allowing it to exceed the limit.

**Current behavior** (lines 133-145 of buffered.py):
1. Check if `buffer >= max_bytes`, raise if true
2. Read from stream (potentially a large chunk)
3. Add data to buffer
4. Loop back to step 1

**Problem**: If buffer has `max_bytes - 1` bytes and the stream returns a 65KB chunk, the buffer will have `max_bytes + 65KB - 1` bytes before the check happens again.

This violates the API contract and could cause memory issues for users who set `max_bytes` to limit memory usage.

## Fix

```diff
--- a/anyio/streams/buffered.py
+++ b/anyio/streams/buffered.py
@@ -130,15 +130,20 @@ class BufferedByteReceiveStream(ByteReceiveStream):
                 del self._buffer[: index + len(delimiter) :]
                 return bytes(found)

-            # Check if the buffer is already at or over the limit
-            if len(self._buffer) >= max_bytes:
-                raise DelimiterNotFound(max_bytes)
-
             # Read more data into the buffer from the socket
+            remaining = max_bytes - len(self._buffer)
+            if remaining <= 0:
+                raise DelimiterNotFound(max_bytes)
+
             try:
-                data = await self.receive_stream.receive()
+                if isinstance(self.receive_stream, ByteReceiveStream):
+                    data = await self.receive_stream.receive(remaining)
+                else:
+                    data = await self.receive_stream.receive()
+                    if len(data) > remaining:
+                        raise DelimiterNotFound(max_bytes)
             except EndOfStream as exc:
                 raise IncompleteRead from exc

             # Move the offset forward and add the new data to the buffer
             offset = max(len(self._buffer) - delimiter_size + 1, 0)
```

This fix:
1. Calculates remaining bytes before the limit
2. For `ByteReceiveStream`, limits the receive to `remaining` bytes
3. For other streams, checks if received data would exceed the limit before adding to buffer
4. Ensures `max_bytes` is never exceeded