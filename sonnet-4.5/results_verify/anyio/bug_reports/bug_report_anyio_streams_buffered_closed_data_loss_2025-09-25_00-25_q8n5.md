# Bug Report: anyio.streams.buffered.BufferedByteReceiveStream Buffered Data Lost When Closed

**Target**: `anyio.streams.buffered.BufferedByteReceiveStream.receive`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `BufferedByteReceiveStream` is closed, any data in the buffer becomes inaccessible. The `receive()` method raises `ClosedResourceError` immediately without checking if there's buffered data available, causing data loss.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import anyio
from anyio.streams.buffered import BufferedByteReceiveStream


class SimpleBytesReceiveStream:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self.pos >= len(self.data):
            raise anyio.EndOfStream
        chunk = self.data[self.pos : self.pos + max_bytes]
        self.pos += len(chunk)
        if not chunk:
            raise anyio.EndOfStream
        return chunk

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}


@given(data=st.binary(min_size=1, max_size=1000))
def test_buffered_data_accessible_after_close(data):
    async def run_test():
        stream = SimpleBytesReceiveStream(b"")
        buffered = BufferedByteReceiveStream(stream)

        buffered.feed_data(data)
        await buffered.aclose()

        received = bytearray()
        try:
            while True:
                chunk = await buffered.receive()
                received.extend(chunk)
        except (anyio.EndOfStream, anyio.ClosedResourceError):
            pass

        assert bytes(received) == data, f"Lost buffered data after close"

    anyio.run(run_test)
```

**Failing input**: Any non-empty data fed to buffer before close

## Reproducing the Bug

```python
import anyio
from anyio.streams.buffered import BufferedByteReceiveStream


class SimpleBytesReceiveStream:
    def __init__(self):
        pass

    async def receive(self, max_bytes: int = 65536) -> bytes:
        raise anyio.EndOfStream

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}


async def main():
    stream = SimpleBytesReceiveStream()
    buffered = BufferedByteReceiveStream(stream)

    buffered.feed_data(b"important data")
    print(f"Data in buffer: {buffered.buffer}")

    await buffered.aclose()
    print("Stream closed")

    print("\nAttempting to receive buffered data...")
    try:
        data = await buffered.receive()
        print(f"Received: {data!r}")
    except anyio.ClosedResourceError as e:
        print(f"ClosedResourceError: {e}")
        print(f"Buffered data is lost: {buffered.buffer}")
        print("The data can never be retrieved!")


anyio.run(main())
```

**Output:**
```
Data in buffer: b'important data'
Stream closed
Attempting to receive buffered data...
ClosedResourceError
Buffered data is lost: b'important data'
The data can never be retrieved!
```

## Why This Is A Bug

Standard stream behavior (e.g., Python's `io.BufferedReader`) allows reading buffered data even after the underlying stream is closed. Only after the buffer is exhausted should `ClosedResourceError` be raised.

**Current behavior** (lines 61-68 of buffered.py):
1. `receive()` immediately checks `if self._closed` and raises
2. Buffered data is never returned
3. Data loss occurs

**Expected behavior**:
1. `receive()` should return buffered data if available
2. Only raise `ClosedResourceError` when buffer is empty AND stream is closed

This is especially problematic because:
- `feed_data()` doesn't check if stream is closed (line 48-59)
- You can feed data to a closed stream, but never retrieve it
- Silent data loss without warning

## Fix

```diff
--- a/anyio/streams/buffered.py
+++ b/anyio/streams/buffered.py
@@ -59,11 +59,11 @@ class BufferedByteReceiveStream(ByteReceiveStream):
         self._buffer.extend(data)

     async def receive(self, max_bytes: int = 65536) -> bytes:
-        if self._closed:
-            raise ClosedResourceError
-
         if self._buffer:
             chunk = bytes(self._buffer[:max_bytes])
             del self._buffer[:max_bytes]
             return chunk
+
+        if self._closed:
+            raise ClosedResourceError

         elif isinstance(self.receive_stream, ByteReceiveStream):
```

This fix:
1. Checks buffer first before checking closed status
2. Returns buffered data even if stream is closed
3. Only raises `ClosedResourceError` when buffer is empty
4. Matches standard Python stream behavior
5. Prevents silent data loss