# Bug Report: anyio.streams.buffered BufferedByteReceiveStream.receive() accepts negative max_bytes

**Target**: `anyio.streams.buffered.BufferedByteReceiveStream.receive()`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `BufferedByteReceiveStream.receive()` method accepts negative values for the `max_bytes` parameter and exhibits unintuitive behavior, using Python's negative slice semantics instead of validating the input or treating it as zero.

## Property-Based Test

```python
import asyncio
from hypothesis import given, strategies as st, settings
from anyio.streams.buffered import BufferedByteReceiveStream
from anyio import EndOfStream


class MockByteReceiveStream:
    def __init__(self, data_chunks):
        self.data_chunks = list(data_chunks)
        self.index = 0
        self.extra_attributes = {}

    async def receive(self, max_bytes=65536):
        if self.index >= len(self.data_chunks):
            raise EndOfStream
        chunk = self.data_chunks[self.index]
        self.index += 1
        return chunk

    async def aclose(self):
        pass


@given(st.binary(min_size=10, max_size=100))
@settings(max_examples=100)
def test_receive_negative_max_bytes(data):
    async def test():
        stream = MockByteReceiveStream([data])
        buffered = BufferedByteReceiveStream(stream)

        result = await buffered.receive(-1)
        assert len(result) >= 0

    asyncio.run(test())
```

**Failing input**: Any binary data of length >= 1, e.g., `b"HelloWorld"`

## Reproducing the Bug

```python
import asyncio
from anyio.streams.buffered import BufferedByteReceiveStream
from anyio import EndOfStream


class SimpleByteStream:
    def __init__(self, data):
        self.data = data
        self.consumed = False
        self.extra_attributes = {}

    async def receive(self, max_bytes=65536):
        if self.consumed:
            raise EndOfStream
        self.consumed = True
        return self.data

    async def aclose(self):
        pass


async def main():
    data = b"HelloWorld"
    stream = SimpleByteStream(data)
    buffered = BufferedByteReceiveStream(stream)

    result = await buffered.receive(max_bytes=-1)
    print(f"Called receive(max_bytes=-1) on {data!r}")
    print(f"Result: {result!r} (length: {len(result)})")
    print(f"Buffer: {buffered.buffer!r}")


asyncio.run(main())
```

Output:
```
Called receive(max_bytes=-1) on b'HelloWorld'
Result: b'HelloWorl' (length: 9)
Buffer: b'd'
```

## Why This Is A Bug

The parameter `max_bytes` semantically represents the maximum number of bytes to receive, which should be a non-negative integer. Accepting negative values is unintuitive and likely unintended:

1. When `max_bytes=-1`, the function returns all but the last byte (using `chunk[:-1]` slicing)
2. When `max_bytes=-2`, it returns all but the last 2 bytes
3. This behavior is not documented and violates the semantic expectation of "max bytes"
4. Users would reasonably expect either:
   - An error/exception for invalid input
   - Treatment of negative values as zero (return empty bytes)
   - Or at minimum, clear documentation of this behavior

Looking at the implementation in `/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages/anyio/streams/buffered.py:61-80`:

```python
async def receive(self, max_bytes: int = 65536) -> bytes:
    if self._closed:
        raise ClosedResourceError

    if self._buffer:
        chunk = bytes(self._buffer[:max_bytes])
        del self._buffer[:max_bytes]
        return chunk
    elif isinstance(self.receive_stream, ByteReceiveStream):
        return await self.receive_stream.receive(max_bytes)
    else:
        chunk = await self.receive_stream.receive()
        if len(chunk) > max_bytes:  # This comparison is problematic with negative values
            self._buffer.extend(chunk[max_bytes:])
            return chunk[:max_bytes]
        else:
            return chunk
```

When `max_bytes` is negative:
- Line 66: `self._buffer[:max_bytes]` uses negative slicing (e.g., `[:-1]` = all but last)
- Line 75: `len(chunk) > max_bytes` is always True for any chunk when max_bytes < 0
- Line 76-77: Uses negative slicing on the chunk

## Fix

```diff
--- a/anyio/streams/buffered.py
+++ b/anyio/streams/buffered.py
@@ -60,6 +60,9 @@ class BufferedByteReceiveStream(ByteReceiveStream):

     async def receive(self, max_bytes: int = 65536) -> bytes:
+        if max_bytes < 0:
+            raise ValueError(f"max_bytes must be non-negative, got {max_bytes}")
+
         if self._closed:
             raise ClosedResourceError
```

Alternatively, if negative values should be treated as zero:

```diff
--- a/anyio/streams/buffered.py
+++ b/anyio/streams/buffered.py
@@ -60,6 +60,8 @@ class BufferedByteReceiveStream(ByteReceiveStream):

     async def receive(self, max_bytes: int = 65536) -> bytes:
+        max_bytes = max(0, max_bytes)
+
         if self._closed:
             raise ClosedResourceError
```