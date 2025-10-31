# Bug Report: anyio BufferedByteReceiveStream.receive_until() Delimiter Boundary Bug

**Target**: `anyio.streams.buffered.BufferedByteReceiveStream.receive_until()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `receive_until()` method incorrectly raises `DelimiterNotFound` when a delimiter is split across the `max_bytes` boundary, even though the delimiter exists in the stream.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from anyio.streams.buffered import BufferedByteReceiveStream
from anyio import DelimiterNotFound
import asyncio


@given(
    delimiter=st.binary(min_size=2, max_size=10),
    max_bytes=st.integers(min_value=10, max_value=1000),
    prefix_size=st.integers(min_value=0, max_value=50)
)
def test_delimiter_at_boundary_should_be_found(delimiter, max_bytes, prefix_size):
    assume(prefix_size < max_bytes)
    position = max_bytes - len(delimiter) + 1
    assume(position > 0)

    async def run_test():
        chunk1 = b'A' * position + delimiter[:len(delimiter)//2]
        chunk2 = delimiter[len(delimiter)//2:] + b'rest'

        stream = BufferedByteReceiveStream(MockStream([chunk1, chunk2]))

        result = await stream.receive_until(delimiter, max_bytes + 100)
        assert delimiter not in result
        assert len(result) == position

    asyncio.run(run_test())
```

**Failing input**: `delimiter=b'XX'`, `max_bytes=50`, `data=[b'A'*49 + b'X', b'X' + b'...']`

## Reproducing the Bug

```python
import asyncio
from anyio.streams.buffered import BufferedByteReceiveStream
from anyio import EndOfStream, DelimiterNotFound
from dataclasses import dataclass


@dataclass
class MockByteReceiveStream:
    chunks: list

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if not self.chunks:
            raise EndOfStream
        return self.chunks.pop(0)

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}


async def test_bug():
    delimiter = b'XX'
    max_bytes = 50

    chunk1 = b'A' * 49 + b'X'
    chunk2 = b'X' + b'rest of data'

    stream = BufferedByteReceiveStream(MockByteReceiveStream([chunk1, chunk2]))

    try:
        result = await stream.receive_until(delimiter, max_bytes)
        print(f"Found: {len(result)} bytes")
    except DelimiterNotFound:
        print("BUG: DelimiterNotFound raised, but delimiter 'XX' exists at positions 49-50!")


asyncio.run(test_bug())
```

## Why This Is A Bug

The docstring states that `max_bytes` is the "maximum number of bytes that will be read before raising DelimiterNotFound." The delimiter `b'XX'` is located at positions 49-50 in the stream (within reasonable search bounds), but the function raises `DelimiterNotFound` because:

1. After reading the first chunk (50 bytes), the buffer contains `b'A'*49 + b'X'`
2. The delimiter `b'XX'` is not found in the buffer yet
3. The check `if len(self._buffer) >= max_bytes` at line 134 evaluates to `True` (50 >= 50)
4. `DelimiterNotFound` is raised *before* reading the second chunk that would complete the delimiter

This violates user expectations: if a delimiter spans the `max_bytes` boundary, it should still be found if the position where it starts is within reasonable bounds.

## Fix

The fix should ensure that we read enough data to check if a delimiter completes at the boundary. Change line 134 from:

```python
if len(self._buffer) >= max_bytes:
    raise DelimiterNotFound(max_bytes)
```

to:

```python
if len(self._buffer) >= max_bytes + delimiter_size - 1:
    raise DelimiterNotFound(max_bytes)
```

This allows reading up to `delimiter_size - 1` additional bytes beyond `max_bytes` to check for delimiters that start before the boundary but complete after it.

```diff
--- a/anyio/streams/buffered.py
+++ b/anyio/streams/buffered.py
@@ -131,7 +131,7 @@ class BufferedByteReceiveStream(ByteReceiveStream):
                 return bytes(found)

             # Check if the buffer is already at or over the limit
-            if len(self._buffer) >= max_bytes:
+            if len(self._buffer) >= max_bytes + delimiter_size - 1:
                 raise DelimiterNotFound(max_bytes)

             # Read more data into the buffer from the socket
```

Alternatively, if `max_bytes` should represent the maximum *result* size (not buffer size), the check should be on the delimiter position:

```diff
--- a/anyio/streams/buffered.py
+++ b/anyio/streams/buffered.py
@@ -127,6 +127,10 @@ class BufferedByteReceiveStream(ByteReceiveStream):
             # Check if the delimiter can be found in the current buffer
             index = self._buffer.find(delimiter, offset)
             if index >= 0:
+                # Check if result would exceed max_bytes
+                if index >= max_bytes:
+                    raise DelimiterNotFound(max_bytes)
+
                 found = self._buffer[:index]
                 del self._buffer[: index + len(delimiter) :]
                 return bytes(found)
```