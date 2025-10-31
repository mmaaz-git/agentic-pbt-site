# Bug Report: BufferedByteReceiveStream.receive_until max_bytes Violation

**Target**: `anyio.streams.buffered.BufferedByteReceiveStream.receive_until`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `receive_until` method can return results that exceed `max_bytes` when the delimiter is found beyond the specified limit, violating its documented contract that it should raise `DelimiterNotFound` if the delimiter is not found within `max_bytes`.

## Property-Based Test

```python
import anyio
from anyio.streams.buffered import BufferedByteReceiveStream
from hypothesis import given, strategies as st, settings, assume
import pytest


class MockByteReceiveStream:
    def __init__(self, data):
        self.data = data
        self.position = 0

    async def receive(self, max_bytes=65536):
        if self.position >= len(self.data):
            raise anyio.EndOfStream
        chunk = self.data[self.position:self.position + max_bytes]
        self.position += len(chunk)
        return chunk

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


@given(
    st.binary(min_size=0, max_size=100),
    st.binary(min_size=1, max_size=10),
    st.binary(min_size=0, max_size=100),
    st.integers(min_value=1, max_value=1000)
)
@settings(max_examples=500)
def test_receive_until_respects_max_bytes(before, delimiter, after, max_bytes):
    async def run_test():
        assume(delimiter not in before)
        full_data = before + delimiter + after
        mock = MockByteReceiveStream(full_data)
        buffered = BufferedByteReceiveStream(mock)

        if len(before) < max_bytes:
            result = await buffered.receive_until(delimiter, max_bytes)
            assert result == before
            assert len(result) < max_bytes
        else:
            with pytest.raises(anyio.DelimiterNotFound):
                await buffered.receive_until(delimiter, max_bytes)

    anyio.run(run_test)
```

**Failing input**: `before=b'\x00', delimiter=b'\x01', after=b'', max_bytes=1`

## Reproducing the Bug

```python
import anyio
from anyio.streams.buffered import BufferedByteReceiveStream


class MockByteReceiveStream:
    def __init__(self, data):
        self.data = data
        self.position = 0

    async def receive(self, max_bytes=65536):
        if self.position >= len(self.data):
            raise anyio.EndOfStream
        chunk = self.data[self.position:self.position + max_bytes]
        self.position += len(chunk)
        return chunk

    async def aclose(self):
        pass

    @property
    def extra_attributes(self):
        return {}


async def main():
    data = b'X|'
    delimiter = b'|'
    max_bytes = 1

    mock = MockByteReceiveStream(data)
    buffered = BufferedByteReceiveStream(mock)

    result = await buffered.receive_until(delimiter, max_bytes)
    print(f"Result: {result!r} (length {len(result)})")


anyio.run(main)
```

Output:
```
Result: b'X' (length 1)
```

Expected: Should raise `anyio.DelimiterNotFound` because the delimiter is at position 1, requiring scanning 2 bytes, which exceeds `max_bytes=1`.

## Why This Is A Bug

The documentation for `receive_until` states:

> :param max_bytes: maximum number of bytes that will be read before raising :exc:`~anyio.DelimiterNotFound`
> :raises ~anyio.DelimiterNotFound: if the delimiter is not found within the bytes read up to the maximum allowed

The issue is in the timing of the `max_bytes` check on line 134 of `buffered.py`:

```python
async def receive_until(self, delimiter: bytes, max_bytes: int) -> bytes:
    delimiter_size = len(delimiter)
    offset = 0
    while True:
        index = self._buffer.find(delimiter, offset)
        if index >= 0:
            found = self._buffer[:index]
            del self._buffer[: index + len(delimiter) :]
            return bytes(found)

        if len(self._buffer) >= max_bytes:  # Line 134
            raise DelimiterNotFound(max_bytes)

        data = await self.receive_stream.receive()
        offset = max(len(self._buffer) - delimiter_size + 1, 0)
        self._buffer.extend(data)
```

The check happens **before** reading more data (line 139) and extending the buffer (line 145). This means:

1. If buffer has 0 bytes and `max_bytes=1`, the check passes
2. We read data, buffer becomes 2+ bytes
3. We find the delimiter at position > max_bytes
4. We return the result, violating the `max_bytes` limit

The correct behavior is to ensure that when a delimiter is found at position `index`, we have `index < max_bytes`. Otherwise, we've exceeded the maximum allowed bytes to scan.

## Fix

```diff
--- a/anyio/streams/buffered.py
+++ b/anyio/streams/buffered.py
@@ -125,10 +125,6 @@ class BufferedByteReceiveStream(ByteReceiveStream):
         while True:
             # Check if the delimiter can be found in the current buffer
             index = self._buffer.find(delimiter, offset)
             if index >= 0:
-                found = self._buffer[:index]
-                del self._buffer[: index + len(delimiter) :]
-                return bytes(found)
-
-            # Check if the buffer is already at or over the limit
-            if len(self._buffer) >= max_bytes:
+                if index >= max_bytes:
+                    raise DelimiterNotFound(max_bytes)
+                found = self._buffer[:index]
+                del self._buffer[: index + len(delimiter) :]
+                return bytes(found)
+
+            # Check if the buffer is already at or over the limit
+            if len(self._buffer) >= max_bytes:
                 raise DelimiterNotFound(max_bytes)
```

This fix ensures that even when the delimiter is found, we check if it's within the `max_bytes` limit before returning the result.