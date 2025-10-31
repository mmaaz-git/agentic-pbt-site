# Bug Report: anyio.streams BufferedByteReceiveStream.receive_until max_bytes Violation

**Target**: `anyio.streams.buffered.BufferedByteReceiveStream.receive_until`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`BufferedByteReceiveStream.receive_until()` succeeds even when the delimiter is found beyond the `max_bytes` limit, violating its documented contract that it should raise `DelimiterNotFound` when the delimiter is not found within `max_bytes`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import anyio
from anyio.streams.buffered import BufferedByteReceiveStream
import pytest


class MockByteReceiveStream:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self.pos >= len(self.data):
            raise anyio.EndOfStream
        chunk = self.data[self.pos:self.pos + max_bytes]
        self.pos += len(chunk)
        return chunk

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}


@given(
    delimiter=st.binary(min_size=2, max_size=10),
    max_bytes=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=200)
def test_receive_until_enforces_max_bytes_limit(delimiter, max_bytes):
    assume(len(delimiter) > max_bytes)

    data = delimiter

    async def test():
        mock_stream = MockByteReceiveStream(data)
        buffered = BufferedByteReceiveStream(mock_stream)

        from anyio import DelimiterNotFound
        with pytest.raises(DelimiterNotFound):
            await buffered.receive_until(delimiter, max_bytes)

    anyio.run(test)
```

**Failing input**: `delimiter=b'\x00\x00', max_bytes=1`

## Reproducing the Bug

```python
import anyio
from anyio.streams.buffered import BufferedByteReceiveStream
from anyio import DelimiterNotFound


class MockByteReceiveStream:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self.pos >= len(self.data):
            raise anyio.EndOfStream
        chunk = self.data[self.pos:self.pos + max_bytes]
        self.pos += len(chunk)
        return chunk

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}


async def demonstrate_bug():
    data = b'\x00\x00'
    delimiter = b'\x00\x00'
    max_bytes = 1

    mock_stream = MockByteReceiveStream(data)
    buffered = BufferedByteReceiveStream(mock_stream)

    result = await buffered.receive_until(delimiter, max_bytes)

    print(f"Result: {result!r}")
    print(f"Expected: DelimiterNotFound to be raised")
    print(f"Bug: Function succeeded even though delimiter ends at byte {len(delimiter)}, exceeding max_bytes={max_bytes}")


anyio.run(demonstrate_bug)
```

## Why This Is A Bug

The documentation for `receive_until` states:

> :param max_bytes: maximum number of bytes that will be read before raising :exc:`~anyio.DelimiterNotFound`

In the reproduction case, the delimiter is found at position 0 and ends at position 2. To find this delimiter, the implementation must buffer at least 2 bytes. Since `max_bytes=1`, the function should raise `DelimiterNotFound(1)`, but instead it successfully returns `b''`.

The current implementation only checks if `len(buffer) >= max_bytes` BEFORE reading more data, but fails to check if the delimiter's position exceeds the limit AFTER finding it.

## Fix

```diff
--- a/anyio/streams/buffered.py
+++ b/anyio/streams/buffered.py
@@ -126,6 +126,9 @@ class BufferedByteReceiveStream(ByteReceiveStream):
             # Check if the delimiter can be found in the current buffer
             index = self._buffer.find(delimiter, offset)
             if index >= 0:
+                # Check if delimiter position exceeds the max_bytes limit
+                if index + delimiter_size > max_bytes:
+                    raise DelimiterNotFound(max_bytes)
                 found = self._buffer[:index]
                 del self._buffer[: index + len(delimiter) :]
                 return bytes(found)
```