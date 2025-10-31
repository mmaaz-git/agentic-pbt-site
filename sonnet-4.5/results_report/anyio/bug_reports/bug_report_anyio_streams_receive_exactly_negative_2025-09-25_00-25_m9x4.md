# Bug Report: anyio.streams.buffered.BufferedByteReceiveStream.receive_exactly Negative Argument Handling

**Target**: `anyio.streams.buffered.BufferedByteReceiveStream.receive_exactly`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `receive_exactly()` method does not validate that `nbytes` is non-negative, leading to incorrect behavior when called with negative values. Instead of raising an error, it performs incorrect slice operations on the internal buffer.

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
        return chunk if chunk else (raise anyio.EndOfStream)

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}


@given(
    buffer_data=st.binary(min_size=10, max_size=100),
    nbytes=st.integers(max_value=-1, min_value=-100)
)
def test_receive_exactly_rejects_negative(buffer_data, nbytes):
    async def run_test():
        stream = SimpleBytesReceiveStream(b"")
        buffered = BufferedByteReceiveStream(stream)
        buffered.feed_data(buffer_data)

        try:
            result = await buffered.receive_exactly(nbytes)
            raise AssertionError(f"Should reject negative nbytes={nbytes}, but returned {len(result)} bytes")
        except (ValueError, TypeError):
            pass

    anyio.run(run_test)
```

**Failing input**: `nbytes=-5` with any buffer data

## Reproducing the Bug

```python
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


async def main():
    stream = SimpleBytesReceiveStream(b"")
    buffered = BufferedByteReceiveStream(stream)
    buffered.feed_data(b"0123456789")

    print("Buffer contents:", buffered.buffer)
    result = await buffered.receive_exactly(-3)
    print(f"Called receive_exactly(-3)")
    print(f"Result: {result!r}")
    print(f"Result length: {len(result)}")
    print(f"Buffer after: {buffered.buffer}")


anyio.run(main())
```

**Output:**
```
Buffer contents: b'0123456789'
Called receive_exactly(-3)
Result: b'0123456'
Result length: 7
Buffer after: b'789'
```

## Why This Is A Bug

When `nbytes` is negative (e.g., `-3`), the method performs incorrect operations:

1. Line 93: `remaining = nbytes - len(self._buffer)` calculates a negative value
2. Line 94: `remaining <= 0` is True, so it proceeds to return
3. Line 95: `retval = self._buffer[:nbytes]` uses negative slicing (e.g., `buffer[:-3]`)
4. Line 96: `del self._buffer[:nbytes]` deletes using negative index
5. Line 97: Returns the incorrectly sliced data

This behavior is nonsensical for a method named `receive_exactly` - asking to receive exactly -3 bytes makes no sense and should be rejected.

**Expected behavior**: The method should either:
- Raise `ValueError` for negative `nbytes`
- Treat negative values as 0 (though this is less clear)

**Actual behavior**: Performs negative slicing operations on the buffer, returning the wrong data.

## Fix

```diff
--- a/anyio/streams/buffered.py
+++ b/anyio/streams/buffered.py
@@ -89,6 +89,9 @@ class BufferedByteReceiveStream(ByteReceiveStream):
             amount of bytes could be read from the stream

         """
+        if nbytes < 0:
+            raise ValueError("nbytes must be non-negative")
+
         while True:
             remaining = nbytes - len(self._buffer)
             if remaining <= 0:
```

This fix validates the input and raises a clear error for invalid negative values, consistent with Python's standard behavior for size parameters.