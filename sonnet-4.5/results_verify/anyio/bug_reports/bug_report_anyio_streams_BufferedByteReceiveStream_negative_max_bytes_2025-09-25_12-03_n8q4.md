# Bug Report: anyio.streams.buffered.BufferedByteReceiveStream.receive_until() Missing Input Validation

**Target**: `anyio.streams.buffered.BufferedByteReceiveStream.receive_until()`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`BufferedByteReceiveStream.receive_until()` does not validate that `max_bytes` is positive, causing immediate `DelimiterNotFound` exception when negative or zero values are passed.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import asyncio
import anyio
from anyio.streams.buffered import BufferedByteReceiveStream


class MockByteReceiveStream:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self._pos >= len(self._data):
            raise anyio.EndOfStream
        chunk = self._data[self._pos : self._pos + max_bytes]
        self._pos += len(chunk)
        return chunk

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}


@given(max_bytes=st.integers(max_value=0))
def test_receive_until_rejects_non_positive_max_bytes(max_bytes: int):
    async def run():
        data = b"test\ndata"
        mock_stream = MockByteReceiveStream(data)
        buffered = BufferedByteReceiveStream(mock_stream)

        try:
            await buffered.receive_until(b"\n", max_bytes=max_bytes)
            assert False, "Should raise ValueError for non-positive max_bytes"
        except ValueError:
            pass
        except anyio.DelimiterNotFound:
            assert False, "Should raise ValueError, not DelimiterNotFound"

    asyncio.run(run())
```

**Failing input**: `max_bytes=-1` or `max_bytes=0`

## Reproducing the Bug

```python
import asyncio
import anyio
from anyio.streams.buffered import BufferedByteReceiveStream


class MockByteReceiveStream:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self._pos >= len(self._data):
            raise anyio.EndOfStream
        chunk = self._data[self._pos : self._pos + max_bytes]
        self._pos += len(chunk)
        return chunk

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}


async def main():
    data = b"Hello\nWorld!"
    mock_stream = MockByteReceiveStream(data)
    buffered = BufferedByteReceiveStream(mock_stream)

    try:
        result = await buffered.receive_until(b"\n", max_bytes=-1)
    except anyio.DelimiterNotFound as e:
        print(f"Raised: {type(e).__name__}(max_bytes={e.args[0]})")
        print(f"Expected: ValueError('max_bytes must be positive')")
        print(f"Actual: DelimiterNotFound({e.args[0]})")


asyncio.run(main())
```

Expected: Raises `ValueError` for non-positive max_bytes
Actual: Raises `DelimiterNotFound(-1)` immediately

## Why This Is A Bug

The `receive_until()` method in `/lib/python3.13/site-packages/anyio/streams/buffered.py` lines 109-146 does not validate the `max_bytes` parameter. When `max_bytes` is negative or zero, the check at line 134:

```python
if len(self._buffer) >= max_bytes:
    raise DelimiterNotFound(max_bytes)
```

will always be `True` (since `len(self._buffer)` is always >= 0), causing an immediate `DelimiterNotFound` exception even before attempting to find the delimiter.

This is confusing to users because:
1. The exception suggests the delimiter was not found within max_bytes, when actually the parameter was invalid
2. It violates the principle of early validation - invalid input should be rejected with a clear `ValueError`
3. The docstring implies max_bytes should be positive ("maximum number of bytes that will be read")

## Fix

Add input validation at the start of the method:

```diff
 async def receive_until(self, delimiter: bytes, max_bytes: int) -> bytes:
     """
     Read from the stream until the delimiter is found or max_bytes have been read.

     :param delimiter: the marker to look for in the stream
     :param max_bytes: maximum number of bytes that will be read before raising
         :exc:`~anyio.DelimiterNotFound`
     :return: the bytes read (not including the delimiter)
     :raises ~anyio.IncompleteRead: if the stream was closed before the delimiter
         was found
     :raises ~anyio.DelimiterNotFound: if the delimiter is not found within the
         bytes read up to the maximum allowed

     """
+    if max_bytes <= 0:
+        raise ValueError("max_bytes must be positive")
+
     delimiter_size = len(delimiter)
     offset = 0
     while True:
```