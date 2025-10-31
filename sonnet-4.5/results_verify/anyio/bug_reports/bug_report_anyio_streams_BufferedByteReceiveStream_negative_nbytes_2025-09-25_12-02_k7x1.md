# Bug Report: anyio.streams.buffered.BufferedByteReceiveStream.receive_exactly() Missing Input Validation

**Target**: `anyio.streams.buffered.BufferedByteReceiveStream.receive_exactly()`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`BufferedByteReceiveStream.receive_exactly()` does not validate that `nbytes` is non-negative, causing unexpected behavior when negative values are passed.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
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


@given(data=st.binary(min_size=10, max_size=100), nbytes=st.integers(max_value=-1))
def test_receive_exactly_rejects_negative_nbytes(data: bytes, nbytes: int):
    async def run():
        mock_stream = MockByteReceiveStream(data)
        buffered = BufferedByteReceiveStream(mock_stream)

        try:
            await buffered.receive_exactly(nbytes)
            assert False, "Should raise ValueError for negative nbytes"
        except ValueError:
            pass

    asyncio.run(run())
```

**Failing input**: `nbytes=-1` (or any negative integer)

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
    data = b"Hello, World!"
    mock_stream = MockByteReceiveStream(data)
    buffered = BufferedByteReceiveStream(mock_stream)

    result = await buffered.receive_exactly(-5)
    print(f"Result: {result}")
    print(f"Expected: ValueError")
    print(f"Actual: Returns {len(result)} bytes using negative slice")


asyncio.run(main())
```

Expected: Raises `ValueError` for negative nbytes
Actual: Returns data using Python's negative slicing semantics (`buffer[:-5]`)

## Why This Is A Bug

The `receive_exactly()` method in `/lib/python3.13/site-packages/anyio/streams/buffered.py` lines 82-108 does not validate the `nbytes` parameter. The docstring states it should "Read exactly the given amount of bytes", which implies a non-negative value.

When `nbytes` is negative, the code at lines 94-97 uses Python's negative slice syntax:

```python
if remaining <= 0:
    retval = self._buffer[:nbytes]  # negative nbytes causes negative slice
    del self._buffer[:nbytes]
    return bytes(retval)
```

This causes `self._buffer[:-5]` to return all but the last 5 bytes, which is not the documented behavior of "reading exactly N bytes".

The function should validate input and raise `ValueError` for invalid arguments, as is standard practice in Python (e.g., `bytes.decode()`, `str.encode()`, etc.).

## Fix

Add input validation at the start of the method:

```diff
 async def receive_exactly(self, nbytes: int) -> bytes:
     """
     Read exactly the given amount of bytes from the stream.

     :param nbytes: the number of bytes to read
     :return: the bytes read
     :raises ~anyio.IncompleteRead: if the stream was closed before the requested
         amount of bytes could be read from the stream

     """
+    if nbytes < 0:
+        raise ValueError("nbytes must be non-negative")
+
     while True:
         remaining = nbytes - len(self._buffer)
         if remaining <= 0:
             retval = self._buffer[:nbytes]
             del self._buffer[:nbytes]
             return bytes(retval)
```