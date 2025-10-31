# Bug Report: anyio.streams.buffered.BufferedByteReceiveStream.receive_exactly Incorrect Behavior with Negative Arguments

**Target**: `anyio.streams.buffered.BufferedByteReceiveStream.receive_exactly`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `receive_exactly()` method returns incorrect data when called with negative `nbytes` values instead of raising an error, causing it to perform negative slice operations on the internal buffer.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages')

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


if __name__ == "__main__":
    test_receive_exactly_rejects_negative()
```

<details>

<summary>
**Failing input**: `buffer_data=b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', nbytes=-1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 51, in <module>
    test_receive_exactly_rejects_negative()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 32, in test_receive_exactly_rejects_negative
    buffer_data=st.binary(min_size=10, max_size=100),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 47, in test_receive_exactly_rejects_negative
    anyio.run(run_test)
    ~~~~~~~~~^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages/anyio/_core/_eventloop.py", line 74, in run
    return async_backend.run(func, args, {}, backend_options)
           ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2316, in run
    return runner.run(wrapper())
           ~~~~~~~~~~^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2304, in wrapper
    return await func(*args)
           ^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 43, in run_test
    raise AssertionError(f"Should reject negative nbytes={nbytes}, but returned {len(result)} bytes")
AssertionError: Should reject negative nbytes=-1, but returned 9 bytes
Falsifying example: test_receive_exactly_rejects_negative(
    # The test always failed when commented parts were varied together.
    buffer_data=b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',  # or any other generated value
    nbytes=-1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages')

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


anyio.run(main)
```

<details>

<summary>
Returns 7 bytes when asked for -3 bytes
</summary>
```
Buffer contents: b'0123456789'
Called receive_exactly(-3)
Result: b'0123456'
Result length: 7
Buffer after: b'789'
```
</details>

## Why This Is A Bug

The `receive_exactly()` method is supposed to receive exactly the specified number of bytes. However, when given a negative value like -3, it performs Python's negative slice operations on the internal buffer instead of raising an error.

Here's what happens internally when `nbytes=-3`:

1. **Line 93**: `remaining = nbytes - len(self._buffer) = -3 - 10 = -13`
2. **Line 94**: `remaining <= 0` evaluates to `True`, so execution continues
3. **Line 95**: `retval = self._buffer[:nbytes]` performs `self._buffer[:-3]`, which returns the first 7 bytes
4. **Line 96**: `del self._buffer[:nbytes]` deletes the first 7 bytes using negative indexing
5. **Line 97**: Returns 7 bytes instead of raising an error

This violates the method's contract - requesting "exactly -3 bytes" is semantically meaningless. The method should reject negative values with a clear error rather than returning an unexpected amount of data.

## Relevant Context

The Python standard library's asyncio module provides similar functionality through `StreamReader.readexactly()`, which explicitly validates input:

```python
# From asyncio/streams.py
async def readexactly(self, n):
    if n < 0:
        raise ValueError('readexactly size can not be less than zero')
    # ... rest of implementation
```

Since anyio aims to provide a compatible interface with asyncio, this inconsistent behavior could surprise developers familiar with asyncio's API. The current behavior could lead to:

- **Silent data corruption**: Applications might process the wrong amount of data without realizing it
- **Difficult-to-debug issues**: The bug only manifests when negative values are accidentally passed
- **API inconsistency**: Different behavior from the equivalent asyncio method

Documentation reference: The method's docstring at line 83-90 doesn't specify behavior for negative values, leaving this as undefined behavior that happens to produce incorrect results rather than failing explicitly.

## Proposed Fix

```diff
--- a/anyio/streams/buffered.py
+++ b/anyio/streams/buffered.py
@@ -89,6 +89,9 @@ class BufferedByteReceiveStream(ByteReceiveStream):
             amount of bytes could be read from the stream

         """
+        if nbytes < 0:
+            raise ValueError(f"nbytes must be non-negative, got {nbytes}")
+
         while True:
             remaining = nbytes - len(self._buffer)
             if remaining <= 0:
```