# Bug Report: anyio BufferedByteReceiveStream.receive_until() Delimiter Boundary Split Bug

**Target**: `anyio.streams.buffered.BufferedByteReceiveStream.receive_until()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `receive_until()` method incorrectly raises `DelimiterNotFound` when a delimiter is split across the `max_bytes` boundary in chunked data, even though the delimiter exists within the allowed search range.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, example
from anyio.streams.buffered import BufferedByteReceiveStream
from anyio import DelimiterNotFound, EndOfStream
import asyncio
from dataclasses import dataclass


@dataclass
class MockStream:
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


@given(
    delimiter=st.binary(min_size=2, max_size=10),
    max_bytes=st.integers(min_value=10, max_value=1000),
    prefix_size=st.integers(min_value=0, max_value=50)
)
@example(delimiter=b'XX', max_bytes=50, prefix_size=0)  # The specific failing case
def test_delimiter_at_boundary_should_be_found(delimiter, max_bytes, prefix_size):
    assume(prefix_size < max_bytes)

    # Calculate position where delimiter will start (right before max_bytes)
    position = max_bytes - len(delimiter) + 1
    assume(position > 0)

    # Create chunks that split the delimiter across the boundary
    # First chunk ends just before completing the delimiter
    chunk1 = b'A' * position + delimiter[:len(delimiter)//2]
    # Second chunk starts with the rest of the delimiter
    chunk2 = delimiter[len(delimiter)//2:] + b'rest'

    async def run_test():
        stream = BufferedByteReceiveStream(MockStream([chunk1, chunk2]))

        try:
            result = await stream.receive_until(delimiter, max_bytes)
            # Should succeed - delimiter exists in the stream
            assert delimiter not in result, "Result should not contain delimiter"
            assert len(result) == position, f"Expected {position} bytes, got {len(result)}"
            print(f"✓ Test passed for delimiter={delimiter!r}, max_bytes={max_bytes}")
        except DelimiterNotFound:
            # This is the bug - delimiter exists but wasn't found
            print(f"✗ BUG: DelimiterNotFound for delimiter={delimiter!r} at position {position} with max_bytes={max_bytes}")
            print(f"  Delimiter spans boundary: chunk1 ends with {chunk1[-5:]!r}, chunk2 starts with {chunk2[:5]!r}")
            raise

    asyncio.run(run_test())


if __name__ == "__main__":
    # Run the property-based test
    test_delimiter_at_boundary_should_be_found()
```

<details>

<summary>
**Failing input**: `delimiter=b'XX', max_bytes=50, prefix_size=0`
</summary>
```
✗ BUG: DelimiterNotFound for delimiter=b'XX' at position 49 with max_bytes=50
  Delimiter spans boundary: chunk1 ends with b'AAAAX', chunk2 starts with b'Xrest'
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 64, in <module>
    test_delimiter_at_boundary_should_be_found()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 26, in test_delimiter_at_boundary_should_be_found
    delimiter=st.binary(min_size=2, max_size=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 59, in test_delimiter_at_boundary_should_be_found
    asyncio.run(run_test())
    ~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 48, in run_test
    result = await stream.receive_until(delimiter, max_bytes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/streams/buffered.py", line 109, in receive_until
    raise DelimiterNotFound(max_bytes)
anyio.DelimiterNotFound: The delimiter was not found among the first 50 bytes
Falsifying explicit example: test_delimiter_at_boundary_should_be_found(
    delimiter=b'XX',
    max_bytes=50,
    prefix_size=0,
)
```
</details>

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

    # Delimiter spans positions 49-50
    chunk1 = b'A' * 49 + b'X'  # First 50 bytes, ends with first 'X'
    chunk2 = b'X' + b'rest of data'  # Starts with second 'X'

    stream = BufferedByteReceiveStream(MockByteReceiveStream([chunk1, chunk2]))

    try:
        result = await stream.receive_until(delimiter, max_bytes)
        print(f"Success: Found delimiter, received {len(result)} bytes")
        print(f"Result: {result[:20]}... (truncated)" if len(result) > 20 else f"Result: {result}")
    except DelimiterNotFound as e:
        print(f"BUG: DelimiterNotFound raised: {e}")
        print(f"But delimiter 'XX' exists at positions 49-50 in the stream!")
        print(f"First chunk ends with: {chunk1[-5:]}")
        print(f"Second chunk starts with: {chunk2[:5]}")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(test_bug())
```

<details>

<summary>
BUG: DelimiterNotFound raised when delimiter spans boundary
</summary>
```
BUG: DelimiterNotFound raised: The delimiter was not found among the first 50 bytes
But delimiter 'XX' exists at positions 49-50 in the stream!
First chunk ends with: b'AAAAX'
Second chunk starts with: b'Xrest'
```
</details>

## Why This Is A Bug

This violates the expected behavior of `receive_until()` for several critical reasons:

1. **Inconsistent chunk-dependent behavior**: The same logical stream data produces different results based on how it's chunked by the underlying transport:
   - If the data arrives as a single chunk containing positions 0-50, the delimiter at position 49-50 would be found
   - If the data arrives split at position 50 (first chunk: 0-49, second chunk: 50+), the delimiter is NOT found

2. **Violates stream abstraction**: Stream processing should be agnostic to how data arrives from the network. Users should not need to know or care about underlying chunking.

3. **Contradicts documentation intent**: The docstring states `max_bytes` is the "maximum number of bytes that will be read before raising DelimiterNotFound." The delimiter starting at position 49 (within the first 50 bytes) should logically be found since its starting position is within the search range.

4. **Premature termination**: The implementation checks `if len(self._buffer) >= max_bytes` at line 134 and raises `DelimiterNotFound` before attempting to read more data that could complete a partially-seen delimiter. In our example:
   - After reading the first chunk (50 bytes), buffer contains `b'A'*49 + b'X'`
   - The check `50 >= 50` evaluates to True
   - Exception is raised without reading the second chunk that contains the completing `b'X'`

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/anyio/streams/buffered.py` at line 134. The issue stems from the interaction between:

- Line 127: Finding delimiter in the current buffer
- Line 134: Checking if buffer has reached `max_bytes` limit
- Line 139: Reading more data from the stream

The current logic doesn't account for delimiters that may be partially present at the boundary. This is a common scenario in network protocols where delimiters can naturally occur at any position in the stream.

Similar implementations in other libraries (e.g., Python's `asyncio.StreamReader.readuntil()`) handle this edge case by allowing reading slightly beyond `max_bytes` to check for delimiter completion.

## Proposed Fix

The fix should allow reading enough additional bytes to check if a delimiter that starts before `max_bytes` completes just after the boundary:

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

This allows the buffer to grow up to `max_bytes + delimiter_size - 1` bytes, ensuring that any delimiter starting at position `max_bytes - 1` can be fully checked.