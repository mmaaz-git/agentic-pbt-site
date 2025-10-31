# Bug Report: anyio.streams.buffered BufferedByteReceiveStream.receive_until Ignores max_bytes When Delimiter Found

**Target**: `anyio.streams.buffered.BufferedByteReceiveStream.receive_until`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`BufferedByteReceiveStream.receive_until()` violates its documented contract by successfully returning data when the delimiter is found, even if finding the delimiter requires reading more bytes than the `max_bytes` limit allows.

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


if __name__ == "__main__":
    test_receive_until_enforces_max_bytes_limit()
```

<details>

<summary>
**Failing input**: `delimiter=b'\x00\x00', max_bytes=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 49, in <module>
    test_receive_until_enforces_max_bytes_limit()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 28, in test_receive_until_enforces_max_bytes_limit
    delimiter=st.binary(min_size=2, max_size=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 45, in test_receive_until_enforces_max_bytes_limit
    anyio.run(test)
    ~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_core/_eventloop.py", line 74, in run
    return async_backend.run(func, args, {}, backend_options)
           ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2310, in run
    return runner.run(wrapper())
           ~~~~~~~~~~^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2298, in wrapper
    return await func(*args)
           ^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 42, in test
    with pytest.raises(DelimiterNotFound):
         ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: DID NOT RAISE <class 'anyio.DelimiterNotFound'>
Falsifying example: test_receive_until_enforces_max_bytes_limit(
    delimiter=b'\x00\x00',  # or any other generated value
    max_bytes=1,
)
```
</details>

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

    try:
        result = await buffered.receive_until(delimiter, max_bytes)
        print(f"Result: {result!r}")
        print(f"Expected: DelimiterNotFound to be raised")
        print(f"Bug: Function succeeded even though delimiter ends at byte {len(delimiter)}, exceeding max_bytes={max_bytes}")
    except DelimiterNotFound as e:
        print(f"DelimiterNotFound raised correctly with max_bytes={e.args[0]}")
        print("No bug - working as expected")


anyio.run(demonstrate_bug)
```

<details>

<summary>
Function returns successfully instead of raising DelimiterNotFound
</summary>
```
Result: b''
Expected: DelimiterNotFound to be raised
Bug: Function succeeded even though delimiter ends at byte 2, exceeding max_bytes=1
```
</details>

## Why This Is A Bug

The `receive_until` method's documentation explicitly states that `max_bytes` is the "maximum number of bytes that will be read before raising DelimiterNotFound". This creates a contract that the function will not read more than `max_bytes` from the stream when searching for the delimiter.

In the failing case:
- The delimiter is `b'\x00\x00'` (2 bytes long)
- `max_bytes` is set to 1
- To find a 2-byte delimiter, the function must read at least 2 bytes from the stream
- Since 2 > 1, the function should raise `DelimiterNotFound(1)` to indicate the delimiter cannot be found within the byte limit

However, the implementation has a logic flaw in lines 99-109 of `buffered.py`:
1. The function first searches for the delimiter in the buffer (line 101)
2. If found, it immediately returns the data before the delimiter (lines 103-105)
3. The `max_bytes` check (line 108) only happens when the delimiter is NOT found
4. This means the function can successfully return even when the delimiter position exceeds `max_bytes`

This violates the documented contract and can lead to:
- Unexpected memory usage by reading more data than intended
- Security implications when `max_bytes` is used to limit data exposure
- Inconsistent behavior where the same limit is enforced differently based on whether the delimiter is found

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/anyio/streams/buffered.py` at lines 99-109. The issue arises from the order of operations: the delimiter search happens before the max_bytes validation.

The `DelimiterNotFound` exception documentation (from anyio's exceptions) states: "Raised during receive_until if the maximum number of bytes has been read without the delimiter being found." This confirms that `max_bytes` should limit the total bytes read, not just the bytes returned.

The current implementation creates an inconsistency: if you call `receive_until(b'XX', max_bytes=1)` on a stream containing `b'XX'`, it succeeds and returns `b''`. But if the stream contains `b'YXX'`, it would raise `DelimiterNotFound(1)` because the buffer would exceed max_bytes before finding the delimiter. This inconsistent behavior makes the API unpredictable.

## Proposed Fix

```diff
--- a/anyio/streams/buffered.py
+++ b/anyio/streams/buffered.py
@@ -100,6 +100,9 @@ class BufferedByteReceiveStream(ByteReceiveStream):
             # Check if the delimiter can be found in the current buffer
             index = self._buffer.find(delimiter, offset)
             if index >= 0:
+                # Check if finding the delimiter exceeds the max_bytes limit
+                if index + delimiter_size > max_bytes:
+                    raise DelimiterNotFound(max_bytes)
                 found = self._buffer[:index]
                 del self._buffer[: index + len(delimiter) :]
                 return bytes(found)
```