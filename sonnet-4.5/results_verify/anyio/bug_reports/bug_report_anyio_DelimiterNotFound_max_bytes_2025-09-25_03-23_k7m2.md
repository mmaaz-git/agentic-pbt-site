# Bug Report: anyio DelimiterNotFound loses max_bytes value

**Target**: `anyio._core._exceptions.DelimiterNotFound`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `DelimiterNotFound` exception accepts a `max_bytes` parameter but only uses it to format an error message, discarding the original integer value. This prevents users from programmatically accessing the limit that caused the exception, forcing them to parse the string message or maintain external state.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import anyio
from anyio.streams.buffered import BufferedByteReceiveStream

@given(
    data=st.binary(min_size=10, max_size=1000),
    delimiter=st.binary(min_size=1, max_size=10),
    max_bytes=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=100)
def test_delimiter_not_found_preserves_max_bytes(data, delimiter, max_bytes):
    assume(delimiter not in data[:max_bytes])
    assume(len(data) >= max_bytes)

    async def run_test():
        stream = BufferedByteReceiveStream(ByteSourceStream(data))
        try:
            await stream.receive_until(delimiter, max_bytes)
        except anyio.DelimiterNotFound as exc:
            assert hasattr(exc, 'max_bytes'), "Exception should preserve max_bytes"
            assert exc.max_bytes == max_bytes, f"Expected {max_bytes}, got {exc.max_bytes}"

    anyio.run(run_test)
```

**Failing input**: Any input where delimiter is not found within max_bytes

## Reproducing the Bug

```python
import anyio
from anyio.streams.buffered import BufferedByteReceiveStream


class ByteSourceStream:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    async def receive(self) -> bytes:
        if self.pos >= len(self.data):
            raise anyio.EndOfStream
        chunk = self.data[self.pos:self.pos + 1024]
        self.pos += len(chunk)
        return chunk

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}


async def main():
    data = b"No delimiter here"
    delimiter = b"|"
    max_bytes = 10

    stream = BufferedByteReceiveStream(ByteSourceStream(data))

    try:
        await stream.receive_until(delimiter, max_bytes)
    except anyio.DelimiterNotFound as e:
        print(f"Exception args: {e.args}")
        print(f"args[0] type: {type(e.args[0])}")
        print(f"Has max_bytes attr: {hasattr(e, 'max_bytes')}")

anyio.run(main)
```

**Output:**
```
Exception args: ('The delimiter was not found among the first 10 bytes',)
args[0] type: <class 'str'>
Has max_bytes attr: False
```

## Why This Is A Bug

The `DelimiterNotFound` exception is designed to carry information about the condition that caused it (the `max_bytes` limit). However, it only uses this value to format a message, then discards it. This violates the principle that exceptions should preserve structured information for programmatic error handling.

**Real-world impact:**
- Users cannot programmatically retry with a larger buffer: `new_limit = e.max_bytes * 2`
- Users cannot log the limit separately from the message
- Users cannot distinguish between different limits without parsing strings
- Testing and error handling become more fragile

**Precedent:** The `BrokenWorkerInterpreter` exception in the same file (lines 32-42) preserves its `excinfo` parameter as an attribute, showing that preserving exception parameters is already a pattern in anyio.

## Fix

```diff
--- a/anyio/_core/_exceptions.py
+++ b/anyio/_core/_exceptions.py
@@ -99,6 +99,7 @@ class DelimiterNotFound(Exception):

     def __init__(self, max_bytes: int) -> None:
+        self.max_bytes = max_bytes
         super().__init__(
             f"The delimiter was not found among the first {max_bytes} bytes"
         )
```

This simple one-line fix preserves the `max_bytes` value as an instance attribute while maintaining backward compatibility with existing code that only uses the string message.