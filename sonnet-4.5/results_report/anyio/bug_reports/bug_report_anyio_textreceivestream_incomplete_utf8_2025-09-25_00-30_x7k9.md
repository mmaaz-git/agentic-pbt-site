# Bug Report: anyio.streams.text.TextReceiveStream Incomplete UTF-8 Sequences

**Target**: `anyio.streams.text.TextReceiveStream.receive()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

TextReceiveStream fails to emit replacement characters for incomplete UTF-8 sequences at the end of a stream when using `errors='replace'`, silently dropping the incomplete bytes instead.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import anyio
from anyio.streams.text import TextReceiveStream


@given(
    incomplete_byte=st.sampled_from([
        b'\xc2',      # First byte of 2-byte sequence
        b'\xe0',      # First byte of 3-byte sequence
        b'\xf0',      # First byte of 4-byte sequence
    ])
)
def test_incomplete_utf8_sequence_at_end_of_stream(incomplete_byte):
    """
    Property: When a stream ends with an incomplete UTF-8 sequence,
    TextReceiveStream with errors='replace' should emit replacement characters.
    """
    async def test_impl():
        send_stream, receive_stream = anyio.create_memory_object_stream(1)
        text_receive = TextReceiveStream(receive_stream, encoding='utf-8', errors='replace')

        await send_stream.send(incomplete_byte)
        await send_stream.aclose()

        received = []
        async for chunk in text_receive:
            received.append(chunk)

        result = ''.join(received)

        assert '\ufffd' in result, \
            f"Expected replacement character for incomplete sequence {incomplete_byte!r}, got {result!r}"

    anyio.run(test_impl)
```

**Failing input**: `b'\xc2'` (first byte of a 2-byte UTF-8 sequence)

## Reproducing the Bug

```python
import anyio
from anyio.streams.text import TextReceiveStream

async def reproduce_bug():
    send_stream, receive_stream = anyio.create_memory_object_stream(1)
    text_receive = TextReceiveStream(receive_stream, encoding='utf-8', errors='replace')

    await send_stream.send(b'\xc2')
    await send_stream.aclose()

    received = []
    async for chunk in text_receive:
        received.append(chunk)

    result = ''.join(received)

    print(f"Received: {result!r}")
    print(f"Expected: '\\ufffd' (replacement character)")

    import codecs
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    correct = decoder.decode(b'\xc2', False) + decoder.decode(b'', True)
    print(f"Python's incremental decoder produces: {correct!r}")

anyio.run(reproduce_bug)
```

Output:
```
Received: ''
Expected: '\ufffd' (replacement character)
Python's incremental decoder produces: '�'
```

## Why This Is A Bug

Python's `IncrementalDecoder` has two modes:
- `decode(bytes, final=False)`: Returns complete characters, buffers incomplete sequences
- `decode(bytes, final=True)`: Returns complete characters and converts any buffered incomplete sequences to replacement characters

The TextReceiveStream.receive() method at `/home/npc/miniconda/lib/python3.13/site-packages/anyio/streams/text.py` lines 30-35:

```python
async def receive(self) -> str:
    while True:
        chunk = await self.transport_stream.receive()
        decoded = self._decoder.decode(chunk)
        if decoded:
            return decoded
```

The bug occurs because:
1. When `chunk=b'\xc2'`, `decode()` implicitly uses `final=False`, returning `''`
2. The `if decoded:` condition skips empty strings, continuing the loop
3. When the stream ends (raises `EndOfStream`), the exception propagates without calling `decode(b'', final=True)`
4. Incomplete sequences buffered in the decoder are lost

This violates the documented behavior of the `errors` parameter, which states it's "handling scheme for decoding errors" - incomplete sequences at stream end are decoding errors that should be handled.

## Fix

```diff
--- a/anyio/streams/text.py
+++ b/anyio/streams/text.py
@@ -28,10 +28,17 @@ class TextReceiveStream(ObjectReceiveStream[str]):
         self._decoder = decoder_class(errors=errors)

     async def receive(self) -> str:
-        while True:
-            chunk = await self.transport_stream.receive()
-            decoded = self._decoder.decode(chunk)
-            if decoded:
-                return decoded
+        try:
+            while True:
+                chunk = await self.transport_stream.receive()
+                decoded = self._decoder.decode(chunk)
+                if decoded:
+                    return decoded
+        except EndOfStream:
+            # Flush any incomplete sequences from the decoder
+            decoded = self._decoder.decode(b'', True)
+            if decoded:
+                return decoded
+            raise

     async def aclose(self) -> None:
```

This ensures that when the stream ends, any incomplete UTF-8 sequences are flushed and converted to replacement characters according to the specified error handling mode.