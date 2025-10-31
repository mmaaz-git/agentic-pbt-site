# Bug Report: anyio.streams.text.TextReceiveStream Empty String Handling

**Target**: `anyio.streams.text.TextReceiveStream.receive`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`TextReceiveStream.receive()` incorrectly raises `EndOfStream` when attempting to receive an empty string that was encoded and sent, breaking the encode/decode round-trip property for empty strings.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import anyio
from anyio.streams.text import TextReceiveStream, TextSendStream


class MockByteReceiveStream:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    async def receive(self, max_bytes: int = 65536) -> bytes:
        from anyio import EndOfStream
        if self.pos >= len(self.data):
            raise EndOfStream
        chunk = self.data[self.pos:self.pos + max_bytes]
        self.pos += len(chunk)
        return chunk

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}


class MockByteSendStream:
    def __init__(self):
        self.sent_data = bytearray()

    async def send(self, data: bytes) -> None:
        self.sent_data.extend(data)

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}


@given(st.text())
def test_text_stream_encode_decode_roundtrip(text):
    async def test():
        mock_byte_stream = MockByteSendStream()
        text_send = TextSendStream(mock_byte_stream)
        await text_send.send(text)

        mock_receive = MockByteReceiveStream(bytes(mock_byte_stream.sent_data))
        text_receive = TextReceiveStream(mock_receive)

        received = await text_receive.receive()
        assert received == text

    anyio.run(test)
```

**Failing input**: `text=''`

## Reproducing the Bug

```python
import anyio
from anyio.streams.text import TextReceiveStream, TextSendStream


class MockByteReceiveStream:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    async def receive(self, max_bytes: int = 65536) -> bytes:
        from anyio import EndOfStream
        if self.pos >= len(self.data):
            raise EndOfStream
        chunk = self.data[self.pos:self.pos + max_bytes]
        self.pos += len(chunk)
        return chunk

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}


class MockByteSendStream:
    def __init__(self):
        self.sent_data = bytearray()

    async def send(self, data: bytes) -> None:
        self.sent_data.extend(data)

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}


async def main():
    text = ""

    mock_byte_stream = MockByteSendStream()
    text_send = TextSendStream(mock_byte_stream)
    await text_send.send(text)

    mock_receive = MockByteReceiveStream(bytes(mock_byte_stream.sent_data))
    text_receive = TextReceiveStream(mock_receive)

    received = await text_receive.receive()
    assert received == text


anyio.run(main)
```

## Why This Is A Bug

The `TextReceiveStream.receive()` method implements a round-trip encoding property where `decode(encode(text)) == text` should hold for all strings. However, it fails for empty strings because:

1. An empty string encodes to `b''` (empty bytes)
2. When `receive()` is called, it reads the empty bytes and decodes them to `''`
3. The condition `if decoded:` on line 57 treats the empty string as falsy
4. The loop continues and tries to receive more bytes
5. Since there are no more bytes, `EndOfStream` is raised

This violates the fundamental property that encoding and decoding should be inverse operations.

## Fix

The issue is on line 57 where `if decoded:` treats empty strings as falsy. The loop should return the decoded result when it's a complete string (even if empty), or when the underlying stream has no more data. One approach is to check if we received an empty chunk (indicating end of data) and return the decoded result in that case:

```diff
--- a/anyio/streams/text.py
+++ b/anyio/streams/text.py
@@ -53,8 +53,11 @@ class TextReceiveStream(ObjectReceiveStream[str]):
     async def receive(self) -> str:
         while True:
             chunk = await self.transport_stream.receive()
             decoded = self._decoder.decode(chunk)
-            if decoded:
+            if decoded or not chunk:
                 return decoded
+
```

This ensures that when we receive an empty chunk (which happens with empty strings), we return the decoded result immediately rather than continuing to loop.