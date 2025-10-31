# Bug Report: anyio.streams.text.TextStream Resource Leak in aclose()

**Target**: `anyio.streams.text.TextStream.aclose()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`TextStream.aclose()` fails to close the receive stream if closing the send stream raises an exception, leading to a resource leak.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import asyncio
from anyio.streams.text import TextStream


class MockByteStream:
    def __init__(self, send_fails: bool):
        self.send_fails = send_fails
        self.send_closed = False
        self.receive_closed = False

    async def send(self, item: bytes):
        pass

    async def receive(self, max_bytes: int = 65536):
        return b"test"

    async def send_eof(self):
        pass

    async def aclose(self):
        if self.send_fails and not self.send_closed:
            self.send_closed = True
            raise RuntimeError("Send close failed")
        if self.send_closed and not self.receive_closed:
            self.receive_closed = True
        else:
            self.send_closed = True
            self.receive_closed = True

    @property
    def extra_attributes(self):
        return {}


@given(st.booleans())
def test_textstream_aclose_closes_both_streams(send_fails: bool):
    async def run():
        stream_mock = MockByteStream(send_fails)
        text_stream = TextStream(stream_mock)

        try:
            await text_stream.aclose()
        except RuntimeError:
            pass

        assert stream_mock.receive_closed, \
            "Receive stream should be closed even if send stream close fails"

    asyncio.run(run())
```

**Failing input**: `send_fails=True`

## Reproducing the Bug

```python
import asyncio
from anyio.streams.text import TextStream, TextReceiveStream, TextSendStream


class FailingByteStream:
    async def send(self, item: bytes):
        pass

    async def receive(self, max_bytes: int = 65536):
        return b"test"

    async def send_eof(self):
        pass

    async def aclose(self):
        raise RuntimeError("Send stream close failed!")

    @property
    def extra_attributes(self):
        return {}


async def demonstrate_leak():
    text_stream = TextStream(FailingByteStream())

    try:
        await text_stream.aclose()
    except RuntimeError as e:
        print(f"Exception raised: {e}")
        print("Bug: receive_stream.aclose() was never called!")


asyncio.run(demonstrate_leak())
```

Expected: Both send and receive streams are closed, even if an exception occurs
Actual: If send_stream.aclose() raises, receive_stream.aclose() is never called

## Why This Is A Bug

The `TextStream.aclose()` method in `/lib/python3.13/site-packages/anyio/streams/text.py` lines 146-148:

```python
async def aclose(self) -> None:
    await self._send_stream.aclose()
    await self._receive_stream.aclose()
```

If `self._send_stream.aclose()` raises an exception, the execution stops and `self._receive_stream.aclose()` is never called. This leaves the receive stream open, causing a resource leak. The same pattern exists in `StapledByteStream.aclose()` (stapled.py:46-48) and `StapledObjectStream.aclose()` (stapled.py:82-84).

Python's async context managers and cleanup code should ensure all resources are closed even when exceptions occur.

## Fix

Use exception handling to ensure both streams are closed:

```diff
 async def aclose(self) -> None:
-    await self._send_stream.aclose()
-    await self._receive_stream.aclose()
+    try:
+        await self._send_stream.aclose()
+    finally:
+        await self._receive_stream.aclose()
```

Alternatively, use `asyncio.gather` with `return_exceptions=True`:

```diff
+import asyncio
+
 async def aclose(self) -> None:
-    await self._send_stream.aclose()
-    await self._receive_stream.aclose()
+    await asyncio.gather(
+        self._send_stream.aclose(),
+        self._receive_stream.aclose(),
+        return_exceptions=True
+    )
```

The same fix should be applied to `StapledByteStream.aclose()` and `StapledObjectStream.aclose()` which have identical issues.