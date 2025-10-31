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
        return "BUG_CONFIRMED"
    except DelimiterNotFound as e:
        print(f"DelimiterNotFound raised as expected with max_bytes: {e.max_bytes}")
        return "NO_BUG"
    except Exception as e:
        print(f"Unexpected exception: {e}")
        return "ERROR"


print("Testing the reported bug...")
result = anyio.run(demonstrate_bug)
print(f"Test result: {result}")