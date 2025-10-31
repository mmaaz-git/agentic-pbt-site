import asyncio
from hypothesis import given, strategies as st, settings
from anyio.streams.buffered import BufferedByteReceiveStream
from anyio import EndOfStream


class MockByteReceiveStream:
    def __init__(self, data_chunks):
        self.data_chunks = list(data_chunks)
        self.index = 0
        self.extra_attributes = {}

    async def receive(self, max_bytes=65536):
        if self.index >= len(self.data_chunks):
            raise EndOfStream
        chunk = self.data_chunks[self.index]
        self.index += 1
        return chunk

    async def aclose(self):
        pass


@given(st.binary(min_size=10, max_size=100))
@settings(max_examples=100)
def test_receive_negative_max_bytes(data):
    async def test():
        stream = MockByteReceiveStream([data])
        buffered = BufferedByteReceiveStream(stream)

        result = await buffered.receive(-1)
        assert len(result) >= 0

    asyncio.run(test())

# Run the test
test_receive_negative_max_bytes()