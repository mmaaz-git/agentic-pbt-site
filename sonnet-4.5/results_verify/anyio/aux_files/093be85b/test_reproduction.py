import asyncio
from anyio.streams.buffered import BufferedByteReceiveStream
from anyio import EndOfStream


class SimpleByteStream:
    def __init__(self, data):
        self.data = data
        self.consumed = False
        self.extra_attributes = {}

    async def receive(self, max_bytes=65536):
        if self.consumed:
            raise EndOfStream
        self.consumed = True
        return self.data

    async def aclose(self):
        pass


async def main():
    data = b"HelloWorld"
    stream = SimpleByteStream(data)
    buffered = BufferedByteReceiveStream(stream)

    result = await buffered.receive(max_bytes=-1)
    print(f"Called receive(max_bytes=-1) on {data!r}")
    print(f"Result: {result!r} (length: {len(result)})")
    print(f"Buffer: {buffered.buffer!r}")


asyncio.run(main())