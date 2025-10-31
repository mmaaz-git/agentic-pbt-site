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
    # Test with -1
    data = b"HelloWorld"
    stream = SimpleByteStream(data)
    buffered = BufferedByteReceiveStream(stream)
    result = await buffered.receive(max_bytes=-1)
    print(f"max_bytes=-1: data={data!r}, result={result!r} (length={len(result)}), buffer={buffered.buffer!r}")

    # Test with -2
    data = b"HelloWorld"
    stream = SimpleByteStream(data)
    buffered = BufferedByteReceiveStream(stream)
    result = await buffered.receive(max_bytes=-2)
    print(f"max_bytes=-2: data={data!r}, result={result!r} (length={len(result)}), buffer={buffered.buffer!r}")

    # Test with -5
    data = b"HelloWorld"
    stream = SimpleByteStream(data)
    buffered = BufferedByteReceiveStream(stream)
    result = await buffered.receive(max_bytes=-5)
    print(f"max_bytes=-5: data={data!r}, result={result!r} (length={len(result)}), buffer={buffered.buffer!r}")

    # Test with 0
    data = b"HelloWorld"
    stream = SimpleByteStream(data)
    buffered = BufferedByteReceiveStream(stream)
    result = await buffered.receive(max_bytes=0)
    print(f"max_bytes=0: data={data!r}, result={result!r} (length={len(result)}), buffer={buffered.buffer!r}")


asyncio.run(main())