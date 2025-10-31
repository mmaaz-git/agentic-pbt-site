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

        # Let's also check what attributes it does have
        print(f"Exception attributes: {[attr for attr in dir(e) if not attr.startswith('_')]}")

        # Let's check if we can access max_bytes
        try:
            print(f"max_bytes value: {e.max_bytes}")
        except AttributeError as ae:
            print(f"Cannot access max_bytes: {ae}")

anyio.run(main)