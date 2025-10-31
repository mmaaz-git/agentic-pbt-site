from hypothesis import given, strategies as st, settings, assume
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


# Run the test
if __name__ == "__main__":
    test_delimiter_not_found_preserves_max_bytes()
    print("All tests passed!")