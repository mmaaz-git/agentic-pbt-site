from hypothesis import given, strategies as st, assume, settings
import anyio
from anyio.streams.buffered import BufferedByteReceiveStream
import pytest


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


@given(
    delimiter=st.binary(min_size=2, max_size=10),
    max_bytes=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=200)
def test_receive_until_enforces_max_bytes_limit(delimiter, max_bytes):
    assume(len(delimiter) > max_bytes)

    data = delimiter

    async def test():
        mock_stream = MockByteReceiveStream(data)
        buffered = BufferedByteReceiveStream(mock_stream)

        from anyio import DelimiterNotFound
        with pytest.raises(DelimiterNotFound):
            await buffered.receive_until(delimiter, max_bytes)

    anyio.run(test)


if __name__ == "__main__":
    test_receive_until_enforces_max_bytes_limit()