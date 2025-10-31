import anyio
from anyio.streams.buffered import BufferedByteReceiveStream
from anyio import DelimiterNotFound


class MockByteReceiveStream:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.read_count = 0

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self.pos >= len(self.data):
            raise anyio.EndOfStream
        chunk = self.data[self.pos:self.pos + max_bytes]
        self.pos += len(chunk)
        self.read_count += len(chunk)
        print(f"    MockStream.receive() called, returned {len(chunk)} bytes, total read: {self.read_count}")
        return chunk

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self):
        return {}


async def test_interpretation():
    print("Testing interpretation of max_bytes parameter:")
    print("=" * 60)

    print("\nInterpretation A: max_bytes = max data bytes before delimiter")
    print("  (The number of bytes returned, excluding delimiter)")

    print("\nInterpretation B: max_bytes = max total bytes read from stream")
    print("  (The total bytes that need to be read to find delimiter)")

    print("\n" + "=" * 60)

    # Test case that distinguishes the two interpretations
    print("\nTest: Data='01234AB' looking for delimiter='AB' with max_bytes=5")
    print("  - Delimiter is at position 5-6")
    print("  - Will return '01234' (5 bytes)")
    print("  - Total bytes needed to read: 7 (to include the delimiter)")

    data = b'01234AB'
    mock_stream = MockByteReceiveStream(data)
    buffered = BufferedByteReceiveStream(mock_stream)

    try:
        result = await buffered.receive_until(b'AB', 5)
        print(f"\nResult: {result!r} (length: {len(result)})")
        print("SUCCESS: Function returned the data")
        print("\nConclusion: Implementation follows Interpretation A")
        print("  max_bytes limits the returned data size, not total read size")
    except DelimiterNotFound as e:
        print(f"\nDelimiterNotFound raised with max_bytes={e.max_bytes}")
        print("\nConclusion: Implementation follows Interpretation B")
        print("  max_bytes limits the total bytes read from stream")

    print("\n" + "=" * 60)
    print("\nNow testing the bug case with this understanding:")
    print("\nTest: Data='AB' looking for delimiter='AB' with max_bytes=1")
    print("  - Delimiter is at position 0-1")
    print("  - Will return '' (0 bytes)")
    print("  - Total bytes needed to read: 2 (to include the delimiter)")

    data2 = b'AB'
    mock_stream2 = MockByteReceiveStream(data2)
    buffered2 = BufferedByteReceiveStream(mock_stream2)

    try:
        result = await buffered2.receive_until(b'AB', 1)
        print(f"\nResult: {result!r} (length: {len(result)})")
        print("BUG: Function succeeded even though it needed to read 2 bytes")
        print("     to find the delimiter, exceeding max_bytes=1")
    except DelimiterNotFound as e:
        print(f"\nDelimiterNotFound raised with max_bytes={e.max_bytes}")
        print("CORRECT: Function properly enforced the max_bytes limit")


anyio.run(test_interpretation)