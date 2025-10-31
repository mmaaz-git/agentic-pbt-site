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


async def test_cases():
    print("Testing various edge cases:")

    # Test 1: Delimiter at the exact boundary
    print("\n1. Delimiter b'AB' with max_bytes=2 (should work)")
    data = b'AB'
    mock_stream = MockByteReceiveStream(data)
    buffered = BufferedByteReceiveStream(mock_stream)
    try:
        result = await buffered.receive_until(b'AB', 2)
        print(f"   Result: {result!r} - SUCCESS")
    except DelimiterNotFound as e:
        print(f"   DelimiterNotFound raised with max_bytes={e.max_bytes}")

    # Test 2: Data before delimiter within limit
    print("\n2. Data b'XAB' looking for delimiter b'AB' with max_bytes=3")
    data = b'XAB'
    mock_stream = MockByteReceiveStream(data)
    buffered = BufferedByteReceiveStream(mock_stream)
    try:
        result = await buffered.receive_until(b'AB', 3)
        print(f"   Result: {result!r} - SUCCESS")
    except DelimiterNotFound as e:
        print(f"   DelimiterNotFound raised with max_bytes={e.max_bytes}")

    # Test 3: Data before delimiter exceeds limit
    print("\n3. Data b'XYAB' looking for delimiter b'AB' with max_bytes=3")
    data = b'XYAB'
    mock_stream = MockByteReceiveStream(data)
    buffered = BufferedByteReceiveStream(mock_stream)
    try:
        result = await buffered.receive_until(b'AB', 3)
        print(f"   Result: {result!r} - UNEXPECTED SUCCESS")
    except DelimiterNotFound as e:
        print(f"   DelimiterNotFound raised with max_bytes={e.max_bytes} - EXPECTED")

    # Test 4: The reported bug case
    print("\n4. Data b'\\x00\\x00' looking for delimiter b'\\x00\\x00' with max_bytes=1")
    data = b'\x00\x00'
    mock_stream = MockByteReceiveStream(data)
    buffered = BufferedByteReceiveStream(mock_stream)
    try:
        result = await buffered.receive_until(b'\x00\x00', 1)
        print(f"   Result: {result!r} - BUG: Should have raised DelimiterNotFound")
    except DelimiterNotFound as e:
        print(f"   DelimiterNotFound raised with max_bytes={e.max_bytes} - EXPECTED")

    # Test 5: Delimiter not in data
    print("\n5. Data b'ABCD' looking for delimiter b'XY' with max_bytes=10")
    data = b'ABCD'
    mock_stream = MockByteReceiveStream(data)
    buffered = BufferedByteReceiveStream(mock_stream)
    try:
        result = await buffered.receive_until(b'XY', 10)
        print(f"   Result: {result!r} - UNEXPECTED SUCCESS")
    except DelimiterNotFound as e:
        print(f"   DelimiterNotFound raised with max_bytes={e.max_bytes}")
    except anyio.IncompleteRead:
        print(f"   IncompleteRead raised - EXPECTED")


anyio.run(test_cases)