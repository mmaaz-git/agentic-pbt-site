import tempfile
import pytest
import anyio
from anyio.streams.file import FileReadStream


@pytest.mark.anyio
async def test_receive_negative_max_bytes():
    """
    Property: max_bytes should constrain the amount of data read.
    Negative values should either be rejected or treated as a sensible default,
    not read the entire file.
    """
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        test_data = b'x' * 1000000  # 1MB
        f.write(test_data)
        f.flush()
        path = f.name

    try:
        stream = await FileReadStream.from_path(path)
        data = await stream.receive(max_bytes=-1)
        print(f"Test: Received {len(data)} bytes with max_bytes=-1")
        assert len(data) == 1000000  # Entire file was read!
        await stream.aclose()
    finally:
        import os
        os.unlink(path)


async def reproduce_bug():
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        large_data = b'x' * 10_000_000  # 10MB
        f.write(large_data)
        f.flush()
        path = f.name

    try:
        stream = await FileReadStream.from_path(path)

        data = await stream.receive(max_bytes=-1)

        print(f"Requested: max_bytes=-1")
        print(f"Received: {len(data)} bytes (entire file!)")
        print(f"Expected: Should error or read a bounded amount")

        await stream.aclose()
    finally:
        import os
        os.unlink(path)


async def test_max_bytes_zero():
    """Test what happens with max_bytes=0"""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        test_data = b'x' * 1000
        f.write(test_data)
        f.flush()
        path = f.name

    try:
        stream = await FileReadStream.from_path(path)
        try:
            data = await stream.receive(max_bytes=0)
            print(f"Test: Received {len(data)} bytes with max_bytes=0")
        except Exception as e:
            print(f"Test: max_bytes=0 raised {type(e).__name__}: {e}")
        await stream.aclose()
    finally:
        import os
        os.unlink(path)


async def test_normal_behavior():
    """Test normal behavior with positive max_bytes"""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        test_data = b'x' * 100000  # 100KB
        f.write(test_data)
        f.flush()
        path = f.name

    try:
        stream = await FileReadStream.from_path(path)

        # Test with default
        data1 = await stream.receive()
        print(f"Test: Default receive() got {len(data1)} bytes")

        # Reset position
        await stream.seek(0)

        # Test with explicit positive value
        data2 = await stream.receive(max_bytes=1024)
        print(f"Test: receive(1024) got {len(data2)} bytes")

        await stream.aclose()
    finally:
        import os
        os.unlink(path)


if __name__ == "__main__":
    print("=== Testing normal behavior ===")
    anyio.run(test_normal_behavior)

    print("\n=== Reproducing the bug ===")
    anyio.run(reproduce_bug)

    print("\n=== Testing the hypothesis test ===")
    anyio.run(test_receive_negative_max_bytes)

    print("\n=== Testing max_bytes=0 ===")
    anyio.run(test_max_bytes_zero)