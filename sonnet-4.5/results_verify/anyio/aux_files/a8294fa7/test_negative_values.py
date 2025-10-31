import tempfile
import anyio
from anyio.streams.file import FileReadStream


async def test_various_negative_values():
    """Test different negative values to understand the behavior"""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        test_data = b'x' * 100000  # 100KB
        f.write(test_data)
        f.flush()
        path = f.name

    try:
        negative_values = [-1, -10, -100, -65536, -999999]

        for neg_val in negative_values:
            try:
                stream = await FileReadStream.from_path(path)
                data = await stream.receive(max_bytes=neg_val)
                print(f"max_bytes={neg_val}: Received {len(data)} bytes (entire file)")
                await stream.aclose()
            except Exception as e:
                print(f"max_bytes={neg_val}: Raised {type(e).__name__}: {e}")

    finally:
        import os
        os.unlink(path)


async def test_python_file_read_behavior():
    """Test Python's file.read() behavior directly"""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        test_data = b'x' * 100000  # 100KB
        f.write(test_data)
        f.flush()
        path = f.name

    try:
        # Test Python's file.read() directly
        with open(path, 'rb') as f:
            data = f.read(-1)
            print(f"Python file.read(-1): {len(data)} bytes")

        with open(path, 'rb') as f:
            data = f.read(-100)
            print(f"Python file.read(-100): {len(data)} bytes")

        with open(path, 'rb') as f:
            data = f.read(0)
            print(f"Python file.read(0): {len(data)} bytes")

    finally:
        import os
        os.unlink(path)


if __name__ == "__main__":
    print("=== Testing various negative values ===")
    anyio.run(test_various_negative_values)

    print("\n=== Testing Python's file.read() behavior ===")
    anyio.run(test_python_file_read_behavior)