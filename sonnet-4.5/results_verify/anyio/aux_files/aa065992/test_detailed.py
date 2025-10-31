import tempfile
import anyio
from anyio.streams.file import FileReadStream
import os


async def test_various_scenarios():
    """Test various scenarios with FileReadStream.receive"""

    # Create a test file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(b'Hello, World!')
        temp_path = f.name

    try:
        print("Test 1: receive(0) at start of file")
        stream = await FileReadStream.from_path(temp_path)
        try:
            result = await stream.receive(0)
            print(f"  Result: {result!r}")
        except anyio.EndOfStream:
            print("  EndOfStream raised")
        await stream.aclose()

        print("\nTest 2: receive(5) then receive(0)")
        stream = await FileReadStream.from_path(temp_path)
        data1 = await stream.receive(5)
        print(f"  First receive(5): {data1!r}")
        try:
            result = await stream.receive(0)
            print(f"  Then receive(0): {result!r}")
        except anyio.EndOfStream:
            print("  Then receive(0): EndOfStream raised")
        await stream.aclose()

        print("\nTest 3: Read all data, then receive(0)")
        stream = await FileReadStream.from_path(temp_path)
        all_data = await stream.receive(100)
        print(f"  receive(100): {all_data!r}")
        try:
            result = await stream.receive(0)
            print(f"  Then receive(0): {result!r}")
        except anyio.EndOfStream:
            print("  Then receive(0): EndOfStream raised")
        await stream.aclose()

        print("\nTest 4: receive(1) at EOF (should raise EndOfStream)")
        stream = await FileReadStream.from_path(temp_path)
        await stream.receive(100)  # Read all
        try:
            result = await stream.receive(1)
            print(f"  receive(1) at EOF: {result!r}")
        except anyio.EndOfStream:
            print("  receive(1) at EOF: EndOfStream raised (expected)")
        await stream.aclose()

        # Test with empty file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            empty_path = f.name

        print("\nTest 5: receive(0) on empty file")
        stream = await FileReadStream.from_path(empty_path)
        try:
            result = await stream.receive(0)
            print(f"  Result: {result!r}")
        except anyio.EndOfStream:
            print("  EndOfStream raised")
        await stream.aclose()

        print("\nTest 6: receive(1) on empty file")
        stream = await FileReadStream.from_path(empty_path)
        try:
            result = await stream.receive(1)
            print(f"  Result: {result!r}")
        except anyio.EndOfStream:
            print("  EndOfStream raised (expected)")
        await stream.aclose()

        os.unlink(empty_path)

    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    anyio.run(test_various_scenarios)