import pytest
from hypothesis import given, strategies as st, settings
import tempfile
import anyio
from anyio.streams.file import FileReadStream
from anyio import EndOfStream


@given(content=st.binary(min_size=1, max_size=1000))
@settings(max_examples=100)
def test_file_read_stream_receive_zero_bytes(content):
    """
    Property: Calling receive(0) on a FileReadStream with remaining data should
    return an empty bytes object, not raise EndOfStream.
    """
    async def run_test():
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            stream = await FileReadStream.from_path(temp_path)
            result = await stream.receive(0)
            assert result == b''
            await stream.aclose()
        finally:
            import os
            os.unlink(temp_path)

    try:
        anyio.run(run_test)
    except EndOfStream:
        pytest.fail("receive(0) raised EndOfStream instead of returning b''")


async def demonstrate_bug():
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(b'Hello, World!')
        temp_path = f.name

    try:
        stream = await FileReadStream.from_path(temp_path)
        result = await stream.receive(0)
        print(f"Result: {result!r}")
    except anyio.EndOfStream:
        print("Bug: EndOfStream raised when receiving 0 bytes")
    finally:
        import os
        os.unlink(temp_path)


def run_single_test():
    """Run a single test case"""
    async def run_test():
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b'\x00')
            temp_path = f.name

        try:
            stream = await FileReadStream.from_path(temp_path)
            result = await stream.receive(0)
            assert result == b''
            await stream.aclose()
            return "Success: returned b''"
        except EndOfStream:
            return "EndOfStream raised"
        finally:
            import os
            os.unlink(temp_path)

    return anyio.run(run_test)


if __name__ == "__main__":
    # First run the simple demonstration
    print("=== Simple demonstration ===")
    anyio.run(demonstrate_bug)

    print("\n=== Single test with content b'\\x00' ===")
    result = run_single_test()
    print(f"Result: {result}")