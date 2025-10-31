import math
from hypothesis import given, strategies as st
import anyio


@given(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False).filter(lambda x: x > 0 and not x.is_integer()))
def test_memory_stream_accepts_float_buffer_size(max_buffer_size):
    print(f"Testing with max_buffer_size: {max_buffer_size}")
    try:
        send, recv = anyio.create_memory_object_stream(max_buffer_size)
        print(f"Success with {max_buffer_size}")
    except ValueError as e:
        print(f"ValueError with {max_buffer_size}: {e}")
        raise

if __name__ == "__main__":
    test_memory_stream_accepts_float_buffer_size()