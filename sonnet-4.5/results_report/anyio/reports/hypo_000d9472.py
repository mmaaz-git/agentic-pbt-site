from anyio import create_memory_object_stream
from hypothesis import given, strategies as st


@given(max_buffer=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
def test_create_memory_object_stream_accepts_floats(max_buffer):
    send, receive = create_memory_object_stream(max_buffer_size=max_buffer)


# Run the test
test_create_memory_object_stream_accepts_floats()