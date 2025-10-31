import anyio
import math
import pytest
from hypothesis import given, strategies as st, settings, assume
from anyio.streams.memory import (
    MemoryObjectSendStream,
    MemoryObjectReceiveStream,
    MemoryObjectStreamState,
)
from anyio import WouldBlock


@given(st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
@settings(max_examples=10)  # Reduced for testing
def test_buffer_size_capacity_property(buffer_size):
    assume(buffer_size != math.floor(buffer_size))

    print(f"Testing buffer_size={buffer_size}, floor={math.floor(buffer_size)}, ceil={math.ceil(buffer_size)}")

    async def test_capacity():
        state = MemoryObjectStreamState[int](max_buffer_size=buffer_size)
        send_stream = MemoryObjectSendStream(state)
        receive_stream = MemoryObjectReceiveStream(state)

        expected_capacity = math.floor(buffer_size)

        # Add items up to the expected capacity
        for i in range(expected_capacity):
            send_stream.send_nowait(i)

        # This should raise WouldBlock according to the bug report's expectation
        try:
            send_stream.send_nowait(999)
            print(f"  ERROR: Added item {expected_capacity+1} when only {expected_capacity} expected")
            # The test expectation is wrong - it accepts ceil items, not floor
            raise AssertionError(f"Expected WouldBlock but item was added at position {expected_capacity+1}")
        except WouldBlock:
            print(f"  Got WouldBlock at expected position {expected_capacity}")

        send_stream.close()
        receive_stream.close()

    anyio.run(test_capacity)

# Run the test
test_buffer_size_capacity_property()