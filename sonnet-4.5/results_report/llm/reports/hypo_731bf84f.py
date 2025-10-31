import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from unittest import mock
import time
from llm.utils import monotonic_ulid, NANOSECS_IN_MILLISECS

@given(st.integers(min_value=1, max_value=1000))
def test_monotonic_ulid_clock_backward(backward_ms):
    """Test that monotonic_ulid maintains strict monotonicity even when clock goes backward."""
    # Get first ULID
    ulid1 = monotonic_ulid()

    # Simulate clock going backward
    current_time = time.time_ns()
    backward_time = current_time - (backward_ms * NANOSECS_IN_MILLISECS)

    # Get second ULID with backward clock
    with mock.patch('time.time_ns', return_value=backward_time):
        ulid2 = monotonic_ulid()

    # Assert strict monotonicity
    assert ulid1 < ulid2, f"Monotonicity violated: {ulid1} >= {ulid2} when clock went backward by {backward_ms}ms"

if __name__ == "__main__":
    test_monotonic_ulid_clock_backward()