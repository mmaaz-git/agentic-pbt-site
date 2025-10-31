#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from unittest import mock
import time
from llm.utils import monotonic_ulid, NANOSECS_IN_MILLISECS

@given(st.integers(min_value=1, max_value=1000))
@settings(max_examples=10)
def test_monotonic_ulid_clock_backward(backward_ms):
    ulid1 = monotonic_ulid()

    current_time = time.time_ns()
    backward_time = current_time - (backward_ms * NANOSECS_IN_MILLISECS)

    with mock.patch('time.time_ns', return_value=backward_time):
        ulid2 = monotonic_ulid()

    try:
        assert ulid1 < ulid2, f"Monotonicity violated: {ulid1} >= {ulid2}"
        print(f"✓ Test passed with backward_ms={backward_ms}")
    except AssertionError as e:
        print(f"✗ Test failed with backward_ms={backward_ms}: {e}")
        raise

# Run the test
print("Running Hypothesis test...")
try:
    test_monotonic_ulid_clock_backward()
    print("\nAll tests passed!")
except AssertionError:
    print("\nTest failed - bug confirmed!")