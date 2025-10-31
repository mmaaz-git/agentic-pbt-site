#!/usr/bin/env python3
"""Hypothesis test from the bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env')

from anyio import create_memory_object_stream
from hypothesis import given, strategies as st

@given(max_buffer=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
def test_create_memory_object_stream_accepts_floats(max_buffer):
    send, receive = create_memory_object_stream(max_buffer_size=max_buffer)

# Run the test
if __name__ == "__main__":
    test_create_memory_object_stream_accepts_floats()