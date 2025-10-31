#!/usr/bin/env python3
"""Property-based test for format_bytes output length invariant"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

from hypothesis import given, strategies as st
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_output_length_invariant(n):
    """Test that format_bytes output is always <= 10 characters for values < 2^60"""
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"

if __name__ == "__main__":
    # Run the test
    test_format_bytes_output_length_invariant()