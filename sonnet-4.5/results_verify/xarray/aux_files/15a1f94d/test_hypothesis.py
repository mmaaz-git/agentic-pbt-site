#!/usr/bin/env python3
"""Run the property-based test from the bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from xarray.backends.chunks import build_grid_chunks

@given(
    size=st.integers(min_value=1, max_value=1000),
    chunk_size=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=500)
def test_build_grid_chunks_sum_invariant(size, chunk_size):
    chunks = build_grid_chunks(size, chunk_size)
    assert sum(chunks) == size, f"Sum of chunks {sum(chunks)} != size {size}. Chunks: {chunks}, chunk_size={chunk_size}"

if __name__ == "__main__":
    try:
        test_build_grid_chunks_sum_invariant()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")