#!/usr/bin/env python3
"""Property-based test for llm.cosine_similarity that discovers the zero vector bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
import llm

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1)
)
def test_cosine_similarity_handles_zero_vectors(a, b):
    assume(len(a) == len(b))
    result = llm.cosine_similarity(a, b)
    assert -1.0 <= result <= 1.0 or result != result  # NaN is acceptable

if __name__ == "__main__":
    # Run the test
    test_cosine_similarity_handles_zero_vectors()