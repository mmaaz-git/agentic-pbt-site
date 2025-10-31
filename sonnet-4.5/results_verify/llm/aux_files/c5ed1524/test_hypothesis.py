#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(
    st.text(min_size=1),
    st.integers(min_value=1, max_value=1000),
    st.booleans(),
    st.booleans(),
)
def test_truncate_string_length_invariant(text, max_length, normalize_whitespace, keep_end):
    result = truncate_string(text, max_length, normalize_whitespace, keep_end)
    assert len(result) <= max_length, \
        f"Length invariant violated: len({repr(result)}) = {len(result)} > {max_length}"

# Run the test
if __name__ == "__main__":
    test_truncate_string_length_invariant()