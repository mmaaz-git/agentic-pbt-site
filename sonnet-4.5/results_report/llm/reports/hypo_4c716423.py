#!/usr/bin/env python3
"""Hypothesis property-based test for truncate_string"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from llm.utils import truncate_string

@given(st.text(min_size=1), st.integers(min_value=1, max_value=1000))
@settings(max_examples=1000)
def test_truncate_string_length_invariant(text, max_length):
    """Test that truncate_string respects max_length contract"""
    result = truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, f"Result '{result}' (len={len(result)}) exceeds max_length={max_length}"

if __name__ == "__main__":
    test_truncate_string_length_invariant()