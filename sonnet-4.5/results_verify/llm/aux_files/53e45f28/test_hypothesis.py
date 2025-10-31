#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(st.text(), st.integers(min_value=1, max_value=1000))
def test_truncate_string_length(text, max_length):
    result = truncate_string(text, max_length)
    assert len(result) <= max_length, f"Result '{result}' has length {len(result)} but max_length is {max_length}"

# Run the test
if __name__ == "__main__":
    test_truncate_string_length()
    print("Test completed")