import sys
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages")

from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(st.text(), st.integers(min_value=1, max_value=1000))
def test_truncate_string_length_bound(text, max_length):
    result = truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, f"Result '{result}' (len={len(result)}) exceeds max_length={max_length}"

# Run the property-based test
test_truncate_string_length_bound()