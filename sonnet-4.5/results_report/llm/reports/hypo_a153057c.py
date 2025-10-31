import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(
    st.text(min_size=1, max_size=1000),
    st.integers(min_value=1, max_value=500)
)
def test_truncate_string_length_constraint(text, max_length):
    result = truncate_string(text, max_length)
    assert len(result) <= max_length, f"Result length {len(result)} > max_length {max_length}"

if __name__ == "__main__":
    test_truncate_string_length_constraint()