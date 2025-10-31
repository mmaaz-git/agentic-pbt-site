from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(st.text(), st.integers(min_value=1, max_value=1000))
def test_truncate_string_length(text, max_length):
    result = truncate_string(text, max_length)
    assert len(result) <= max_length, f"Result length {len(result)} exceeds max_length {max_length} for text='{text}'"

# Run the test
if __name__ == "__main__":
    test_truncate_string_length()