from llm.utils import truncate_string
from hypothesis import given, strategies as st

@given(
    st.text(min_size=1, max_size=100),
    st.integers(min_value=0, max_value=10)
)
def test_truncate_string_length_invariant(text, max_length):
    result = truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, \
        f"Result length {len(result)} exceeds max_length {max_length}: '{result}'"

# Run the test
if __name__ == "__main__":
    test_truncate_string_length_invariant()