from hypothesis import given, strategies as st
import llm.utils

@given(
    st.text(min_size=1, max_size=1000),
    st.integers(min_value=1, max_value=500)
)
def test_truncate_string_length_property(text, max_length):
    result = llm.utils.truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, f"len('{result}') = {len(result)} > max_length={max_length}"

if __name__ == "__main__":
    test_truncate_string_length_property()