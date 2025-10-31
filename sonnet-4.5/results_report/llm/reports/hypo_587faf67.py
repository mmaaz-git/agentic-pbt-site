from hypothesis import given, settings, strategies as st
from llm.utils import truncate_string

@settings(max_examples=500)
@given(st.text(min_size=1), st.integers(min_value=0, max_value=100))
def test_truncate_string_max_length_property(text, max_length):
    result = truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, (
        f"truncate_string violated max_length constraint: "
        f"len({repr(result)}) = {len(result)} > {max_length}"
    )

# Run the test
test_truncate_string_max_length_property()