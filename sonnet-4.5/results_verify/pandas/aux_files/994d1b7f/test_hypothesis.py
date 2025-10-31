from hypothesis import given, strategies as st
import pandas.util


@given(st.text())
def test_capitalize_preserves_length(s):
    result = pandas.util.capitalize_first_letter(s)
    assert len(result) == len(s), f"Length changed: '{s}' (len={len(s)}) -> '{result}' (len={len(result)})"

# Run the test
if __name__ == "__main__":
    test_capitalize_preserves_length()