from pandas.io.json._normalize import convert_to_line_delimits
from hypothesis import given, strategies as st, assume, settings

@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=100))
@settings(max_examples=100, deadline=None)
def test_convert_to_line_delimits_non_list_unchanged(json_str):
    assume(not (json_str.startswith('[') and json_str.endswith(']')))
    result = convert_to_line_delimits(json_str)
    assert result == json_str

if __name__ == "__main__":
    test_convert_to_line_delimits_non_list_unchanged()