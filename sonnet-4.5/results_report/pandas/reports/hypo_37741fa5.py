from hypothesis import given, strategies as st
from pandas.io.json._normalize import convert_to_line_delimits

@given(st.text(min_size=1).filter(lambda s: s[0] == '[' and s[-1] != ']'))
def test_convert_to_line_delimits_malformed_list(s):
    result = convert_to_line_delimits(s)
    assert result == s, f"Malformed JSON list {repr(s)} should be returned unchanged, got {repr(result)}"

if __name__ == "__main__":
    test_convert_to_line_delimits_malformed_list()