from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list

@given(st.text())
@settings(max_examples=1000)
def test_parse_list_no_empty_strings(s):
    result = parse_list(s)
    assert all(item != '' for item in result), f"parse_list returned empty string in result: {result}"

if __name__ == "__main__":
    test_parse_list_no_empty_strings()