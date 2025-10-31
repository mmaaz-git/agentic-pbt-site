from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list


@settings(max_examples=500)
@given(st.text())
def test_parse_list_returns_list(s):
    result = parse_list(s)
    assert isinstance(result, list), f"parse_list should return a list, got {type(result)}"


if __name__ == "__main__":
    test_parse_list_returns_list()