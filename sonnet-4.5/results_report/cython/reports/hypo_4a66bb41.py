from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list


@given(st.text())
@settings(max_examples=1000)
def test_parse_list_should_not_crash(s):
    result = parse_list(s)


if __name__ == "__main__":
    test_parse_list_should_not_crash()