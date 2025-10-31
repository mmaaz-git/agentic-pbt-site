from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list

@given(st.text())
@settings(max_examples=1000)
def test_parse_list_returns_list(s):
    result = parse_list(s)
    assert isinstance(result, list)

# Run the test
test_parse_list_returns_list()