from hypothesis import given, strategies as st
from Cython.Build.Dependencies import parse_list

@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=0, max_size=100))
def test_parse_list_no_crash(s):
    result = parse_list(s)
    assert isinstance(result, list)

# Run the test to get the minimal failing example
if __name__ == "__main__":
    test_parse_list_no_crash()