from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list


@given(st.text())
@settings(max_examples=1000)
def test_parse_list_should_not_crash(s):
    """Test that parse_list does not crash on any input string"""
    try:
        result = parse_list(s)
        print(f"Success on input: {repr(s)} -> {result}")
    except Exception as e:
        print(f"FAILED on input: {repr(s)} with error: {e}")
        raise

if __name__ == "__main__":
    test_parse_list_should_not_crash()