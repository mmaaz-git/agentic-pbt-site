from hypothesis import given, strategies as st
from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

@given(st.integers(min_value=0, max_value=1000000))
def test_is_valid_tag_rejects_dot_decimal_strings(num):
    name = f".{num}"
    result = is_valid_tag(name)
    assert result == False, f"is_valid_tag('{name}') should return False but returned {result}"

if __name__ == "__main__":
    # Run the test
    test_is_valid_tag_rejects_dot_decimal_strings()