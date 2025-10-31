from hypothesis import given, strategies as st
from Cython.Debugger.DebugWriter import is_valid_tag


@given(st.integers(min_value=0, max_value=999999))
def test_is_valid_tag_decimal_pattern(n):
    name = f".{n}"
    assert is_valid_tag(name) is False


if __name__ == "__main__":
    test_is_valid_tag_decimal_pattern()