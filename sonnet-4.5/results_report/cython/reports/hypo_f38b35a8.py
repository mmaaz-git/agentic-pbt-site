from hypothesis import given, strategies as st, settings
import pytest
from Cython.Plex.Regexps import RawCodeRange

@given(st.integers(min_value=0, max_value=200),
       st.integers(min_value=0, max_value=200))
@settings(max_examples=300)
def test_rawcoderange_str_method(code1, code2):
    if code1 >= code2:
        return

    rcr = RawCodeRange(code1, code2)

    try:
        str_repr = str(rcr)
        assert str_repr is not None
    except AttributeError as e:
        if 'code1' in str(e) or 'code2' in str(e):
            pytest.fail(f"RawCodeRange.calc_str references non-existent attributes: {e}")

if __name__ == "__main__":
    test_rawcoderange_str_method()