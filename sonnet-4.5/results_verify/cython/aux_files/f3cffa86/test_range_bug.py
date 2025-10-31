from hypothesis import given, settings, strategies as st
import pytest
from Cython.Plex import Range


@settings(max_examples=100)
@given(st.text(min_size=1, max_size=20))
def test_range_validates_input_properly(s):
    if len(s) % 2 == 0:
        re = Range(s)
        assert hasattr(re, 'nullable')
    else:
        with pytest.raises(ValueError, match="even length"):
            Range(s)

if __name__ == "__main__":
    test_range_validates_input_properly()