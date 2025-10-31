from hypothesis import given, strategies as st, settings
from pandas.api import types as pat
import re

@given(st.text(min_size=1))
@settings(max_examples=100)
def test_is_re_compilable_returns_bool(pattern):
    try:
        re.compile(pattern)
        expected = True
    except re.error:
        expected = False

    result = pat.is_re_compilable(pattern)
    assert isinstance(result, bool), f"is_re_compilable should return bool, not raise exception"
    assert result == expected, f"is_re_compilable mismatch for pattern: {pattern!r}"

if __name__ == "__main__":
    test_is_re_compilable_returns_bool()