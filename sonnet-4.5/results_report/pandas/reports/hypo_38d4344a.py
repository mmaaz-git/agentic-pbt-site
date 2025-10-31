import pandas.api.types as types
import re
from hypothesis import given, strategies as st


@given(st.text(min_size=1))
def test_is_re_compilable_for_valid_patterns(pattern):
    try:
        re.compile(pattern)
        can_compile = True
    except re.error:
        can_compile = False

    result = types.is_re_compilable(pattern)
    assert result == can_compile

if __name__ == "__main__":
    test_is_re_compilable_for_valid_patterns()