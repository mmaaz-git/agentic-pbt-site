from hypothesis import given, strategies as st
from pandas.api import types
import re


@given(st.text(min_size=1, max_size=100))
def test_is_re_compilable_should_not_raise(pattern_str):
    result = types.is_re_compilable(pattern_str)

    assert isinstance(result, bool)

    if result:
        re.compile(pattern_str)


# Run the test
if __name__ == "__main__":
    test_is_re_compilable_should_not_raise()