from hypothesis import given, strategies as st
from pandas.core.dtypes import inference
import re

@given(st.text(min_size=0, max_size=100))
def test_is_re_compilable_on_regex_patterns(pattern):
    try:
        re.compile(pattern)
        result = inference.is_re_compilable(pattern)
        assert result is True
    except re.error:
        result = inference.is_re_compilable(pattern)
        assert result is False

# Run the test
if __name__ == "__main__":
    test_is_re_compilable_on_regex_patterns()