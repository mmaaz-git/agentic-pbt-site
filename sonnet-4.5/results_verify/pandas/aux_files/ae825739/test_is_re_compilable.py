from hypothesis import given, strategies as st
import pandas.api.types as pt
import re


@given(st.text())
def test_is_re_compilable_consistency_with_re_compile(pattern):
    is_compilable = pt.is_re_compilable(pattern)
    try:
        re.compile(pattern)
        can_compile = True
    except re.error:
        can_compile = False
    assert is_compilable == can_compile

# Run the test
if __name__ == "__main__":
    test_is_re_compilable_consistency_with_re_compile()