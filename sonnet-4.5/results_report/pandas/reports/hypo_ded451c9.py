from hypothesis import given, strategies as st, settings
import pandas.api.types as pat


@given(st.text())
@settings(max_examples=500)
def test_is_re_compilable_consistency(pattern):
    is_compilable = pat.is_re_compilable(pattern)

    if is_compilable:
        import re
        try:
            re.compile(pattern)
        except:
            assert False, f"is_re_compilable returned True but re.compile failed"


if __name__ == "__main__":
    test_is_re_compilable_consistency()