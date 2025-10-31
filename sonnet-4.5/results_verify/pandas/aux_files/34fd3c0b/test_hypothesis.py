import pandas.api.types as pat
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=1, max_size=100))
@settings(max_examples=200)
def test_is_re_compilable_on_strings(s):
    result = pat.is_re_compilable(s)
    assert isinstance(result, bool), f"is_re_compilable should return bool"

if __name__ == "__main__":
    test_is_re_compilable_on_strings()