from hypothesis import given, strategies as st
import pandas.api.types as pat

@given(st.text(min_size=1, max_size=10))
def test_is_re_compilable_returns_bool(s):
    """is_re_compilable should always return a bool, never raise exceptions"""
    result = pat.is_re_compilable(s)
    assert isinstance(result, bool), f"is_re_compilable should return bool"

if __name__ == "__main__":
    # Run the property-based test
    test_is_re_compilable_returns_bool()