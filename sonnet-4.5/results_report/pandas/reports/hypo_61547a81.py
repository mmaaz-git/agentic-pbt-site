from hypothesis import given, strategies as st
import pandas.api.types as pat

@given(st.text(min_size=1, max_size=50))
def test_is_re_compilable_returns_bool(pattern):
    result = pat.is_re_compilable(pattern)
    assert isinstance(result, bool), \
        f"is_re_compilable should always return bool, got exception for {repr(pattern)}"

# Run the test
if __name__ == "__main__":
    test_is_re_compilable_returns_bool()