import re
from hypothesis import given, strategies as st, settings
from pandas.core.dtypes.inference import is_re_compilable

@given(text=st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=20))
@settings(max_examples=500)
def test_is_re_compilable_should_not_crash(text):
    try:
        result = is_re_compilable(text)
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
    except re.PatternError as e:
        raise AssertionError(f"is_re_compilable should not raise PatternError for invalid patterns, but got: {e}")

if __name__ == "__main__":
    print("Running Hypothesis property-based test...")
    try:
        test_is_re_compilable_should_not_crash()
        print("Test PASSED - no crashes detected in 500 examples")
    except AssertionError as e:
        print(f"Test FAILED: {e}")