from hypothesis import given, strategies as st, settings
import pandas.api.types as pat

@given(st.text(min_size=1, max_size=50))
@settings(max_examples=100)
def test_is_re_compilable_returns_bool(pattern):
    try:
        result = pat.is_re_compilable(pattern)
        assert isinstance(result, bool), \
            f"is_re_compilable should always return bool, got {type(result)} for {repr(pattern)}"
    except Exception as e:
        print(f"FAILED: Exception raised for pattern {repr(pattern)}: {type(e).__name__}: {e}")
        raise

# Run the test
print("Running hypothesis test...")
test_is_re_compilable_returns_bool()
print("Test passed!")
