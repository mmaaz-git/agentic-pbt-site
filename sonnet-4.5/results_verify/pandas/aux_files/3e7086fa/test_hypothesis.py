from hypothesis import given, strategies as st
import pandas.api.types as pat
import traceback

@given(st.text(min_size=1, max_size=10))
def test_is_re_compilable_returns_bool(s):
    """is_re_compilable should always return a bool, never raise exceptions"""
    try:
        result = pat.is_re_compilable(s)
        assert isinstance(result, bool), f"is_re_compilable should return bool, got {type(result)}"
    except Exception as e:
        print(f"ERROR: is_re_compilable raised an exception for input '{s}'")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        traceback.print_exc()
        raise

# Run the test
if __name__ == "__main__":
    test_is_re_compilable_returns_bool()
    print("Test passed for all generated inputs")