import pandas.api.types as pat
import re
from hypothesis import given, strategies as st


@given(st.text())
def test_is_re_compilable_should_not_raise(s):
    try:
        re.compile(s)
        can_compile = True
    except Exception:
        can_compile = False

    try:
        result = pat.is_re_compilable(s)
        assert result == can_compile, (
            f"is_re_compilable should match re.compile behavior without raising, "
            f"but got different behavior for {s!r}"
        )
    except Exception as e:
        print(f"FAILED: is_re_compilable raised {type(e).__name__} for input: {s!r}")
        print(f"Error: {e}")
        raise


# Run the test
if __name__ == "__main__":
    print("Running Hypothesis test...")
    try:
        test_is_re_compilable_should_not_raise()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")