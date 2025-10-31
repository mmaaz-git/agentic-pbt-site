import re
from hypothesis import given, strategies as st, settings
import pandas.api.types as pat


@given(st.text())
@settings(max_examples=500)
def test_is_re_compilable_should_not_crash(s):
    try:
        result = pat.is_re_compilable(s)
        assert isinstance(result, bool), f"Should return bool, got {type(result)}"
    except Exception as e:
        raise AssertionError(
            f"is_re_compilable crashed on input {s!r} with {type(e).__name__}: {e}"
        )


if __name__ == "__main__":
    # Run the test
    test_is_re_compilable_should_not_crash()