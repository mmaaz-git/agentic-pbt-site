import pandas as pd
from hypothesis import given, strategies as st, settings


@settings(max_examples=100)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=5, max_size=20),
    st.text(min_size=1, max_size=10).filter(lambda s: s not in ["single", "table"])
)
def test_method_validation_error_message(data, method):
    """
    Property: When an invalid method is provided, the error message should
    be well-formed with properly matched quotes.
    """
    df = pd.DataFrame({'A': data})

    try:
        df.rolling(window=2, method=method)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        # Error message should have matched quotes
        assert error_msg.count("'") % 2 == 0, f"Unmatched quotes in error message: {error_msg}"


if __name__ == "__main__":
    test_method_validation_error_message()