from hypothesis import given, strategies as st, example
from pandas.io.excel._util import _excel2num


@given(st.text())
@example("")  # Explicitly test empty string
def test_excel2num_never_returns_negative(col_name):
    """Test that _excel2num never returns negative values for any input."""
    try:
        result = _excel2num(col_name)
        assert result >= 0, (
            f"_excel2num('{col_name}') returned {result}, "
            "but column indices should never be negative"
        )
    except ValueError:
        pass  # ValueError is expected for invalid inputs


if __name__ == "__main__":
    print("Running property-based test with Hypothesis...")
    try:
        test_excel2num_never_returns_negative()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")