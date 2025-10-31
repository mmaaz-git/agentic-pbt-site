from hypothesis import given, strategies as st
from pandas.io.excel._util import _excel2num


@given(st.text())
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