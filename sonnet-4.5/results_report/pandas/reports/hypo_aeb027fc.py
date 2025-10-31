from hypothesis import given, strategies as st, settings, example
from pandas.io.excel._util import _excel2num

@given(st.text())
@example('')  # Force testing with empty string
@example('   ')  # Force testing with whitespace
@settings(max_examples=100)
def test_excel2num_valid_or_error(col_name):
    """Test that _excel2num either returns a valid index or raises ValueError"""
    try:
        result = _excel2num(col_name)
        assert result >= 0, f"Column index must be non-negative, got {result} for input '{col_name}'"
    except ValueError:
        # ValueError is acceptable for invalid inputs
        pass

if __name__ == "__main__":
    test_excel2num_valid_or_error()