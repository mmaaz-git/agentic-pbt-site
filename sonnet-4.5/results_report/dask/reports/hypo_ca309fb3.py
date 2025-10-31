from hypothesis import given, strategies as st, settings
import dask.utils
import pytest


@given(st.just(''))
@settings(max_examples=1)
def test_parse_timedelta_empty_string(s):
    """Test that parse_timedelta raises ValueError on empty string input."""
    with pytest.raises(ValueError):
        dask.utils.parse_timedelta(s)

if __name__ == "__main__":
    # Run the test
    test_parse_timedelta_empty_string()