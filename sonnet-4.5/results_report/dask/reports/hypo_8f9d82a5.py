from hypothesis import given, strategies as st
from dask.utils import format_bytes


@given(st.integers(min_value=1, max_value=2**60 - 1))
def test_format_bytes_output_length(n):
    """Test that format_bytes output is always <= 10 characters for values < 2**60"""
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) returned {result!r} with length {len(result)}"


if __name__ == "__main__":
    # Run the test
    test_format_bytes_output_length()