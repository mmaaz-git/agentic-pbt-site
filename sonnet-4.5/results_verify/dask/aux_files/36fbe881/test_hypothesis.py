from hypothesis import given, strategies as st
from dask.widgets.widgets import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_length_under_2_60(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = {result!r} has length {len(result)} > 10"

if __name__ == "__main__":
    # Run the test
    test_format_bytes_length_under_2_60()