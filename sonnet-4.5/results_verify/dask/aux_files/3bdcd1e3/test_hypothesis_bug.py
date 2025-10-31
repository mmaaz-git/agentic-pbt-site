from hypothesis import given, strategies as st, settings
from dask.widgets import FILTERS


@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_output_length(n):
    format_bytes = FILTERS['format_bytes']
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"

# Run the test
if __name__ == "__main__":
    try:
        test_format_bytes_output_length()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")