from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes


@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_length_claim(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = {result!r} has length {len(result)}, expected <= 10"

if __name__ == "__main__":
    test_format_bytes_length_claim()