from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes

@settings(max_examples=1000)
@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_max_length_10(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)}, expected <= 10"

test_format_bytes_max_length_10()