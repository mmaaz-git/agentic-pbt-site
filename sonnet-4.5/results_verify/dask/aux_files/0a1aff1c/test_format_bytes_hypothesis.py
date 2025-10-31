from hypothesis import given, strategies as st, settings
import dask.utils

@given(st.integers(min_value=0, max_value=2**60))
@settings(max_examples=500)
def test_format_bytes_length_claim(n):
    formatted = dask.utils.format_bytes(n)
    print(f"Testing n={n}: formatted='{formatted}', length={len(formatted)}")
    assert len(formatted) <= 10, f"Failed for n={n}: formatted='{formatted}' has length {len(formatted)}"

if __name__ == "__main__":
    test_format_bytes_length_claim()