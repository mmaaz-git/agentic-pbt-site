from hypothesis import given, strategies as st
from dask.widgets import FILTERS


@given(st.binary(min_size=1))
def test_key_split_bytes_returns_string(b):
    key_split = FILTERS['key_split']
    result = key_split(b)
    assert isinstance(result, str)

# Run the test
if __name__ == "__main__":
    test_key_split_bytes_returns_string()