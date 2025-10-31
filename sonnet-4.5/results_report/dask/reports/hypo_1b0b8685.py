from hypothesis import given, strategies as st
from dask.utils import key_split

@given(st.binary())
def test_key_split_bytes(b):
    result = key_split(b)
    assert isinstance(result, str)

if __name__ == "__main__":
    test_key_split_bytes()