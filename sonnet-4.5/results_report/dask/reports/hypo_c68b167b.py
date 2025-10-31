from hypothesis import given, strategies as st, settings
from dask.utils import key_split


@given(st.binary())
@settings(max_examples=300)
def test_key_split_bytes(b):
    result = key_split(b)
    assert isinstance(result, str), f"key_split should return str for bytes, got {type(result)}"


if __name__ == "__main__":
    test_key_split_bytes()