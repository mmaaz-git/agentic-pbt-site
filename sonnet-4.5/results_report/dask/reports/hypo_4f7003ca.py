from io import BytesIO
from hypothesis import given, strategies as st, assume
from fsspec.utils import read_block

@given(st.binary(min_size=1, max_size=1000), st.integers(min_value=0, max_value=100))
def test_read_block_length_none_reads_to_end(data, offset):
    assume(offset < len(data))
    f = BytesIO(data)
    result = read_block(f, offset, None, delimiter=None)
    expected = data[offset:]
    assert result == expected

if __name__ == "__main__":
    test_read_block_length_none_reads_to_end()