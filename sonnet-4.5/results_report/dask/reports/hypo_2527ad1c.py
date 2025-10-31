from hypothesis import given, strategies as st, settings
from dask.bytes.core import read_bytes
import tempfile
import os


@given(st.integers(min_value=0, max_value=1000))
@settings(max_examples=200)
def test_sample_return_type_with_integer(sample_size):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'test.txt')

        test_data = b'x' * 1000
        with open(test_file, 'wb') as f:
            f.write(test_data)

        sample, blocks = read_bytes(test_file, sample=sample_size, blocksize=None)

        assert isinstance(sample, bytes), \
            f"sample={sample_size} should return bytes, got {type(sample).__name__}"


if __name__ == "__main__":
    test_sample_return_type_with_integer()