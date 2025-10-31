from hypothesis import given, strategies as st, settings, assume
import tempfile
import os
from dask.bytes.core import read_bytes

@given(
    file_size=st.integers(min_value=1000, max_value=50000),
    sample_size=st.integers(min_value=100, max_value=500)
)
@settings(max_examples=50)
def test_sample_with_delimiter_reads_entire_file_if_no_delimiter(file_size, sample_size):
    content = b'x' * file_size
    assume(sample_size < file_size)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(content)
        f.flush()
        temp_path = f.name

    try:
        sample, blocks = read_bytes(temp_path, blocksize=None, sample=sample_size, delimiter=b'\n')

        print(f"File size: {file_size}, Sample requested: {sample_size}, Actual sample: {len(sample)}")
        assert len(sample) <= sample_size * 2, \
            f"Sample exceeded reasonable size: requested {sample_size}, got {len(sample)}"
    finally:
        os.unlink(temp_path)

# Run the test
if __name__ == "__main__":
    test_sample_with_delimiter_reads_entire_file_if_no_delimiter()