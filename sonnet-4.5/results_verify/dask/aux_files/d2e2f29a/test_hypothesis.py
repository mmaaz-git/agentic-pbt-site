import tempfile
from pathlib import Path

from hypothesis import given, settings, strategies as st

from dask.bytes.core import read_bytes


@given(
    st.binary(min_size=1, max_size=100),
    st.binary(min_size=1, max_size=100),
    st.integers(min_value=50, max_value=200)
)
@settings(max_examples=500)
def test_delimiter_sampling_always_ends_with_delimiter(data_before, data_after, sample_size):
    delimiter = b'\n'

    if delimiter in data_before or delimiter in data_after:
        return

    full_data = data_before + delimiter + data_after

    with tempfile.NamedTemporaryFile(delete=False, mode='wb') as f:
        f.write(full_data)
        f.flush()
        temp_path = f.name

    try:
        sample, blocks = read_bytes(
            temp_path,
            blocksize=None,
            delimiter=delimiter,
            sample=sample_size
        )

        assert sample.endswith(delimiter), \
            f"Sample must end with delimiter. Got: {sample!r}"

    finally:
        Path(temp_path).unlink()

if __name__ == "__main__":
    # Run the test
    test_delimiter_sampling_always_ends_with_delimiter()
    print("Test completed")