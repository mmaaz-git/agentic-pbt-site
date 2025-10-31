from hypothesis import given, strategies as st, settings
import numpy as np
import tempfile
import scipy.io.wavfile
import os

@given(
    sample_rate=st.integers(min_value=1000, max_value=48000),
    n_samples=st.integers(min_value=1, max_value=5000),
    n_channels=st.integers(min_value=1, max_value=8),
    dtype=st.sampled_from([np.int16, np.int32, np.float32])
)
@settings(max_examples=200)
def test_wavfile_roundtrip_multichannel(sample_rate, n_samples, n_channels, dtype):
    if dtype in [np.int16, np.int32]:
        iinfo = np.iinfo(dtype)
        data = np.random.randint(iinfo.min, iinfo.max + 1, (n_samples, n_channels), dtype=dtype)
    else:
        data = (np.random.rand(n_samples, n_channels) * 2 - 1).astype(dtype)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name

    try:
        scipy.io.wavfile.write(temp_file, sample_rate, data)
        loaded_rate, loaded_data = scipy.io.wavfile.read(temp_file)

        assert loaded_data.shape == data.shape, f"Shape mismatch: {loaded_data.shape} != {data.shape}"
    finally:
        os.unlink(temp_file)

if __name__ == "__main__":
    test_wavfile_roundtrip_multichannel()