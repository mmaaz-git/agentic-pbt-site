from hypothesis import given, strategies as st, settings
from scipy.io import wavfile
import tempfile
import numpy as np
import os

@given(
    st.integers(min_value=8000, max_value=48000),
    st.integers(min_value=10, max_value=500),
    st.integers(min_value=1, max_value=8)
)
@settings(max_examples=20)
def test_wavfile_multichannel(rate, n_samples, n_channels):
    data = np.random.randint(-10000, 10000, size=(n_samples, n_channels), dtype=np.int16)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        filename = f.name

    try:
        wavfile.write(filename, rate, data)
        result_rate, result_data = wavfile.read(filename)

        print(f"Test case: rate={rate}, n_samples={n_samples}, n_channels={n_channels}")
        print(f"  Input shape: {data.shape}")
        print(f"  Output shape: {result_data.shape}")

        assert result_rate == rate, f"Rate mismatch: {result_rate} != {rate}"
        assert result_data.shape == data.shape, f"Shape mismatch: {result_data.shape} != {data.shape}"

        # For single channel, need to reshape for comparison
        if n_channels == 1 and result_data.ndim == 1:
            assert np.array_equal(result_data, data.flatten()), "Data mismatch"
        else:
            assert np.array_equal(result_data, data), "Data mismatch"

        print(f"  PASS")
    except AssertionError as e:
        print(f"  FAIL: {e}")
        raise
    finally:
        if os.path.exists(filename):
            os.unlink(filename)

# Run the test
print("Running Hypothesis test...")
print("=" * 60)
test_wavfile_multichannel()