import numpy as np
import scipy.fftpack as fftpack
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis import strategies as st

@given(arrays(dtype=np.float64, shape=st.integers(min_value=1, max_value=100)))
def test_hilbert_ihilbert_roundtrip(x):
    result = fftpack.ihilbert(fftpack.hilbert(x))
    assert np.allclose(result, x, rtol=1e-10, atol=1e-12)

# Run the test
try:
    test_hilbert_ihilbert_roundtrip()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed with assertion error")
except Exception as e:
    print(f"Test failed with error: {e}")