import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays


@given(arrays(
    dtype=np.complex128,
    shape=st.integers(min_value=1, max_value=50),
    elements=st.complex_numbers(allow_nan=False, allow_infinity=False, max_magnitude=1e6)
))
@settings(max_examples=500)
def test_hfft_ihfft_roundtrip(a):
    result = np.fft.ihfft(np.fft.hfft(a))
    assert np.allclose(result, a)


if __name__ == "__main__":
    test_hfft_ihfft_roundtrip()