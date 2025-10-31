import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

float_elements = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False
)

@given(arrays(dtype=np.float64, shape=st.integers(min_value=1, max_value=100), elements=float_elements))
def test_rfft_irfft_no_crash(a):
    rfft_result = np.fft.rfft(a)
    result = np.fft.irfft(rfft_result)
    assert len(result) > 0

if __name__ == "__main__":
    test_rfft_irfft_no_crash()