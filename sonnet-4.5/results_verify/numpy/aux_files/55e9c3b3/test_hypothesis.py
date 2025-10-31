import numpy as np
import numpy.fft
from hypothesis import given, strategies as st, settings


@given(st.floats(allow_nan=False, allow_infinity=False,
                min_value=-1e6, max_value=1e6))
@settings(max_examples=100)
def test_hfft_single_element_crash(value):
    a = np.array([value])
    result = numpy.fft.hfft(a)
    print(f"Test passed with value {value}: result = {result}")

if __name__ == "__main__":
    test_hfft_single_element_crash()