import numpy as np
import numpy.fft as fft
from hypothesis import given, strategies as st, settings


@st.composite
def complex_arrays(draw, min_size=1, max_size=100):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    real_part = draw(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                                      min_value=-1e6, max_value=1e6),
                             min_size=size, max_size=size))
    imag_part = draw(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                                      min_value=-1e6, max_value=1e6),
                             min_size=size, max_size=size))
    return np.array([complex(r, i) for r, i in zip(real_part, imag_part)])


@given(complex_arrays(min_size=1, max_size=100))
@settings(max_examples=500)
def test_rfft_discards_imaginary(a):
    real_part_only = a.real
    result_complex = fft.rfft(a)
    result_real = fft.rfft(real_part_only)
    assert np.allclose(result_complex, result_real, rtol=1e-10, atol=1e-12), \
        f"rfft doesn't discard imaginary part as claimed"

if __name__ == "__main__":
    test_rfft_discards_imaginary()
    print("Test passed!")