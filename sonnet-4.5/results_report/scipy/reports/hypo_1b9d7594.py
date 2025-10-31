from hypothesis import given, strategies as st
import numpy as np
import scipy.fft


@given(
    st.lists(
        st.floats(
            allow_nan=False,
            allow_infinity=False,
            min_value=-1e10,
            max_value=1e10
        ),
        min_size=1,
        max_size=1000
    )
)
def test_rfft_irfft_round_trip(data):
    x = np.array(data)
    result = scipy.fft.irfft(scipy.fft.rfft(x))
    assert np.allclose(result, x, rtol=1e-10, atol=1e-12)

if __name__ == "__main__":
    test_rfft_irfft_round_trip()