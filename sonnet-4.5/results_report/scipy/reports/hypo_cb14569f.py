from hypothesis import given, strategies as st, settings, assume
import numpy as np
import scipy.signal


@given(
    signal=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=50),
    divisor=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=30),
)
@settings(max_examples=500)
def test_deconvolve_round_trip(signal, divisor):
    assume(len(divisor) <= len(signal))
    assume(any(abs(d) > 1e-6 for d in divisor))

    signal_arr = np.array(signal)
    divisor_arr = np.array(divisor)

    quotient, remainder = scipy.signal.deconvolve(signal_arr, divisor_arr)

    reconstructed = scipy.signal.convolve(divisor_arr, quotient) + remainder

    np.testing.assert_allclose(reconstructed, signal_arr, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    # Run the test
    test_deconvolve_round_trip()