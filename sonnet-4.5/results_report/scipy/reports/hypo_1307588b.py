from hypothesis import given, settings, assume, strategies as st
import numpy as np
import scipy.signal

@settings(max_examples=200)
@given(
    signal=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=10, max_size=50),
    divisor=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=3, max_size=10)
)
def test_deconvolve_inverse_property(signal, divisor):
    signal_arr = np.array(signal, dtype=np.float64)
    divisor_arr = np.array(divisor, dtype=np.float64)

    assume(np.abs(divisor_arr[0]) > 1e-10)

    quotient, remainder = scipy.signal.deconvolve(signal_arr, divisor_arr)

    reconstructed = scipy.signal.convolve(divisor_arr, quotient, mode='full')
    reconstructed_with_remainder = reconstructed.copy()
    min_len = min(len(reconstructed_with_remainder), len(remainder))
    reconstructed_with_remainder[:min_len] += remainder[:min_len]

    expected_len = len(signal_arr)
    reconstructed_trimmed = reconstructed_with_remainder[:expected_len]

    np.testing.assert_allclose(signal_arr, reconstructed_trimmed, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    test_deconvolve_inverse_property()