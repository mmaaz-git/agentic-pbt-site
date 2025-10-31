from hypothesis import given, strategies as st, settings, assume
import numpy as np
import scipy.signal as signal

@given(
    signal_array=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100),
    divisor_array=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100)
)
@settings(max_examples=500)
def test_deconvolve_round_trip(signal_array, divisor_array):
    """
    Property: deconvolve docstring states:
    "Returns the quotient and remainder such that
    signal = convolve(divisor, quotient) + remainder"
    """
    signal_arr = np.array(signal_array)
    divisor_arr = np.array(divisor_array)

    assume(np.any(np.abs(divisor_arr) > 1e-10))

    quotient, remainder = signal.deconvolve(signal_arr, divisor_arr)
    reconstructed = signal.convolve(divisor_arr, quotient, mode='full')

    if len(remainder) > 0:
        remainder_padded = np.pad(remainder, (0, len(reconstructed) - len(remainder)), mode='constant')
        reconstructed = reconstructed + remainder_padded

    min_len = min(len(signal_arr), len(reconstructed))
    assert np.allclose(signal_arr[:min_len], reconstructed[:min_len], rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_deconvolve_round_trip()
    print("Test completed.")