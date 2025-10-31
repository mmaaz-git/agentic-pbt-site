from hypothesis import given, settings, assume, strategies as st
import numpy as np
import scipy.signal

def test_deconvolve_inverse_property_bare(signal, divisor):
    signal_arr = np.array(signal, dtype=np.float64)
    divisor_arr = np.array(divisor, dtype=np.float64)

    if np.abs(divisor_arr[0]) <= 1e-10:
        return  # Skip if divisor[0] is too small

    quotient, remainder = scipy.signal.deconvolve(signal_arr, divisor_arr)

    reconstructed = scipy.signal.convolve(divisor_arr, quotient, mode='full')
    reconstructed_with_remainder = reconstructed.copy()
    min_len = min(len(reconstructed_with_remainder), len(remainder))
    reconstructed_with_remainder[:min_len] += remainder[:min_len]

    expected_len = len(signal_arr)
    reconstructed_trimmed = reconstructed_with_remainder[:expected_len]

    np.testing.assert_allclose(signal_arr, reconstructed_trimmed, rtol=1e-5, atol=1e-8)

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

# Test with the specific failing case
if __name__ == "__main__":
    # Run the specific failing case
    signal = [65.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.875]
    divisor = [0.125, 0.0, 2.0]

    try:
        test_deconvolve_inverse_property_bare(signal, divisor)
        print("Test passed for the specific case")
    except AssertionError as e:
        print(f"Test failed for the specific case: {e}")