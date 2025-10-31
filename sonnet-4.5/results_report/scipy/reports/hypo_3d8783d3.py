import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy import signal


def array_1d_strategy():
    return st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20
    ).map(np.array)


@settings(max_examples=200, deadline=5000)
@given(array_1d_strategy(), array_1d_strategy())
def test_deconvolve_property(signal_arr, divisor):
    assume(len(divisor) > 0 and len(signal_arr) >= len(divisor))
    assume(np.max(np.abs(divisor)) > 0.1)

    quotient, remainder = signal.deconvolve(signal_arr, divisor)

    reconstructed = signal.convolve(divisor, quotient, mode='full') + remainder
    trimmed_reconstructed = reconstructed[:len(signal_arr)]

    assert np.allclose(trimmed_reconstructed, signal_arr, rtol=1e-6, atol=1e-10)

if __name__ == "__main__":
    test_deconvolve_property()