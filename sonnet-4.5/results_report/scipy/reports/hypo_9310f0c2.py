import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings
from scipy import signal


@settings(max_examples=1000)
@given(
    original_signal=st.lists(st.floats(allow_nan=False, allow_infinity=False,
                                        min_value=-1e6, max_value=1e6),
                             min_size=1, max_size=50),
    divisor=st.lists(st.floats(allow_nan=False, allow_infinity=False,
                                min_value=-1e6, max_value=1e6),
                     min_size=1, max_size=50)
)
def test_deconvolve_convolve_roundtrip(original_signal, divisor):
    original_signal = np.array(original_signal)
    divisor = np.array(divisor)

    assume(np.abs(divisor).max() > 1e-10)
    assume(np.abs(divisor[0]) > 1e-10)

    recorded = signal.convolve(divisor, original_signal)
    quotient, remainder = signal.deconvolve(recorded, divisor)
    reconstructed = signal.convolve(divisor, quotient) + remainder

    assert reconstructed.shape == recorded.shape
    assert np.allclose(reconstructed, recorded, rtol=1e-8, atol=1e-10)

if __name__ == "__main__":
    test_deconvolve_convolve_roundtrip()