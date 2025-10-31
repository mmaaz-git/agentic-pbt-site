from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.signal.windows as windows

@given(
    M=st.integers(min_value=2, max_value=20),
    beta=st.floats(min_value=100.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=30)
def test_kaiser_very_large_beta(M, beta):
    window = windows.kaiser(M, beta, sym=True)

    assert len(window) == M
    assert np.all(np.isfinite(window)), \
        f"Kaiser window should have finite values even for large beta"

if __name__ == "__main__":
    test_kaiser_very_large_beta()
    print("All tests passed!")