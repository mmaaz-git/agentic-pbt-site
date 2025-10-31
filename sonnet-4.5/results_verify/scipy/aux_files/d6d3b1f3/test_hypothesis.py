import numpy as np
from hypothesis import given, strategies as st, settings
import scipy.signal.windows as windows


@given(st.integers(min_value=2, max_value=100))
@settings(max_examples=300)
def test_get_window_consistency(M):
    for window_name in ['hann', 'hamming', 'blackman', 'bartlett']:
        func = getattr(windows, window_name)

        direct_result = func(M, sym=True)
        get_window_result = windows.get_window(window_name, M, fftbins=True)

        assert np.allclose(direct_result, get_window_result, rtol=1e-10, atol=1e-10), \
            f"get_window('{window_name}', {M}) should match {window_name}({M})"

if __name__ == "__main__":
    test_get_window_consistency()